#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 5 + Hailo-8 NPU
======================================================
A fullscreen "smart mirror" kiosk for the XL Fitness gym.

A live camera feed runs fullscreen on a wall-mounted TV. A YOLOv8-pose model on
the Hailo-8 NPU finds people in frame and watches for a *double bicep flex*. An
on-screen avatar greets each person who steps up, prompts them to strike the
pose, and celebrates when they nail it.

Smoothness comes from a 3-part pipeline so nothing blocks the screen:
    [Camera thread]    -> latest frame
    [Inference thread] -> latest keypoints   (Hailo, network activated once)
    [Render loop]      -> draws newest frame + newest keypoints @ display rate

Camera is mounted low pointing up, so detection uses the UPPER body only
(shoulders / elbows / wrists / nose).

Run (on the Pi's display):
    cd ~/Desktop/XL-SmartMirror && python3 smart_mirror.py

Validate the pose decoder on a still image (no camera needed):
    python3 smart_mirror.py --image somebody_flexing.jpg
    # writes selftest_out.jpg next to the script

Press Q or ESC to quit.
"""

import os
import sys
import time
import random
import argparse
import threading

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

# Liveness heartbeat — the render loop refreshes this file every ~2s. The daily
# health check restarts the service if it goes stale (frozen but "running").
HEARTBEAT_FILE = os.path.expanduser("~/.cache/xl-mirror/heartbeat")

ASSET_FILES = {
    "logo":        "SMARTMIRROR.png",
    "wave":        "XLAvatar-Wave.png",
    "point":       "XLAvatar-Point.png",
    "pose":        "XLAvatar-01Pose.png",
    "celebrating": "XLAvatar-Celebrating.png",
    "thumbsup":    "XLAvatar-ThumbsUp.png",
}

# Hailo pose HEFs shipped in /usr/share/hailo-models. First that configures wins
# (h8 for Hailo-8, h8l for Hailo-8L).
HEF_CANDIDATES = [
    "/usr/share/hailo-models/yolov8s_pose_h8.hef",
    "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef",
]

# ── Tunable timings (seconds) ──────────────────────────────────────────────────
PRESENCE_CONFIRM   = 1.5
ABSENCE_CONFIRM    = 3.0
POSE_CONFIRM       = 1.5
GREETING_SECONDS   = 2.5
PROMPT_SECONDS     = 2.0
CELEBRATE_SECONDS  = 3.5
COMPLIMENT_SECONDS = 3.0

# ── Detection thresholds ───────────────────────────────────────────────────────
PERSON_CONF = 0.50    # min person score to keep a detection
KPT_CONF    = 0.30    # min keypoint confidence to use/draw a joint
NMS_IOU     = 0.45
MAX_PERSONS = 5

# ── Look & feel ────────────────────────────────────────────────────────────────
AVATAR_HEIGHT_FRAC = 0.40
LOGO_WIDTH_FRAC    = 0.27
AVATAR_MARGIN      = 24
SHOW_SKELETON      = True
INVERT_LOGO        = True
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30

HYPE_MSGS = [
    "BEAST MODE!", "ABSOLUTE UNIT!", "LETS GOOO!", "CHAMPION!",
    "UNSTOPPABLE!", "CRUSHING IT!", "PURE POWER!", "LEGENDARY!",
]

# ── COCO-17 keypoint indices (the format the decoder emits) ─────────────────────
# Each person is an (17, 3) float array of [y_norm, x_norm, confidence].
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW       = 7, 8
KP_L_WRIST, KP_R_WRIST       = 9, 10
KP_L_HIP, KP_R_HIP           = 11, 12

SKELETON_BONES = [
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_NOSE, KP_L_SHOULDER), (KP_NOSE, KP_R_SHOULDER),
]
UPPER_DRAW = (KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW,
              KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST)


# ════════════════════════════════════════════════════════════════════════════════
#  Math helpers
# ════════════════════════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def nms(boxes, scores, iou_thr=NMS_IOU):
    """Plain greedy NMS. boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


# ════════════════════════════════════════════════════════════════════════════════
#  Hailo YOLOv8-pose backend
# ════════════════════════════════════════════════════════════════════════════════
class HailoPoseBackend:
    """
    Raw YOLOv8-pose on the Hailo NPU via HailoRT (hailo_platform).

    The HEF emits per-scale conv tensors (box DFL / class score / keypoints) with
    no on-chip NMS, so we decode them here. The network group is activated ONCE
    and the InferVStreams pipe is kept open for the life of the object — this is
    what keeps inference fast. Construct and use it from a single thread.
    """
    name = "Hailo YOLOv8-pose"

    def __init__(self, hef_candidates=HEF_CANDIDATES):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                    ConfigureParams, InputVStreamParams,
                                    OutputVStreamParams, InferVStreams, FormatType)
        self._InferVStreams = InferVStreams
        self._vdevice = VDevice()

        last_err = None
        self._ng = None
        for hp in hef_candidates:
            if not os.path.exists(hp):
                continue
            try:
                hef = HEF(hp)
                cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
                self._ng = self._vdevice.configure(hef, cfg)[0]
                self._hef = hef
                self.hef_path = hp
                break
            except Exception as e:
                last_err = e
        if self._ng is None:
            raise RuntimeError(f"No usable pose HEF (last error: {last_err})")

        in_info = self._hef.get_input_vstream_infos()[0]
        self._in_name = in_info.name
        self.H, self.W, _ = in_info.shape          # 640, 640, 3

        self._ng_params = self._ng.create_params()
        self._ivp = InputVStreamParams.make(self._ng, format_type=FormatType.UINT8)
        self._ovp = OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)

        # Keep activation + pipe open for the object's lifetime (perf-critical).
        self._act_ctx = self._ng.activate(self._ng_params)
        self._act_ctx.__enter__()
        self._pipe_ctx = self._InferVStreams(self._ng, self._ivp, self._ovp)
        self._pipe = self._pipe_ctx.__enter__()

        # Map grid size -> stride (640/grid): 80->8, 40->16, 20->32
        self._strides = {self.H // 8: 8, self.H // 16: 16, self.H // 32: 32}

    def close(self):
        try:
            self._pipe_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._act_ctx.__exit__(None, None, None)
        except Exception:
            pass

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.W / w, self.H / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.H, self.W, 3), 114, np.uint8)
        px, py = (self.W - nw) // 2, (self.H - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, px, py, nw, nh

    def infer(self, frame_bgr):
        canvas, px, py, nw, nh = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res = self._pipe.infer({self._in_name: rgb[None].astype(np.uint8)})
        return self._decode(res, px, py, nw, nh)

    def _decode(self, res, px, py, nw, nh):
        # Group tensors per scale by channel count.
        scales = {}
        for name, arr in res.items():
            a = np.array(arr)[0]                 # (H, W, C) NHWC float32
            grid, c = a.shape[0], a.shape[2]
            d = scales.setdefault(grid, {})
            if c == 64:
                d["box"] = a
            elif c == 1:
                d["cls"] = a
            elif c == 51:
                d["kpt"] = a

        boxes, scores, kpts = [], [], []
        for grid, d in scales.items():
            if not all(k in d for k in ("box", "cls", "kpt")):
                continue
            stride = self._strides.get(grid, self.H / grid)
            cls = sigmoid(d["cls"][:, :, 0])      # (grid, grid)
            mask = cls > PERSON_CONF
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            sc = cls[ys, xs]

            # Box DFL decode -> ltrb distances (in cells)
            box_raw = d["box"][ys, xs].reshape(-1, 4, 16)
            dist = (softmax(box_raw, axis=2) * np.arange(16)).sum(axis=2)   # (N,4)
            ax, ay = xs + 0.5, ys + 0.5
            x1 = (ax - dist[:, 0]) * stride
            y1 = (ay - dist[:, 1]) * stride
            x2 = (ax + dist[:, 2]) * stride
            y2 = (ay + dist[:, 3]) * stride

            # Keypoint decode (ultralytics): (k*2 + col)*stride, conf = sigmoid
            kp = d["kpt"][ys, xs].reshape(-1, 17, 3)
            kx = (kp[:, :, 0] * 2.0 + xs[:, None]) * stride
            ky = (kp[:, :, 1] * 2.0 + ys[:, None]) * stride
            kc = sigmoid(kp[:, :, 2])

            for i in range(len(sc)):
                boxes.append((x1[i], y1[i], x2[i], y2[i]))
                scores.append(sc[i])
                kpts.append(np.stack([kx[i], ky[i], kc[i]], axis=1))   # (17,3) px

        if not boxes:
            return []
        keep = nms(np.array(boxes, np.float32), np.array(scores, np.float32))

        persons = []
        for i in keep[:MAX_PERSONS]:
            k = kpts[i]
            out = np.zeros((17, 3), np.float32)
            out[:, 0] = (k[:, 1] - py) / nh        # y_norm (letterbox inverse)
            out[:, 1] = (k[:, 0] - px) / nw        # x_norm
            out[:, 2] = k[:, 2]
            persons.append(out)
        return persons


# ════════════════════════════════════════════════════════════════════════════════
#  Pose analysis
# ════════════════════════════════════════════════════════════════════════════════
def has_upper_body(kp, threshold=KPT_CONF):
    if kp is None:
        return False
    upper = (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW)
    return sum(1 for i in upper if kp[i][2] > threshold) >= 3


def is_double_bicep_flex(kp, threshold=0.20):
    """Lenient, flip-invariant double-bicep detector (upper body only).

    Core silhouette: elbows spread at least as wide as the shoulders, and both
    wrists raised to roughly shoulder height or above.
    """
    if kp is None:
        return False
    ls_y, ls_x, ls_c = kp[KP_L_SHOULDER]
    rs_y, rs_x, rs_c = kp[KP_R_SHOULDER]
    le_y, le_x, le_c = kp[KP_L_ELBOW]
    re_y, re_x, re_c = kp[KP_R_ELBOW]
    lw_y, lw_x, lw_c = kp[KP_L_WRIST]
    rw_y, rw_x, rw_c = kp[KP_R_WRIST]
    if min(ls_c, rs_c, le_c, re_c, lw_c, rw_c) < threshold:
        return False
    shoulder_span = abs(ls_x - rs_x)
    if shoulder_span < 0.04:
        return False
    if abs(le_x - re_x) < shoulder_span * 1.05:    # elbows must flare out
        return False
    shoulder_mid_y = (ls_y + rs_y) / 2
    slack = shoulder_span * 0.40
    return (lw_y <= shoulder_mid_y + slack) and (rw_y <= shoulder_mid_y + slack)


def any_present(persons):
    return any(has_upper_body(p) for p in persons)


def any_flexing(persons):
    return any(is_double_bicep_flex(p) for p in persons)


# ════════════════════════════════════════════════════════════════════════════════
#  Rendering helpers
# ════════════════════════════════════════════════════════════════════════════════
def overlay_png(bg, overlay, x, y):
    if overlay is None:
        return
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    patch = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    roi = bg[y1:y2, x1:x2]
    if overlay.shape[2] == 4:
        a = patch[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (patch[:, :, :3] * a + roi * (1 - a)).clip(0, 255).astype(np.uint8)
    else:
        roi[:] = patch[:, :, :3]


def resize_to_h(img, target_h):
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * target_h / h)), target_h), interpolation=cv2.INTER_AREA)


def resize_to_w(img, target_w):
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (target_w, max(1, int(h * target_w / w))), interpolation=cv2.INTER_AREA)


def invert_rgb(img):
    if img is None:
        return None
    out = img.copy()
    out[:, :, :3] = 255 - out[:, :, :3]
    return out


def hsv_to_bgr(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:    r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def text_shadow(frame, text, x, y, scale, color, thickness):
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x + 3, y + 3), f, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), f, scale, color, thickness, cv2.LINE_AA)


def text_centred(frame, text, cy, scale, color, thickness):
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(text, f, scale, thickness)
    text_shadow(frame, text, (frame.shape[1] - tw) // 2, cy, scale, color, thickness)


def draw_bubble(frame, text, cx, cy, scale=1.2, thickness=2):
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, f, scale, thickness)
    pad = 24
    x1, y1 = max(0, cx - tw // 2 - pad), max(0, cy - th - pad)
    x2, y2 = min(frame.shape[1] - 1, cx + tw // 2 + pad), min(frame.shape[0] - 1, cy + bl + pad)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text_shadow(frame, text, cx - tw // 2, cy, scale, (255, 255, 255), thickness)


def draw_skeleton(frame, persons, rainbow=False, t=0.0, thr=KPT_CONF):
    h, w = frame.shape[:2]
    n = len(SKELETON_BONES)
    for kp in persons:
        for idx, (i, j) in enumerate(SKELETON_BONES):
            ya, xa, ca = kp[i]
            yb, xb, cb = kp[j]
            if ca < thr or cb < thr:
                continue
            c = hsv_to_bgr(((idx / n) * 360 + t * 400) % 360, 1, 1) if rainbow else (51, 87, 255)
            cv2.line(frame, (int(xa * w), int(ya * h)), (int(xb * w), int(yb * h)), c, 5, cv2.LINE_AA)
        for i in UPPER_DRAW:
            y, x, c = kp[i]
            if c < thr:
                continue
            col = hsv_to_bgr(((i / 17) * 360 + t * 400) % 360, 1, 1) if rainbow else (51, 87, 255)
            cv2.circle(frame, (int(x * w), int(y * h)), 7, col, -1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════════
#  State machine
# ════════════════════════════════════════════════════════════════════════════════
IDLE, GREETING, PROMPT, WAITING, CELEBRATE, COMPLIMENT, DONE = range(7)
STATE_NAMES = ["IDLE", "GREETING", "PROMPT", "WAITING", "CELEBRATE", "COMPLIMENT", "DONE"]


class Mirror:
    def __init__(self):
        self.state = IDLE
        self.entered = time.time()
        self.present_since = None
        self.absent_since = None
        self.flex_since = None
        self.hype = ""

    def _go(self, state, now):
        self.state = state
        self.entered = now
        if state == CELEBRATE:
            self.hype = random.choice(HYPE_MSGS)

    def update(self, persons, now):
        present = any_present(persons)
        flexing = any_flexing(persons)

        if present:
            self.absent_since = None
            if self.present_since is None:
                self.present_since = now
        else:
            self.present_since = None
            if self.absent_since is None:
                self.absent_since = now
        self.flex_since = now if (flexing and self.flex_since is None) else \
            (None if not flexing else self.flex_since)

        conf_present = self.present_since is not None and (now - self.present_since) >= PRESENCE_CONFIRM
        conf_absent  = self.absent_since is not None and (now - self.absent_since) >= ABSENCE_CONFIRM
        conf_flex    = self.flex_since is not None and (now - self.flex_since) >= POSE_CONFIRM
        elapsed = now - self.entered

        s = self.state
        if s == IDLE:
            if conf_present:
                self._go(GREETING, now)
        elif s == GREETING:
            if conf_absent:
                self._go(IDLE, now)
            elif elapsed >= GREETING_SECONDS:
                self._go(PROMPT, now)
        elif s == PROMPT:
            if conf_absent:
                self._go(IDLE, now)
            elif conf_flex:
                self._go(CELEBRATE, now)
            elif elapsed >= PROMPT_SECONDS:
                self._go(WAITING, now)
        elif s == WAITING:
            if conf_absent:
                self._go(IDLE, now)
            elif conf_flex:
                self._go(CELEBRATE, now)
        elif s == CELEBRATE:
            if elapsed >= CELEBRATE_SECONDS:
                self._go(COMPLIMENT, now)
        elif s == COMPLIMENT:
            if elapsed >= COMPLIMENT_SECONDS:
                self._go(DONE, now)
        elif s == DONE:
            if conf_absent:
                self._go(IDLE, now)


def state_avatar_and_text(state):
    return {
        IDLE:       ("wave",     "Step up & strike a pose!"),
        GREETING:   ("wave",     "Hey! Welcome to XL Fitness!"),
        PROMPT:     ("point",    "Can you do THIS? Strike a pose!"),
        WAITING:    ("pose",     "Flex like this! \U0001F4AA"),
        COMPLIMENT: ("thumbsup", "You look incredible! \U0001F4AA"),
        DONE:       ("thumbsup", ""),
    }.get(state, ("wave", ""))


# ════════════════════════════════════════════════════════════════════════════════
#  Threads
# ════════════════════════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    """Continuously grabs frames; render loop always reads the newest."""
    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.index = index
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self.opened = False

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        cap = self._open()
        self.opened = cap.isOpened()
        last_good = time.time()
        while not self._stop.is_set():
            ok, frame = (False, None)
            try:
                ok, frame = cap.read()
            except Exception:
                ok = False
            if not ok or frame is None:
                if time.time() - last_good > 3.0:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = self._open()
                    last_good = time.time()
                time.sleep(0.03)
                continue
            last_good = time.time()
            frame = cv2.flip(frame, 1)               # mirror
            with self._lock:
                self._frame = frame
        try:
            cap.release()
        except Exception:
            pass

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()


class InferenceThread(threading.Thread):
    """Owns the Hailo backend; publishes the latest person keypoints."""
    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera
        self._lock = threading.Lock()
        self._persons = []
        self._stop = threading.Event()
        self.fps = 0
        self.error = None
        self.ready = threading.Event()

    def run(self):
        try:
            backend = HailoPoseBackend()
            print(f"✓ Pose backend: {backend.name}  ({os.path.basename(backend.hef_path)})")
        except Exception as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        n, t0 = 0, time.time()
        while not self._stop.is_set():
            frame = self.camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                persons = backend.infer(frame)
            except Exception as e:
                print(f"[mirror] inference error: {e}")
                persons = []
            with self._lock:
                self._persons = persons
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps, n, t0 = n, 0, time.time()
        backend.close()

    def latest(self):
        with self._lock:
            return list(self._persons)

    def stop(self):
        self._stop.set()


# ════════════════════════════════════════════════════════════════════════════════
#  Setup helpers
# ════════════════════════════════════════════════════════════════════════════════
def load_assets():
    assets, missing = {}, []
    for key, fname in ASSET_FILES.items():
        img = cv2.imread(os.path.join(ASSETS_DIR, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            missing.append(fname)
        assets[key] = img
    if missing:
        print(f"WARNING: missing assets: {', '.join(missing)}")
    return assets


def detect_screen_size(default=(1920, 1080)):
    try:
        import subprocess, re
        out = subprocess.check_output(["xrandr"], text=True)
        for line in out.splitlines():
            if " connected" in line:
                m = re.search(r"(\d+)x(\d+)\+", line)
                if m:
                    return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return default


def draw_avatar(display, av, screen_w, screen_h):
    if av is None:
        return
    overlay_png(display, av, screen_w - av.shape[1] - AVATAR_MARGIN, screen_h - av.shape[0])


# ════════════════════════════════════════════════════════════════════════════════
#  Static-image self-test (validate the decoder without a camera)
# ════════════════════════════════════════════════════════════════════════════════
def run_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: cannot read image {path}")
        sys.exit(1)
    backend = HailoPoseBackend()
    print(f"✓ {backend.name} ({os.path.basename(backend.hef_path)})  input {backend.W}x{backend.H}")
    persons = backend.infer(frame)
    print(f"Detected {len(persons)} person(s).")
    for i, kp in enumerate(persons):
        flex = is_double_bicep_flex(kp)
        vis = int((kp[:, 2] > KPT_CONF).sum())
        print(f"  person {i}: {vis}/17 kpts visible  flex={flex}")
    draw_skeleton(frame, persons)
    for kp in persons:
        for j in UPPER_DRAW:
            if kp[j][2] > KPT_CONF:
                cv2.circle(frame, (int(kp[j][1] * frame.shape[1]), int(kp[j][0] * frame.shape[0])),
                           4, (0, 255, 0), -1)
    out = os.path.join(SCRIPT_DIR, "selftest_out.jpg")
    cv2.imwrite(out, frame)
    print(f"Wrote {out} — open it to check the skeleton lands on the body.")
    backend.close()


# ════════════════════════════════════════════════════════════════════════════════
#  Live kiosk
# ════════════════════════════════════════════════════════════════════════════════
def run_live():
    print("=" * 50)
    print("  XL Fitness Smart Mirror")
    print("=" * 50)

    assets = load_assets()
    screen_w, screen_h = detect_screen_size()
    print(f"✓ Display: {screen_w}x{screen_h}")

    av_h = int(screen_h * AVATAR_HEIGHT_FRAC)
    avatars = {k: resize_to_h(assets.get(k), av_h)
               for k in ("wave", "point", "pose", "celebrating", "thumbsup")}
    logo = resize_to_w(assets.get("logo"), int(screen_w * LOGO_WIDTH_FRAC))
    if INVERT_LOGO:
        logo = invert_rgb(logo)

    camera = CameraThread(0)
    camera.start()
    infer = InferenceThread(camera)
    infer.start()
    infer.ready.wait(timeout=30)
    if infer.error is not None:
        print(f"ERROR: pose backend failed to start: {infer.error}")
        camera.stop()
        sys.exit(1)

    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mirror = Mirror()
    fps_t, fps_n, fps_disp = time.time(), 0, 0
    try:
        os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)
    except Exception:
        pass
    last_hb = 0.0
    print("\n✓ Running — strike a double bicep flex! Press Q to quit.\n")

    while True:
        now = time.time()
        if now - last_hb >= 2.0:               # liveness heartbeat
            try:
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(str(int(now)))
            except Exception:
                pass
            last_hb = now
        frame = camera.latest()
        persons = infer.latest()

        if frame is None:
            display = np.zeros((screen_h, screen_w, 3), np.uint8)
            text_centred(display, "Waiting for camera...", screen_h // 2, 1.5, (255, 255, 255), 2)
        else:
            display = cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            mirror.update(persons, now)
            state = mirror.state

            if SHOW_SKELETON and state in (WAITING, CELEBRATE):
                draw_skeleton(display, persons, rainbow=(state == CELEBRATE), t=now)
            if logo is not None:
                overlay_png(display, logo, (screen_w - logo.shape[1]) // 2, 20)

            if state == CELEBRATE:
                hue = (now * 200) % 360
                pulse = 1.0 + 0.15 * abs(np.sin((now - mirror.entered) * 6))
                text_centred(display, mirror.hype, int(screen_h * 0.46),
                             4.5 * pulse, hsv_to_bgr(hue, 1, 1), 8)
                draw_avatar(display, avatars.get("celebrating"), screen_w, screen_h)
            else:
                key, text = state_avatar_and_text(state)
                av = avatars.get(key)
                draw_avatar(display, av, screen_w, screen_h)
                if text and av is not None:
                    ax = screen_w - av.shape[1] - AVATAR_MARGIN
                    draw_bubble(display, text, ax + av.shape[1] // 2, screen_h - av.shape[0] - 20)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps_disp, fps_n, fps_t = fps_n, 0, now
            cv2.putText(display, f"render {fps_disp}fps  hailo {infer.fps}fps  {STATE_NAMES[state]}",
                        (10, screen_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        try:
            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
        except Exception as e:
            print(f"[mirror] display error: {e}")
            time.sleep(0.1)

    infer.stop()
    camera.stop()
    cv2.destroyAllWindows()
    print("Goodbye!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="run pose decode on a still image and exit")
    args = ap.parse_args()
    if args.image:
        run_image(args.image)
    else:
        run_live()


if __name__ == "__main__":
    main()
