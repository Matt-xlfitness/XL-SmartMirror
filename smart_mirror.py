#!/usr/bin/env python3
"""
XL Fitness Smart Mirror
=======================
A fullscreen "smart mirror" kiosk for the XL Fitness gym.

A live camera feed is shown fullscreen. A pose-detection model finds people in
frame and watches for a *double bicep flex*. An on-screen avatar greets each
person who steps up, prompts them to strike the pose, and celebrates when they
nail it.

Hardware target
---------------
  Raspberry Pi 5 + Hailo-8L AI Hat (NPU), USB webcam mounted low pointing up,
  landscape TV over HDMI (kiosk, no keyboard/mouse).

Pose backends (auto-selected, best first)
----------------------------------------
  1. Hailo  NPU   — hailo_platform SDK + a pose HEF        (fastest)
  2. MediaPipe    — mediapipe Pose Lite on CPU             (Python 3.11 venv)
  3. MoveNet      — tflite-runtime MoveNet Lightning       (always works)

The camera is mounted at floor level pointing up, so the lower body is often out
of frame. All detection therefore relies on the UPPER body only
(shoulders / elbows / wrists / nose).

Run
---
    cd ~/XL-SmartMirror && python3 smart_mirror.py
    DISPLAY=:0 python3 smart_mirror.py     # over SSH

Press Q or ESC to quit.
"""

import os
import sys
import time
import random

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

ASSET_FILES = {
    "logo":        "SMARTMIRROR.png",        # top-centre heading, always shown
    "wave":        "XLAvatar-Wave.png",      # Idle, Greeting
    "point":       "XLAvatar-Point.png",     # Prompt (pointing at viewer)
    "pose":        "XLAvatar-01Pose.png",    # Waiting (double bicep example)
    "celebrating": "XLAvatar-Celebrating.png",  # Celebrate
    "thumbsup":    "XLAvatar-ThumbsUp.png",  # Compliment, Done
}

MOVENET_FILE = os.path.join(ASSETS_DIR, "movenet_lightning.tflite")
# Optional Hailo pose HEF — drop one here to enable the NPU backend.
HAILO_HEF = os.path.join(ASSETS_DIR, "yolov8s_pose.hef")

# ── Tunable timings (seconds) ──────────────────────────────────────────────────
PRESENCE_CONFIRM   = 1.5   # person must persist this long before we greet
ABSENCE_CONFIRM    = 3.0   # frame must stay empty this long before we reset
POSE_CONFIRM       = 1.5   # flex must hold this long before we celebrate
GREETING_SECONDS   = 2.5   # "Welcome!" dwell before prompting
PROMPT_SECONDS     = 2.0   # "Can you do THIS?" dwell before showing example
CELEBRATE_SECONDS  = 3.5
COMPLIMENT_SECONDS = 3.0

# ── Look & feel ────────────────────────────────────────────────────────────────
AVATAR_HEIGHT_FRAC = 0.40   # avatar height as a fraction of screen height
LOGO_WIDTH_FRAC    = 0.27   # logo width as a fraction of screen width
AVATAR_MARGIN      = 24
SHOW_SKELETON      = True    # subtle skeleton while waiting / celebrating
INVERT_LOGO        = True    # logo art is dark; invert so it reads on the feed

HYPE_MSGS = [
    "BEAST MODE!", "ABSOLUTE UNIT!", "LETS GOOO!", "CHAMPION!",
    "UNSTOPPABLE!", "CRUSHING IT!", "PURE POWER!", "LEGENDARY!",
]

# ── COCO-17 keypoint indices (the shared format every backend emits) ────────────
# Each backend returns a list of persons; each person is an (17, 3) float array
# of [y_norm, x_norm, confidence], with y/x normalised 0..1 in the camera frame.
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW       = 7, 8
KP_L_WRIST, KP_R_WRIST       = 9, 10
KP_L_HIP, KP_R_HIP           = 11, 12

# Bones drawn for the (optional) skeleton overlay — upper body only.
SKELETON_BONES = [
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_NOSE, KP_L_SHOULDER), (KP_NOSE, KP_R_SHOULDER),
]


# ════════════════════════════════════════════════════════════════════════════════
#  Pose backends
# ════════════════════════════════════════════════════════════════════════════════
class PoseBackend:
    """Interface: infer(frame_bgr) -> list[np.ndarray(17, 3)]."""
    name = "base"

    def infer(self, frame_bgr):
        raise NotImplementedError


class MoveNetBackend(PoseBackend):
    """MoveNet Lightning via tflite-runtime. Single person, runs anywhere."""
    name = "MoveNet (TFLite)"

    def __init__(self, model_path=MOVENET_FILE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter  # fallback for dev machines
        self._interp = Interpreter(model_path=model_path, num_threads=4)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self._h = self._inp["shape"][1]
        self._w = self._inp["shape"][2]

    def infer(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._w, self._h))
        img = np.expand_dims(img, 0).astype(self._inp["dtype"])
        self._interp.set_tensor(self._inp["index"], img)
        self._interp.invoke()
        kp = self._interp.get_tensor(self._out["index"])[0][0]  # (17, 3)
        return [kp.astype(np.float32)]


class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose (Lite). Single person. Needs Python 3.11 venv."""
    name = "MediaPipe Pose"

    # MediaPipe landmark index -> COCO-17 index (only the joints we use).
    _MAP = {
        0: KP_NOSE,
        11: KP_L_SHOULDER, 12: KP_R_SHOULDER,
        13: KP_L_ELBOW,    14: KP_R_ELBOW,
        15: KP_L_WRIST,    16: KP_R_WRIST,
        23: KP_L_HIP,      24: KP_R_HIP,
    }

    def __init__(self):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            model_complexity=0,           # "Lite"
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return []
        kp = np.zeros((17, 3), np.float32)
        for mp_idx, coco_idx in self._MAP.items():
            lm = res.pose_landmarks.landmark[mp_idx]
            kp[coco_idx] = (lm.y, lm.x, lm.visibility)
        return [kp]


class HailoBackend(PoseBackend):
    """
    Hailo-8L NPU backend (yolov8 pose HEF) via the hailo_platform SDK.

    Multi-person and the fastest option, but the exact output decode depends on
    the HEF you compile (whether NMS + keypoint decode run on-chip). This is
    wired against the common "NMS-on-chip pose" HEF and is guarded so any
    mismatch falls back to MediaPipe/MoveNet. Finish-tune on-device.
    """
    name = "Hailo NPU"

    def __init__(self, hef_path=HAILO_HEF):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(hef_path)
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                    ConfigureParams, InputVStreamParams,
                                    OutputVStreamParams, InferVStreams)
        self._np = np
        self._hef = HEF(hef_path)
        self._target = VDevice()
        cfg = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._target.configure(self._hef, cfg)[0]
        self._ng_params = self._network_group.create_params()
        self._in_vp = InputVStreamParams.make(self._network_group)
        self._out_vp = OutputVStreamParams.make(self._network_group)
        vi = self._hef.get_input_vstream_infos()[0]
        self._in_name = vi.name
        self._h, self._w = vi.shape[0], vi.shape[1]
        self._InferVStreams = InferVStreams

    def infer(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._w, self._h))
        data = {self._in_name: np.expand_dims(img, 0).astype(np.uint8)}
        with self._InferVStreams(self._network_group, self._in_vp, self._out_vp) as pipe:
            with self._network_group.activate(self._ng_params):
                results = pipe.infer(data)
        return self._decode(results)

    def _decode(self, results):
        """Decode HEF output -> list[(17,3)]. Tuned per-HEF on-device."""
        persons = []
        for out in results.values():
            arr = np.array(out)
            # Expecting per-detection rows ending in 17*3 keypoint values.
            flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
            for row in flat:
                if row.shape[0] < 51:
                    continue
                kp_block = row[-51:].reshape(17, 3)          # [x, y, conf]
                kp = np.zeros((17, 3), np.float32)
                kp[:, 0] = kp_block[:, 1]                     # y
                kp[:, 1] = kp_block[:, 0]                     # x
                kp[:, 2] = kp_block[:, 2]                     # conf
                if kp[:, 2].max() > 0.1:
                    persons.append(kp)
        return persons


def select_backend():
    """Try Hailo -> MediaPipe -> MoveNet, returning the first that loads."""
    for cls in (HailoBackend, MediaPipeBackend, MoveNetBackend):
        try:
            backend = cls()
            print(f"✓ Pose backend: {backend.name}")
            return backend
        except Exception as e:
            print(f"  · {cls.name} unavailable: {e}")
    print("ERROR: no pose backend available.")
    print("  Install one of: hailo_platform / mediapipe / tflite-runtime")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════════
#  Pose analysis
# ════════════════════════════════════════════════════════════════════════════════
def has_upper_body(kp, threshold=0.30):
    """True if enough upper-body joints are confidently visible."""
    if kp is None:
        return False
    upper = (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW)
    return sum(1 for i in upper if kp[i][2] > threshold) >= 3


def is_double_bicep_flex(kp, threshold=0.20):
    """
    Lenient double-bicep detector (upper body only, flip-invariant).

    The core silhouette is: elbows spread at least as wide as the shoulders, and
    both wrists raised to roughly shoulder height or above. No strict angle
    checks — the camera looks up from the floor, so we keep it forgiving.
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
    if shoulder_span < 0.04:            # too small / sideways to judge
        return False

    elbow_span = abs(le_x - re_x)
    if elbow_span < shoulder_span * 1.05:   # elbows must flare out
        return False

    shoulder_mid_y = (ls_y + rs_y) / 2
    slack = shoulder_span * 0.40
    wrists_up = (lw_y <= shoulder_mid_y + slack) and (rw_y <= shoulder_mid_y + slack)
    return wrists_up


def any_present(persons):
    return any(has_upper_body(p) for p in persons)


def any_flexing(persons):
    return any(is_double_bicep_flex(p) for p in persons)


# ════════════════════════════════════════════════════════════════════════════════
#  Rendering helpers
# ════════════════════════════════════════════════════════════════════════════════
def overlay_png(bg, overlay, x, y):
    """Alpha-composite an RGBA overlay onto a BGR background at (x, y)."""
    if overlay is None:
        return
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    ox1, oy1 = x1 - x, y1 - y
    patch = overlay[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]
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
    return cv2.resize(img, (max(1, int(w * target_h / h)), target_h),
                      interpolation=cv2.INTER_AREA)


def resize_to_w(img, target_w):
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (target_w, max(1, int(h * target_w / w))),
                      interpolation=cv2.INTER_AREA)


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
    """Translucent speech bubble centred horizontally on cx, baseline at cy."""
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, f, scale, thickness)
    pad = 24
    x1, y1 = cx - tw // 2 - pad, cy - th - pad
    x2, y2 = cx + tw // 2 + pad, cy + bl + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text_shadow(frame, text, cx - tw // 2, cy, scale, (255, 255, 255), thickness)


def draw_skeleton(frame, persons, rainbow=False, t=0.0, thr=0.30):
    h, w = frame.shape[:2]
    n = len(SKELETON_BONES)
    for kp in persons:
        for idx, (i, j) in enumerate(SKELETON_BONES):
            ya, xa, ca = kp[i]
            yb, xb, cb = kp[j]
            if ca < thr or cb < thr:
                continue
            c = hsv_to_bgr(((idx / n) * 360 + t * 400) % 360, 1, 1) if rainbow else (51, 87, 255)
            cv2.line(frame, (int(xa * w), int(ya * h)), (int(xb * w), int(yb * h)),
                     c, 5, cv2.LINE_AA)
        for i in (KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW,
                  KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST):
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

        # Presence / absence hysteresis
        if present:
            self.absent_since = None
            if self.present_since is None:
                self.present_since = now
        else:
            self.present_since = None
            if self.absent_since is None:
                self.absent_since = now

        # Flex hold
        if flexing:
            if self.flex_since is None:
                self.flex_since = now
        else:
            self.flex_since = None

        confirmed_present = self.present_since is not None and (now - self.present_since) >= PRESENCE_CONFIRM
        confirmed_absent  = self.absent_since is not None and (now - self.absent_since) >= ABSENCE_CONFIRM
        confirmed_flex    = self.flex_since is not None and (now - self.flex_since) >= POSE_CONFIRM
        elapsed = now - self.entered

        s = self.state
        if s == IDLE:
            if confirmed_present:
                self._go(GREETING, now)
        elif s == GREETING:
            if confirmed_absent:
                self._go(IDLE, now)
            elif elapsed >= GREETING_SECONDS:
                self._go(PROMPT, now)
        elif s == PROMPT:
            if confirmed_absent:
                self._go(IDLE, now)
            elif confirmed_flex:
                self._go(CELEBRATE, now)
            elif elapsed >= PROMPT_SECONDS:
                self._go(WAITING, now)
        elif s == WAITING:
            if confirmed_absent:
                self._go(IDLE, now)
            elif confirmed_flex:
                self._go(CELEBRATE, now)
        elif s == CELEBRATE:
            if elapsed >= CELEBRATE_SECONDS:
                self._go(COMPLIMENT, now)
        elif s == COMPLIMENT:
            if elapsed >= COMPLIMENT_SECONDS:
                self._go(DONE, now)
        elif s == DONE:
            if confirmed_absent:
                self._go(IDLE, now)


# ════════════════════════════════════════════════════════════════════════════════
#  Asset loading
# ════════════════════════════════════════════════════════════════════════════════
def load_assets():
    assets = {}
    missing = []
    for key, fname in ASSET_FILES.items():
        path = os.path.join(ASSETS_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            missing.append(fname)
        assets[key] = img
    if missing:
        print(f"WARNING: missing assets in {ASSETS_DIR}: {', '.join(missing)}")
    return assets


def open_camera(width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


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


# ════════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("  XL Fitness Smart Mirror")
    print("=" * 50)

    backend = select_backend()
    assets = load_assets()

    print("Opening camera...")
    cap = open_camera()
    if not cap.isOpened():
        print("ERROR: cannot open camera.")
        sys.exit(1)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {cam_w}x{cam_h}")

    screen_w, screen_h = detect_screen_size()
    print(f"✓ Display: {screen_w}x{screen_h}")

    # Pre-scale avatars (height-based) and logo (width-based).
    av_h = int(screen_h * AVATAR_HEIGHT_FRAC)
    avatars = {k: resize_to_h(assets.get(k), av_h)
               for k in ("wave", "point", "pose", "celebrating", "thumbsup")}
    logo = resize_to_w(assets.get("logo"), int(screen_w * LOGO_WIDTH_FRAC))
    if INVERT_LOGO:
        logo = invert_rgb(logo)

    # Camera -> screen cover-scale (crop to fill, no letterboxing).
    s = max(screen_w / cam_w, screen_h / cam_h)
    sw, sh = int(cam_w * s), int(cam_h * s)
    ox, oy = (sw - screen_w) // 2, (sh - screen_h) // 2

    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mirror = Mirror()
    fps_t, fps_count, fps_disp = time.time(), 0, 0
    last_good = time.time()
    print("\n✓ Running — strike a double bicep flex! Press Q to quit.\n")

    while True:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[mirror] camera read error: {e}")
            ret, frame = False, None

        if not ret or frame is None:
            if time.time() - last_good > 3.0:
                print("[mirror] camera stalled — reopening...")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = open_camera()
                last_good = time.time()
            time.sleep(0.05)
            continue
        last_good = time.time()

        frame = cv2.flip(frame, 1)          # mirror image
        now = time.time()

        try:
            persons = backend.infer(frame)
        except Exception as e:
            print(f"[mirror] inference error: {e}")
            persons = []

        mirror.update(persons, now)
        state = mirror.state

        # Build the display frame (camera, cover-scaled & cropped to screen).
        display = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        display = display[oy:oy + screen_h, ox:ox + screen_w]

        celebrating = (state == CELEBRATE)
        if SHOW_SKELETON and state in (WAITING, CELEBRATE):
            draw_skeleton(display, persons, rainbow=celebrating, t=now)

        # Logo, always top-centre.
        if logo is not None:
            overlay_png(display, logo, (screen_w - logo.shape[1]) // 2, 20)

        # Per-state avatar + copy.
        if state == CELEBRATE:
            hue = (now * 200) % 360
            pulse = 1.0 + 0.15 * abs(np.sin((now - mirror.entered) * 6))
            text_centred(display, mirror.hype, int(screen_h * 0.46),
                         scale=4.5 * pulse, color=hsv_to_bgr(hue, 1, 1), thickness=8)
            _draw_avatar(display, avatars.get("celebrating"), screen_w, screen_h)
        else:
            avatar_key, text = _state_avatar_and_text(state)
            av = avatars.get(avatar_key)
            _draw_avatar(display, av, screen_w, screen_h)
            if text and av is not None:
                ax = screen_w - av.shape[1] - AVATAR_MARGIN
                draw_bubble(display, text, ax + av.shape[1] // 2,
                            screen_h - av.shape[0] - 20)

        # FPS counter (small, bottom-left).
        fps_count += 1
        if now - fps_t >= 1.0:
            fps_disp, fps_count, fps_t = fps_count, 0, now
        cv2.putText(display, f"{fps_disp}fps {STATE_NAMES[state]}",
                    (10, screen_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        try:
            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
        except Exception as e:
            print(f"[mirror] display error: {e}")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


def _state_avatar_and_text(state):
    if state == IDLE:
        return "wave", "Step up & strike a pose!"
    if state == GREETING:
        return "wave", "Hey! Welcome to XL Fitness!"
    if state == PROMPT:
        return "point", "Can you do THIS? Strike a pose!"
    if state == WAITING:
        return "pose", "Flex like this! \U0001F4AA"
    if state == COMPLIMENT:
        return "thumbsup", "You look incredible! \U0001F4AA"
    if state == DONE:
        return "thumbsup", ""
    return "wave", ""


def _draw_avatar(display, av, screen_w, screen_h):
    """Bottom-right, anchored flush to the bottom edge."""
    if av is None:
        return
    ax = screen_w - av.shape[1] - AVATAR_MARGIN
    ay = screen_h - av.shape[0]
    overlay_png(display, av, ax, ay)


if __name__ == "__main__":
    # Watchdog: keep the wall display alive if main() throws unexpectedly.
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            print("\nInterrupted — bye.")
            break
        except Exception as e:
            print(f"\n[mirror] FATAL: {e}")
            print("[mirror] restarting in 3s...")
            time.sleep(3)
