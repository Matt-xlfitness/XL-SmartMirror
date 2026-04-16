#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 4
=========================================
Uses TFLite + MoveNet Lightning for fast pose detection on ARM.

Requirements:
    pip3 install tflite-runtime opencv-python numpy --break-system-packages

Usage:
    python3 smart_mirror.py

Press Q or ESC to quit.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import sys
import random

# ── Asset config ──────────────────────────────────────────────────────────────

GITHUB_RAW = "https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/assets"
ASSETS_DIR = os.path.expanduser("~/smart_mirror_assets")

ASSET_FILES = {
    "wave":        "XLAvatar-Wave.png",
    "point":       "XLAvatar-Point.png",
    "pose":        "XLAvatar-01Pose.png",
    "thumbsup":    "XLAvatar-ThumbsUp.png",
    "celebrating": "XLAvatar-Celebrating.png",
    "logo":        "SMARTMIRROR.png",
}

MOVENET_MODEL_URL = "https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/assets/movenet_lightning.tflite"
MOVENET_MODEL_FILE = "movenet_lightning.tflite"

# ── Hype content ──────────────────────────────────────────────────────────────

HYPE_MSGS = [
    "BEAST MODE!",
    "ABSOLUTE UNIT!",
    "LETS GOOO!",
    "CHAMPION!",
    "UNSTOPPABLE!",
    "CRUSHING IT!",
]

COMPLIMENTS = [
    "You look incredible!",
    "That's the one!",
    "Looking strong!",
    "Pure power right there!",
    "You're a legend!",
    "Absolutely nailed it!",
]

# ── MoveNet keypoint indices (COCO 17) ───────────────────────────────────────
# https://www.tensorflow.org/hub/tutorials/movenet
KP_NOSE          = 0
KP_L_EYE         = 1
KP_R_EYE         = 2
KP_L_EAR         = 3
KP_R_EAR         = 4
KP_L_SHOULDER    = 5
KP_R_SHOULDER    = 6
KP_L_ELBOW       = 7
KP_R_ELBOW       = 8
KP_L_WRIST       = 9
KP_R_WRIST       = 10
KP_L_HIP         = 11
KP_R_HIP         = 12
KP_L_KNEE        = 13
KP_R_KNEE        = 14
KP_L_ANKLE       = 15
KP_R_ANKLE       = 16

SKELETON_CONNECTIONS = [
    (KP_L_SHOULDER, KP_R_SHOULDER),                          # shoulders
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),  # left arm
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),  # right arm
    (KP_L_SHOULDER, KP_L_HIP),   (KP_R_SHOULDER, KP_R_HIP), # torso sides
    (KP_L_HIP, KP_R_HIP),                                    # hips
    (KP_L_HIP, KP_L_KNEE),       (KP_L_KNEE, KP_L_ANKLE),   # left leg
    (KP_R_HIP, KP_R_KNEE),       (KP_R_KNEE, KP_R_ANKLE),   # right leg
    (KP_NOSE, KP_L_SHOULDER),    (KP_NOSE, KP_R_SHOULDER),   # head
]

# ── Download helpers ──────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_if_missing(url, dest, label=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return True
    print(f"  Downloading {label or os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  ✓ {os.path.basename(dest)} ({size//1024}KB)")
        return size > 500
    except Exception as e:
        print(f"  ✗ Failed to download {label}: {e}")
        return False

def load_assets():
    ensure_dir(ASSETS_DIR)
    assets = {}
    print("Checking assets...")
    for key, filename in ASSET_FILES.items():
        dest = os.path.join(ASSETS_DIR, filename)
        url  = f"{GITHUB_RAW}/{filename}"
        if download_if_missing(url, dest, key):
            img = cv2.imread(dest, cv2.IMREAD_UNCHANGED)
            if img is not None:
                assets[key] = img
            else:
                print(f"  ✗ Could not read image: {filename}")
                assets[key] = None
        else:
            assets[key] = None
    return assets

def load_movenet_model():
    """Download and load MoveNet Lightning TFLite model."""
    model_path = os.path.join(ASSETS_DIR, MOVENET_MODEL_FILE)
    if not download_if_missing(MOVENET_MODEL_URL, model_path, "MoveNet model"):
        print("ERROR: Could not download MoveNet model.")
        sys.exit(1)

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        print("ERROR: tflite-runtime not installed.")
        print("Run: pip3 install tflite-runtime --break-system-packages")
        sys.exit(1)

    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

# ── MoveNet inference ─────────────────────────────────────────────────────────

def run_movenet(interpreter, frame):
    """
    Run MoveNet on a frame. Returns keypoints as (17, 3) array:
    each row is [y, x, confidence] in normalized 0-1 coords.
    Returns None if inference fails.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # [1, 192, 192, 3]
    input_h, input_w = input_shape[1], input_shape[2]

    # Resize and prepare input
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))

    if input_details[0]['dtype'] == np.uint8:
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    else:
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints[0][0]  # shape: (17, 3) — [y, x, confidence]

# ── Image overlay ─────────────────────────────────────────────────────────────

def overlay_png(bg, overlay, x, y):
    """Overlay RGBA PNG onto BGR background at (x,y). Clips to bounds."""
    if overlay is None:
        return
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    patch = overlay[oy1:oy2, ox1:ox2]
    roi   = bg[y1:y2, x1:x2]

    if overlay.shape[2] == 4:
        a = patch[:, :, 3:4].astype(np.float32) / 255.0
        rgb = patch[:, :, :3].astype(np.float32)
        roi_f = roi.astype(np.float32)
        blended = rgb * a + roi_f * (1.0 - a)
        roi[:] = blended.clip(0, 255).astype(np.uint8)
    else:
        roi[:] = patch[:, :, :3]

def resize_asset(img, target_h):
    if img is None:
        return None
    h, w = img.shape[:2]
    target_w = int(w * target_h / h)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

def invert_logo(img):
    """Invert black logo to white for dark background."""
    if img is None:
        return None
    result = img.copy()
    result[:, :, :3] = 255 - result[:, :, :3]
    return result

# ── Text helpers ──────────────────────────────────────────────────────────────

def put_text_shadow(frame, text, x, y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x+2, y+2), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),     font, scale, color,   thickness,   cv2.LINE_AA)

def put_text_centred(frame, text, cy, scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    put_text_shadow(frame, text, x, cy, scale, color, thickness)

def draw_bubble(frame, text, cx, cy, scale=0.85):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, 2)
    pad = 16
    x1, y1 = cx - tw//2 - pad, cy - th - pad
    x2, y2 = cx + tw//2 + pad, cy + bl + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1]-1, x2)
    y2 = min(frame.shape[0]-1, y2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (80,80,80), 1)
    put_text_shadow(frame, text, cx - tw//2, cy, scale, (255,255,255), 2)

# ── Skeleton drawing ──────────────────────────────────────────────────────────

def draw_skeleton(frame, keypoints, w, h, threshold=0.3):
    """Draw skeleton from MoveNet keypoints (17, 3) array."""
    if keypoints is None:
        return

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        ky_a, kx_a, kc_a = keypoints[i]
        ky_b, kx_b, kc_b = keypoints[j]
        if kc_a < threshold or kc_b < threshold:
            continue
        x1, y1 = int(kx_a * w), int(ky_a * h)
        x2, y2 = int(kx_b * w), int(ky_b * h)
        cv2.line(frame, (x1, y1), (x2, y2), (51, 87, 255), 3, cv2.LINE_AA)

    # Draw joints
    for i in range(17):
        ky, kx, kc = keypoints[i]
        if kc > threshold:
            cx, cy = int(kx * w), int(ky * h)
            cv2.circle(frame, (cx, cy), 5, (51, 87, 255), -1, cv2.LINE_AA)

# ── Pose detection helpers ────────────────────────────────────────────────────

def has_upper_body(keypoints, threshold=0.3):
    """Returns True if at least 3 upper body landmarks are visible."""
    if keypoints is None:
        return False
    upper = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST]
    count = sum(1 for i in upper if keypoints[i][2] > threshold)
    return count >= 3

def has_bicep_flex(keypoints, threshold=0.25):
    """
    Detects double bicep flex OR arms raised above head.
    Very lenient — designed for a camera at floor level pointing up.
    MoveNet keypoints are [y, x, confidence] in normalized coords.
    """
    if keypoints is None:
        return False

    ls_y, ls_x, ls_c = keypoints[KP_L_SHOULDER]
    rs_y, rs_x, rs_c = keypoints[KP_R_SHOULDER]
    le_y, le_x, le_c = keypoints[KP_L_ELBOW]
    re_y, re_x, re_c = keypoints[KP_R_ELBOW]
    lw_y, lw_x, lw_c = keypoints[KP_L_WRIST]
    rw_y, rw_x, rw_c = keypoints[KP_R_WRIST]
    nose_y, nose_x, nose_c = keypoints[KP_NOSE]

    if ls_c < threshold or rs_c < threshold:
        return False

    shoulder_width = abs(ls_x - rs_x)
    if shoulder_width < 0.05:
        return False

    shoulder_mid_y = (ls_y + rs_y) / 2
    score = 0

    # Left elbow wide of left shoulder
    if le_c > threshold:
        if le_x < ls_x - shoulder_width * 0.05:
            if abs(le_y - shoulder_mid_y) < shoulder_width * 1.5:
                score += 1

    # Right elbow wide of right shoulder
    if re_c > threshold:
        if re_x > rs_x + shoulder_width * 0.05:
            if abs(re_y - shoulder_mid_y) < shoulder_width * 1.5:
                score += 1

    # Arms raised above head fallback
    head_y = nose_y if nose_c > threshold else shoulder_mid_y - shoulder_width * 0.5
    if lw_c > threshold and lw_y < head_y:
        score += 1
    if rw_c > threshold and rw_y < head_y:
        score += 1

    return score >= 1

# ── State machine ─────────────────────────────────────────────────────────────

class MirrorSM:
    def __init__(self):
        self.state = "idle"
        self.t = time.time()
        self.person_present = False
        self.person_first_seen = None
        self.person_last_seen  = None
        self.pose_first_seen   = None
        self.hype_msg    = ""
        self.compliment  = ""
        self.cel_toggle  = False
        self.cel_t       = 0.0

    def _go(self, s):
        self.state = s
        self.t = time.time()
        if s == "celebrate":
            self.hype_msg   = random.choice(HYPE_MSGS)
            self.cel_toggle = False
            self.cel_t      = time.time()
        elif s == "compliment":
            self.compliment = random.choice(COMPLIMENTS)
        elif s == "idle":
            self.pose_first_seen  = None
            self.person_present   = False
            self.person_first_seen = None

    def update(self, person_detected, pose_detected):
        now = time.time()
        elapsed = now - self.t

        # ── Person presence hysteresis ──
        if person_detected:
            self.person_last_seen = now
            if not self.person_present:
                if self.person_first_seen is None:
                    self.person_first_seen = now
                elif now - self.person_first_seen >= 1.5:
                    self.person_present = True
                    if self.state == "idle":
                        self._go("greeting")
        else:
            self.person_first_seen = None
            if self.person_present and self.person_last_seen:
                if now - self.person_last_seen >= 3.0:
                    self._go("idle")

        # ── State transitions ──
        if self.state == "greeting" and elapsed >= 2.5:
            self._go("show_pose")

        elif self.state == "show_pose" and elapsed >= 3.5:
            self._go("prompt")

        elif self.state == "prompt":
            if pose_detected and self.person_present:
                if self.pose_first_seen is None:
                    self.pose_first_seen = now
                elif now - self.pose_first_seen >= 1.5:
                    self._go("celebrate")
            else:
                self.pose_first_seen = None

        elif self.state == "celebrate":
            # Toggle avatar
            if now - self.cel_t >= 0.5:
                self.cel_toggle = not self.cel_toggle
                self.cel_t = now
            if elapsed >= 3.5:
                self._go("compliment")

        elif self.state == "compliment" and elapsed >= 3.5:
            self._go("done")

        # "done" — wait for person to leave (handled above)

    @property
    def avatar_key(self):
        if self.state in ("idle", "greeting"):    return "wave"
        if self.state in ("show_pose", "prompt"): return "pose"
        if self.state == "celebrate":             return "celebrating" if not self.cel_toggle else "pose"
        if self.state in ("compliment", "done"):  return "thumbsup"
        return "wave"

    @property
    def bubble_text(self):
        if self.state == "idle":       return "Step up & strike a pose!"
        if self.state == "greeting":   return "Hey! Welcome to XL Fitness!"
        if self.state == "show_pose":  return "Check out this pose!"
        if self.state == "prompt":     return "Now YOU do it! Arms out & flex!"
        if self.state == "celebrate":  return self.hype_msg
        if self.state == "compliment": return self.compliment
        if self.state == "done":       return "Great work! See you next time!"
        return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  XL Fitness Smart Mirror")
    print("=" * 50)

    # Load assets
    assets = load_assets()
    logo_raw = assets.get("logo")

    # Load MoveNet model
    print("\nLoading MoveNet pose model...")
    interpreter = load_movenet_model()
    print("✓ MoveNet Lightning loaded")

    # Open camera at 720p for crisp display — MoveNet downscales internally to 192x192
    # so higher capture resolution costs almost nothing for inference.
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    # MJPG fourcc MUST be set before width/height for most USB cameras
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera (index 0).")
        print("Try: python3 smart_mirror.py --cam 1")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {cam_w}x{cam_h}")

    # Detect screen resolution
    screen_w, screen_h = 1920, 1080
    try:
        import subprocess, re
        out = subprocess.check_output(['xrandr'], text=True)
        for line in out.splitlines():
            if ' connected' in line:
                m = re.search(r'(\d+)x(\d+)\+', line)
                if m:
                    screen_w, screen_h = int(m.group(1)), int(m.group(2))
                    break
    except Exception:
        pass
    print(f"✓ Display: {screen_w}x{screen_h}")

    # Pre-scale assets to SCREEN resolution (overlays drawn crisp at full res)
    av_h = int(screen_h * 0.52)
    scaled_avatars = {}
    for key, img in assets.items():
        if key == "logo" or img is None:
            continue
        scaled_avatars[key] = resize_asset(img, av_h)

    logo_h = int(screen_h * 0.48)
    logo_img = None
    if logo_raw is not None:
        lh, lw = logo_raw.shape[:2]
        logo_w = int(logo_h * lw / lh)
        logo_resized = cv2.resize(logo_raw, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
        logo_img = invert_logo(logo_resized)

    # Pre-compute camera → screen scaling (cover crop)
    cam_scale = max(screen_w / cam_w, screen_h / cam_h)
    scaled_cam_w = int(cam_w * cam_scale)
    scaled_cam_h = int(cam_h * cam_scale)
    crop_ox = (scaled_cam_w - screen_w) // 2
    crop_oy = (scaled_cam_h - screen_h) // 2

    # Fullscreen window
    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    sm = MirrorSM()
    keypoints = None
    frame_n = 0
    fps_t = time.time()
    fps_count = 0
    fps_disp = 0

    print("\n✓ Running — press Q or ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_n += 1
        fps_count += 1

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # ── Pose inference every 2nd frame ──
        if frame_n % 2 == 0:
            keypoints = run_movenet(interpreter, frame)

        person = has_upper_body(keypoints)
        flex   = has_bicep_flex(keypoints)
        sm.update(person, flex)

        # ── Scale camera to screen (cover crop) ──
        display = cv2.resize(frame, (scaled_cam_w, scaled_cam_h),
                             interpolation=cv2.INTER_LINEAR)
        display = display[crop_oy:crop_oy+screen_h,
                          crop_ox:crop_ox+screen_w]

        # ── Skeleton overlay (at screen res for crispness) ──
        draw_skeleton(display, keypoints,
                      w=screen_w, h=screen_h,
                      threshold=0.3)

        # ── Logo top-centre ──
        if logo_img is not None:
            lx = (screen_w - logo_img.shape[1]) // 2
            overlay_png(display, logo_img, lx, 12)

        # ── Avatar bottom-right ──
        avatar = scaled_avatars.get(sm.avatar_key)
        if avatar is not None:
            ax = screen_w - avatar.shape[1] - 10
            ay = screen_h - avatar.shape[0] + 15
            overlay_png(display, avatar, ax, ay)

        # ── Speech bubble / hype text ──
        bubble = sm.bubble_text
        if bubble:
            if sm.state == "celebrate":
                put_text_centred(display, bubble,
                                 screen_h // 2,
                                 scale=2.8, color=(51,87,255), thickness=4)
            elif sm.state == "compliment":
                put_text_centred(display, bubble,
                                 screen_h - int(screen_h * 0.42),
                                 scale=1.6, color=(80,220,80), thickness=3)
            else:
                bx = screen_w - int(screen_w * 0.27)
                by = screen_h - int(screen_h * 0.54)
                draw_bubble(display, bubble, bx, by)

        # ── FPS ──
        now = time.time()
        if now - fps_t >= 1.0:
            fps_disp  = fps_count
            fps_count = 0
            fps_t     = now
        cv2.putText(display, f"{fps_disp}fps",
                    (10, screen_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()
