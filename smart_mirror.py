#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 4
=========================================
Simple version: live camera + skeleton overlay.
When you do a double bicep flex — BIG CELEBRATION.

Requirements:
    pip3 install tflite-runtime opencv-python "numpy<2" --break-system-packages

Usage:
    python3 smart_mirror.py
    DISPLAY=:0 python3 smart_mirror.py   # when running over SSH

Press Q or ESC to quit.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import sys
import random

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_RAW = "https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/assets"
ASSETS_DIR = os.path.expanduser("~/smart_mirror_assets")

LOGO_FILE = "SMARTMIRROR.png"
CELEBRATE_AVATAR = "XLAvatar-Celebrating.png"
MOVENET_URL = "https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/assets/movenet_lightning.tflite"
MOVENET_FILE = "movenet_lightning.tflite"

HYPE_MSGS = [
    "BEAST MODE!",
    "ABSOLUTE UNIT!",
    "LETS GOOO!",
    "CHAMPION!",
    "UNSTOPPABLE!",
    "CRUSHING IT!",
    "PURE POWER!",
    "LEGENDARY!",
]

CELEBRATION_SECONDS = 3.5
COOLDOWN_SECONDS    = 2.0   # After celebration, wait before detecting again

# ── MoveNet keypoint indices (COCO 17) ────────────────────────────────────────
KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER = 0, 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP     = 11, 12
KP_L_KNEE, KP_R_KNEE   = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

SKELETON = [
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP),   (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE), (KP_R_KNEE, KP_R_ANKLE),
    (KP_NOSE, KP_L_SHOULDER), (KP_NOSE, KP_R_SHOULDER),
]

# ── Downloads ─────────────────────────────────────────────────────────────────

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def download_if_missing(url, dest, label=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return True
    print(f"  Downloading {label or os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {os.path.basename(dest)} ({os.path.getsize(dest)//1024}KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def load_optional_asset(filename):
    """Download an asset from the repo and return the image (or None)."""
    dest = os.path.join(ASSETS_DIR, filename)
    url  = f"{GITHUB_RAW}/{filename}"
    if download_if_missing(url, dest, filename):
        return cv2.imread(dest, cv2.IMREAD_UNCHANGED)
    return None

def load_movenet():
    ensure_dir(ASSETS_DIR)
    model_path = os.path.join(ASSETS_DIR, MOVENET_FILE)
    if not download_if_missing(MOVENET_URL, model_path, "MoveNet model"):
        sys.exit(1)
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        print("ERROR: tflite-runtime not installed.")
        print("Run: pip3 install tflite-runtime --break-system-packages")
        sys.exit(1)
    interp = Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    return interp

def run_movenet(interp, frame):
    """Run MoveNet on a frame. Returns (17, 3) array of [y, x, conf]."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    h, w = inp['shape'][1], inp['shape'][2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    dtype = inp['dtype']
    img = np.expand_dims(img, 0).astype(dtype)

    interp.set_tensor(inp['index'], img)
    interp.invoke()
    return interp.get_tensor(out['index'])[0][0]

# ── Image overlay ─────────────────────────────────────────────────────────────

def overlay_png(bg, overlay, x, y):
    if overlay is None:
        return
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    ox1, oy1 = x1 - x, y1 - y
    patch = overlay[oy1:oy1+(y2-y1), ox1:ox1+(x2-x1)]
    roi   = bg[y1:y2, x1:x2]
    if overlay.shape[2] == 4:
        a = patch[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (patch[:, :, :3].astype(np.float32) * a +
                  roi.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)
    else:
        roi[:] = patch[:, :, :3]

def resize_to_h(img, target_h):
    if img is None: return None
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)

def invert_rgb(img):
    if img is None: return None
    out = img.copy()
    out[:, :, :3] = 255 - out[:, :, :3]
    return out

# ── Text ──────────────────────────────────────────────────────────────────────

def text_with_shadow(frame, text, x, y, scale, color, thickness):
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x+3, y+3), f, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),     f, scale, color,   thickness,   cv2.LINE_AA)

def text_centred(frame, text, cy, scale, color, thickness):
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(text, f, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    text_with_shadow(frame, text, x, cy, scale, color, thickness)

# ── Skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame, kp, w, h, threshold=0.3, color=(51, 87, 255)):
    if kp is None:
        return
    for i, j in SKELETON:
        ya, xa, ca = kp[i]
        yb, xb, cb = kp[j]
        if ca < threshold or cb < threshold:
            continue
        cv2.line(frame, (int(xa*w), int(ya*h)), (int(xb*w), int(yb*h)),
                 color, 4, cv2.LINE_AA)
    for i in range(17):
        y, x, c = kp[i]
        if c > threshold:
            cv2.circle(frame, (int(x*w), int(y*h)), 7, color, -1, cv2.LINE_AA)

# ── Flex detection ────────────────────────────────────────────────────────────

def is_double_bicep_flex(kp, threshold=0.25):
    """
    Detects a double-arm bicep flex: both elbows roughly at shoulder height,
    out wide, with wrists bent up/in (above elbow).
    """
    if kp is None:
        return False

    ls_y, ls_x, ls_c = kp[KP_L_SHOULDER]
    rs_y, rs_x, rs_c = kp[KP_R_SHOULDER]
    le_y, le_x, le_c = kp[KP_L_ELBOW]
    re_y, re_x, re_c = kp[KP_R_ELBOW]
    lw_y, lw_x, lw_c = kp[KP_L_WRIST]
    rw_y, rw_x, rw_c = kp[KP_R_WRIST]

    # Need both shoulders + both elbows + both wrists visible
    if min(ls_c, rs_c, le_c, re_c, lw_c, rw_c) < threshold:
        return False

    shoulder_w = abs(ls_x - rs_x)
    if shoulder_w < 0.05:
        return False

    shoulder_mid_y = (ls_y + rs_y) / 2

    # Left elbow should be OUT wide (further left than left shoulder, in mirror image)
    # Right elbow should be OUT wide (further right than right shoulder)
    left_wide  = le_x < ls_x + shoulder_w * 0.1
    right_wide = re_x > rs_x - shoulder_w * 0.1
    if not (left_wide and right_wide):
        return False

    # Elbows roughly at shoulder height (not hanging down, not overhead)
    elbow_ok = (abs(le_y - shoulder_mid_y) < shoulder_w * 1.2 and
                abs(re_y - shoulder_mid_y) < shoulder_w * 1.2)
    if not elbow_ok:
        return False

    # Wrists should be ABOVE (smaller y) the elbows — that's the flex
    left_flexed  = lw_y < le_y - shoulder_w * 0.1
    right_flexed = rw_y < re_y - shoulder_w * 0.1

    return left_flexed and right_flexed

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  XL Fitness Smart Mirror")
    print("=" * 50)

    ensure_dir(ASSETS_DIR)

    print("Loading MoveNet...")
    interp = load_movenet()
    print("✓ MoveNet loaded")

    print("Loading assets...")
    logo_raw = load_optional_asset(LOGO_FILE)
    celebrate_raw = load_optional_asset(CELEBRATE_AVATAR)

    # Camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {cam_w}x{cam_h}")

    # Screen resolution
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

    # Pre-scale logo + avatar at screen resolution
    logo_img = None
    if logo_raw is not None:
        logo_img = invert_rgb(resize_to_h(logo_raw, int(screen_h * 0.18)))

    celebrate_img = resize_to_h(celebrate_raw, int(screen_h * 0.7))

    # Camera → screen scaling (cover crop)
    s = max(screen_w / cam_w, screen_h / cam_h)
    sw, sh = int(cam_w * s), int(cam_h * s)
    ox, oy = (sw - screen_w) // 2, (sh - screen_h) // 2

    # Window
    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # State
    keypoints = None
    celebrating_until = 0.0
    cooldown_until    = 0.0
    current_hype      = ""
    pulse_t           = 0.0

    frame_n = 0
    fps_t, fps_count, fps_disp = time.time(), 0, 0
    print("\n✓ Running — strike a double bicep flex! Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame_n += 1
        fps_count += 1

        # Mirror
        frame = cv2.flip(frame, 1)

        # Inference every 2nd frame
        if frame_n % 2 == 0:
            keypoints = run_movenet(interp, frame)

        now = time.time()
        celebrating = now < celebrating_until

        # Flex detection (only when not already celebrating and cooldown expired)
        if not celebrating and now >= cooldown_until:
            if is_double_bicep_flex(keypoints):
                celebrating_until = now + CELEBRATION_SECONDS
                cooldown_until    = celebrating_until + COOLDOWN_SECONDS
                current_hype      = random.choice(HYPE_MSGS)
                pulse_t           = now
                celebrating = True

        # Scale camera to screen
        display = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        display = display[oy:oy+screen_h, ox:ox+screen_w]

        # Skeleton (green when celebrating, orange otherwise)
        skel_color = (0, 255, 80) if celebrating else (51, 87, 255)
        draw_skeleton(display, keypoints, screen_w, screen_h, color=skel_color)

        # Logo (top centre, always)
        if logo_img is not None:
            lx = (screen_w - logo_img.shape[1]) // 2
            overlay_png(display, logo_img, lx, 20)

        # Celebration overlay
        if celebrating:
            # Semi-transparent dark overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (screen_w, screen_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

            # Pulsing hype text
            pulse = 1.0 + 0.15 * abs(np.sin((now - pulse_t) * 6))
            hype_scale = 4.5 * pulse
            text_centred(display, current_hype,
                         int(screen_h * 0.45),
                         scale=hype_scale, color=(0, 255, 80), thickness=8)

            # Celebrating avatar bottom-right
            if celebrate_img is not None:
                ax = screen_w - celebrate_img.shape[1] - 20
                ay = screen_h - celebrate_img.shape[0] - 20
                overlay_png(display, celebrate_img, ax, ay)

        # FPS
        if now - fps_t >= 1.0:
            fps_disp, fps_count, fps_t = fps_count, 0, now
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
