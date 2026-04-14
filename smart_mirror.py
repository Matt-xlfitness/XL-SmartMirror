#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 4
=========================================
Runs fully offline after first model download.
Uses OpenCV + PoseNet (MobileNet V1) via cv2.dnn — no TFLite, no MediaPipe needed.

Requirements (already installed):
    pip3 install opencv-python --break-system-packages

Assets are downloaded from GitHub on first run and cached locally.

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

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_RAW = "https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/assets"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets_cache")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")

ASSET_FILES = {
    "wave":        "XLAvatar-Wave.png",
    "point":       "XLAvatar-Point.png",
    "pose":        "XLAvatar-01Pose.png",
    "thumbsup":    "XLAvatar-ThumbsUp.png",
    "celebrating": "XLAvatar-Celebrating.png",
    "logo":        "SMARTMIRROR.png",
}

# PoseNet MobileNet V1 — works via cv2.dnn on ARM, no TFLite needed
MODEL_URL   = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
MODEL_FILE  = os.path.join(MODEL_DIR, "posenet.tflite")

# Camera settings — keep low for Pi 4
CAM_W, CAM_H = 640, 480
CAM_FPS      = 15

# Inference — run every N frames (Pi 4: every 5 frames = ~3fps at 15fps camera)
INFER_EVERY  = 5

# Keypoint indices (COCO format used by PoseNet/MoveNet)
KP_NOSE       = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW    = 7
KP_R_ELBOW    = 8
KP_L_WRIST    = 9
KP_R_WRIST    = 10

SKELETON_PAIRS = [
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # legs
    (0,1),(0,2),(1,3),(2,4),           # face
]

UPPER_KPS = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST]

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

# ── Download helpers ──────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_if_missing(url, dest, label=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    print(f"Downloading {label or os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {os.path.basename(dest)} ({os.path.getsize(dest)//1024}KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def load_assets():
    ensure_dir(ASSETS_DIR)
    assets = {}
    for key, filename in ASSET_FILES.items():
        dest = os.path.join(ASSETS_DIR, filename)
        url  = f"{GITHUB_RAW}/{filename}"
        if download_if_missing(url, dest, key):
            img = cv2.imread(dest, cv2.IMREAD_UNCHANGED)
            if img is not None:
                assets[key] = img
            else:
                print(f"  ✗ Could not load image: {dest}")
        else:
            assets[key] = None
    return assets

# ── Image overlay helpers ─────────────────────────────────────────────────────

def overlay_png(background, overlay, x, y, max_h=None, max_w=None):
    """Overlay a PNG with alpha channel onto background at (x, y)."""
    if overlay is None:
        return background

    h, w = overlay.shape[:2]
    if max_h and h > max_h:
        scale = max_h / h
        w = int(w * scale)
        h = max_h
        overlay = cv2.resize(overlay, (w, h))
    if max_w and w > max_w:
        scale = max_w / w
        h = int(h * scale)
        w = max_w
        overlay = cv2.resize(overlay, (w, h))

    # Clip to frame bounds
    bh, bw = background.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + w), min(bh, y + h)
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    roi = background[y1:y2, x1:x2]
    patch = overlay[oy1:oy2, ox1:ox2]

    if overlay.shape[2] == 4:
        alpha = patch[:, :, 3:4] / 255.0
        rgb   = patch[:, :, :3]
        roi[:] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        roi[:] = patch[:, :, :3]

    return background

def draw_text_with_shadow(frame, text, x, y, font_scale=1.5, color=(255,255,255),
                           thickness=3, shadow_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x+2, y+2), font, font_scale, shadow_color, thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_text_centred(frame, text, cy, font_scale=1.5, color=(255,255,255), thickness=3):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    draw_text_with_shadow(frame, text, x, cy, font_scale, color, thickness)

def draw_bubble(frame, text, cx, cy, font_scale=0.8, bg=(0,0,0,160), fg=(255,255,255)):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 2)
    pad = 14
    x1 = cx - tw//2 - pad
    y1 = cy - th - pad
    x2 = cx + tw//2 + pad
    y2 = cy + baseline + pad
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg[:3], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80,80,80), 1)
    draw_text_with_shadow(frame, text, cx - tw//2, cy, font_scale, fg, 2)

# ── Pose detection ────────────────────────────────────────────────────────────

def run_posenet(net, frame):
    """Run PoseNet inference. Returns list of keypoints [(x,y,score), ...]."""
    input_size = 257
    blob = cv2.dnn.blobFromImage(
        frame, 1.0/255.0, (input_size, input_size),
        mean=(0,0,0), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    h, w = frame.shape[:2]
    keypoints = []

    # PoseNet output: heatmaps [1, H, W, 17] and offsets [1, H, W, 34]
    if len(outputs) >= 2:
        heatmaps = outputs[0][0]  # (H, W, 17)
        offsets  = outputs[1][0]  # (H, W, 34)
        hm_h, hm_w, num_kp = heatmaps.shape

        for kp_idx in range(num_kp):
            hm = heatmaps[:, :, kp_idx]
            # Find peak
            flat_idx = np.argmax(hm)
            hy, hx = divmod(flat_idx, hm_w)
            score = float(1 / (1 + np.exp(-hm[hy, hx])))  # sigmoid

            # Apply offsets
            offset_y = offsets[hy, hx, kp_idx]
            offset_x = offsets[hy, hx, kp_idx + num_kp]

            y = int((hy / hm_h) * h + offset_y)
            x = int((hx / hm_w) * w + offset_x)
            keypoints.append((x, y, score))
    else:
        keypoints = [(0, 0, 0.0)] * 17

    return keypoints

def has_upper_body(keypoints, threshold=0.25):
    count = sum(1 for idx in UPPER_KPS if keypoints[idx][2] > threshold)
    return count >= 3

def has_bicep_flex(keypoints, threshold=0.2):
    ls = keypoints[KP_L_SHOULDER]
    rs = keypoints[KP_R_SHOULDER]
    le = keypoints[KP_L_ELBOW]
    re = keypoints[KP_R_ELBOW]
    lw = keypoints[KP_L_WRIST]
    rw = keypoints[KP_R_WRIST]
    nose = keypoints[KP_NOSE]

    if ls[2] < threshold or rs[2] < threshold:
        return False

    shoulder_width = abs(ls[0] - rs[0])
    if shoulder_width < 20:
        return False

    shoulder_mid_y = (ls[1] + rs[1]) / 2
    flex_score = 0

    # Left elbow wide and at shoulder height
    if le[2] > threshold:
        if le[0] < ls[0] - shoulder_width * 0.1:
            if abs(le[1] - shoulder_mid_y) < shoulder_width * 1.1:
                flex_score += 1

    # Right elbow wide and at shoulder height
    if re[2] > threshold:
        if re[0] > rs[0] + shoulder_width * 0.1:
            if abs(re[1] - shoulder_mid_y) < shoulder_width * 1.1:
                flex_score += 1

    # Arms-up fallback
    head_y = nose[1] if nose[2] > threshold else shoulder_mid_y - shoulder_width * 0.5
    if lw[2] > threshold and lw[1] < head_y:
        flex_score += 1
    if rw[2] > threshold and rw[1] < head_y:
        flex_score += 1

    return flex_score >= 1

def draw_skeleton(frame, keypoints, threshold=0.25):
    h, w = frame.shape[:2]
    for (i, j) in SKELETON_PAIRS:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1, s1 = keypoints[i]
        x2, y2, s2 = keypoints[j]
        if s1 < threshold or s2 < threshold:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), (255, 87, 51), 3, cv2.LINE_AA)

    for (x, y, score) in keypoints:
        if score > threshold:
            cv2.circle(frame, (x, y), 5, (255, 87, 51), -1, cv2.LINE_AA)

# ── State machine ─────────────────────────────────────────────────────────────

class MirrorStateMachine:
    STATES = ["idle", "greeting", "show_pose", "prompt", "celebrate", "compliment", "done"]

    def __init__(self):
        self.state = "idle"
        self.state_entered = time.time()
        self.person_present = False
        self.person_first_seen = None
        self.person_last_seen = None
        self.pose_first_seen = None
        self.hype_msg = ""
        self.compliment_msg = ""
        self.celebrate_toggle = False
        self.celebrate_last_toggle = 0

    def update(self, person_detected, pose_detected):
        now = time.time()

        # Track person presence with hysteresis
        if person_detected:
            self.person_last_seen = now
            if not self.person_present:
                if self.person_first_seen is None:
                    self.person_first_seen = now
                elif now - self.person_first_seen >= 1.5:
                    self.person_present = True
                    if self.state == "idle":
                        self._transition("greeting")
        else:
            self.person_first_seen = None
            if self.person_present and self.person_last_seen:
                if now - self.person_last_seen >= 3.0:
                    self.person_present = False
                    self.pose_first_seen = None
                    if self.state != "idle":
                        self._transition("idle")

        # State-specific logic
        elapsed = now - self.state_entered

        if self.state == "greeting":
            if elapsed >= 2.5:
                self._transition("show_pose")

        elif self.state == "show_pose":
            if elapsed >= 3.5:
                self._transition("prompt")

        elif self.state == "prompt":
            if pose_detected and self.person_present:
                if self.pose_first_seen is None:
                    self.pose_first_seen = now
                elif now - self.pose_first_seen >= 1.5:
                    self._transition("celebrate")
            else:
                self.pose_first_seen = None

        elif self.state == "celebrate":
            # Toggle avatar for animation
            if now - self.celebrate_last_toggle >= 0.5:
                self.celebrate_toggle = not self.celebrate_toggle
                self.celebrate_last_toggle = now
            if elapsed >= 3.5:
                self._transition("compliment")

        elif self.state == "compliment":
            if elapsed >= 3.0:
                self._transition("done")

        elif self.state == "done":
            pass  # Wait for person to leave (handled above)

    def _transition(self, new_state):
        import random
        self.state = new_state
        self.state_entered = time.time()
        if new_state == "celebrate":
            self.hype_msg = random.choice(HYPE_MSGS)
            self.celebrate_toggle = False
            self.celebrate_last_toggle = time.time()
        elif new_state == "compliment":
            self.compliment_msg = random.choice(COMPLIMENTS)
        elif new_state == "idle":
            self.pose_first_seen = None

    def get_avatar_key(self):
        if self.state == "idle":       return "wave"
        if self.state == "greeting":   return "wave"
        if self.state == "show_pose":  return "pose"   # show example
        if self.state == "prompt":     return "pose"   # show example
        if self.state == "celebrate":  return "celebrating" if not self.celebrate_toggle else "pose"
        if self.state == "compliment": return "thumbsup"
        if self.state == "done":       return "thumbsup"
        return "wave"

    def get_bubble_text(self):
        if self.state == "idle":       return "Step up & strike a pose!"
        if self.state == "greeting":   return "Hey! Welcome to XL Fitness!"
        if self.state == "show_pose":  return "Check out this pose - Double Bicep Flex!"
        if self.state == "prompt":     return "Now YOU do it! Arms out & flex!"
        if self.state == "celebrate":  return self.hype_msg
        if self.state == "compliment": return self.compliment_msg
        if self.state == "done":       return "Great work! See you next time!"
        return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("XL Fitness Smart Mirror — Starting up...")
    print(f"OpenCV version: {cv2.__version__}")

    # Download assets
    print("\nChecking assets...")
    assets = load_assets()

    # Download model
    ensure_dir(MODEL_DIR)
    if not download_if_missing(MODEL_URL, MODEL_FILE, "PoseNet model"):
        print("ERROR: Could not download pose model. Check internet connection.")
        sys.exit(1)

    # Load model
    print("\nLoading pose model...")
    try:
        net = cv2.dnn.readNetFromTFLite(MODEL_FILE)
        print("  ✓ Model loaded via cv2.dnn")
    except Exception as e:
        print(f"  ✗ cv2.dnn failed: {e}")
        print("  Trying fallback (no pose detection, display only)...")
        net = None

    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # Reduce buffer lag

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✓ Camera opened at {actual_w}x{actual_h}")

    # Create fullscreen window
    win_name = "XL Fitness Smart Mirror"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get display size
    screen_w = 1920
    screen_h = 1080
    try:
        import subprocess
        result = subprocess.run(['xrandr'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if ' connected primary' in line or (' connected' in line and '*' in line):
                import re
                m = re.search(r'(\d+)x(\d+)', line)
                if m:
                    screen_w, screen_h = int(m.group(1)), int(m.group(2))
                    break
    except:
        pass
    print(f"  Display: {screen_w}x{screen_h}")

    sm = MirrorStateMachine()
    keypoints = [(0, 0, 0.0)] * 17
    frame_count = 0
    fps_counter = 0
    fps_display = 0
    fps_timer = time.time()

    print("\nRunning! Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        fps_counter += 1

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # ── Pose inference (throttled) ──
        person_detected = False
        pose_detected   = False

        if net is not None and frame_count % INFER_EVERY == 0:
            try:
                keypoints = run_posenet(net, frame)
                person_detected = has_upper_body(keypoints)
                pose_detected   = has_bicep_flex(keypoints)
            except Exception as e:
                pass  # Skip this frame on error

        elif net is None:
            # No model — just use motion detection as fallback
            person_detected = True

        # Update state machine
        sm.update(person_detected, pose_detected)

        # ── Scale frame to screen ──
        scale = max(screen_w / actual_w, screen_h / actual_h)
        disp_w = int(actual_w * scale)
        disp_h = int(actual_h * scale)
        display = cv2.resize(frame, (disp_w, disp_h))

        # Crop to screen
        ox = (disp_w - screen_w) // 2
        oy = (disp_h - screen_h) // 2
        display = display[oy:oy+screen_h, ox:ox+screen_w]

        # ── Draw skeleton ──
        if person_detected and keypoints:
            scaled_kps = []
            for (x, y, s) in keypoints:
                sx = int(x * scale) - ox
                sy = int(y * scale) - oy
                scaled_kps.append((sx, sy, s))
            draw_skeleton(display, scaled_kps)

        # ── Logo — top centre ──
        logo = assets.get("logo")
        if logo is not None:
            logo_w = int(screen_w * 0.45)
            lh, lw = logo.shape[:2]
            logo_h = int(logo_w * lh / lw)
            logo_resized = cv2.resize(logo, (logo_w, logo_h))
            lx = (screen_w - logo_w) // 2
            ly = 10
            # Invert logo (it's black on white, we want white on dark)
            if logo_resized.shape[2] == 4:
                bgr = logo_resized[:,:,:3]
                alpha = logo_resized[:,:,3]
                bgr_inv = 255 - bgr
                logo_resized = np.dstack([bgr_inv, alpha])
            overlay_png(display, logo_resized, lx, ly)

        # ── Avatar — bottom right ──
        avatar_key = sm.get_avatar_key()
        avatar = assets.get(avatar_key)
        if avatar is not None:
            av_h = int(screen_h * 0.55)
            avh, avw = avatar.shape[:2]
            av_w = int(av_h * avw / avh)
            ax = screen_w - av_w - 10
            ay = screen_h - av_h + 20
            overlay_png(display, avatar, ax, ay, max_h=av_h)

        # ── Speech bubble ──
        bubble = sm.get_bubble_text()
        if bubble:
            state = sm.state
            if state == "celebrate":
                # Big hype text
                draw_text_centred(display, bubble,
                                  screen_h // 2,
                                  font_scale=2.8,
                                  color=(51, 87, 255),
                                  thickness=4)
            elif state == "compliment":
                draw_text_centred(display, bubble,
                                  screen_h - int(screen_h * 0.45),
                                  font_scale=1.6,
                                  color=(80, 220, 80),
                                  thickness=3)
            else:
                bx = screen_w - int(screen_w * 0.28)
                by = screen_h - int(screen_h * 0.52)
                draw_bubble(display, bubble, bx, by, font_scale=0.85)

        # ── FPS counter ──
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = now
        cv2.putText(display, f"{fps_display} fps",
                    (10, screen_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255,80), 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()
