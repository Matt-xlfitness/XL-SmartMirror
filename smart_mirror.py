#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 4
=========================================
Uses MediaPipe Pose (mediapipe-rpi4) for smooth 10-15fps pose detection.

Requirements:
    pip3 install opencv-python mediapipe-rpi4 --break-system-packages

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

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_NOSE          = 0
MP_L_SHOULDER    = 11
MP_R_SHOULDER    = 12
MP_L_ELBOW       = 13
MP_R_ELBOW       = 14
MP_L_WRIST       = 15
MP_R_WRIST       = 16
MP_L_HIP         = 23
MP_R_HIP         = 24

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(24,26),(26,28),           # legs
    (0,11),(0,12),                             # head to shoulders
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

def draw_skeleton(frame, landmarks, w, h, threshold=0.5):
    if landmarks is None:
        return
    lm = landmarks.landmark

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        if i >= len(lm) or j >= len(lm):
            continue
        a, b = lm[i], lm[j]
        if a.visibility < threshold or b.visibility < threshold:
            continue
        x1, y1 = int(a.x * w), int(a.y * h)
        x2, y2 = int(b.x * w), int(b.y * h)
        cv2.line(frame, (x1,y1), (x2,y2), (51, 87, 255), 3, cv2.LINE_AA)

    # Draw joints
    for lmk in lm:
        if lmk.visibility > threshold:
            cx, cy = int(lmk.x * w), int(lmk.y * h)
            cv2.circle(frame, (cx, cy), 5, (51, 87, 255), -1, cv2.LINE_AA)

# ── Pose detection helpers ────────────────────────────────────────────────────

def has_upper_body(landmarks, threshold=0.4):
    """Returns True if at least 3 upper body landmarks are visible."""
    if landmarks is None:
        return False
    lm = landmarks.landmark
    upper = [MP_L_SHOULDER, MP_R_SHOULDER, MP_L_ELBOW, MP_R_ELBOW, MP_L_WRIST, MP_R_WRIST]
    count = sum(1 for i in upper if lm[i].visibility > threshold)
    return count >= 3

def has_bicep_flex(landmarks, threshold=0.35):
    """
    Detects double bicep flex OR arms raised above head.
    Very lenient — designed for a camera at floor level pointing up.
    """
    if landmarks is None:
        return False
    lm = landmarks.landmark

    ls = lm[MP_L_SHOULDER]
    rs = lm[MP_R_SHOULDER]
    le = lm[MP_L_ELBOW]
    re = lm[MP_R_ELBOW]
    lw = lm[MP_L_WRIST]
    rw = lm[MP_R_WRIST]
    nose = lm[MP_NOSE]

    if ls.visibility < threshold or rs.visibility < threshold:
        return False

    shoulder_width = abs(ls.x - rs.x)
    if shoulder_width < 0.05:
        return False

    shoulder_mid_y = (ls.y + rs.y) / 2
    score = 0

    # Left elbow wide of left shoulder
    if le.visibility > threshold:
        if le.x < ls.x - shoulder_width * 0.05:
            if abs(le.y - shoulder_mid_y) < shoulder_width * 1.5:
                score += 1

    # Right elbow wide of right shoulder
    if re.visibility > threshold:
        if re.x > rs.x + shoulder_width * 0.05:
            if abs(re.y - shoulder_mid_y) < shoulder_width * 1.5:
                score += 1

    # Arms raised above head fallback
    head_y = nose.y if nose.visibility > threshold else shoulder_mid_y - shoulder_width * 0.5
    if lw.visibility > threshold and lw.y < head_y:
        score += 1
    if rw.visibility > threshold and rw.y < head_y:
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

    # Import MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        print(f"✓ MediaPipe {mp.__version__} loaded")
    except ImportError:
        print("ERROR: mediapipe not installed.")
        print("Run: pip3 install mediapipe-rpi4 --break-system-packages")
        sys.exit(1)

    # Load assets
    assets = load_assets()
    logo_raw = assets.get("logo")

    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          15)
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

    # Pre-scale assets once
    av_h = int(screen_h * 0.52)
    scaled_avatars = {}
    for key, img in assets.items():
        if key == "logo" or img is None:
            continue
        scaled_avatars[key] = resize_asset(img, av_h)

    logo_h = int(screen_h * 0.12)
    logo_img = None
    if logo_raw is not None:
        lh, lw = logo_raw.shape[:2]
        logo_w = int(logo_h * lw / lh)
        logo_resized = cv2.resize(logo_raw, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
        logo_img = invert_logo(logo_resized)

    # Init MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,          # 0 = fastest (Lite model)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Fullscreen window
    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    sm = MirrorSM()
    landmarks = None
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            landmarks = result.pose_landmarks

        person = has_upper_body(landmarks)
        flex   = has_bicep_flex(landmarks)
        sm.update(person, flex)

        # ── Scale camera to screen (cover) ──
        scale = max(screen_w / cam_w, screen_h / cam_h)
        dw = int(cam_w * scale)
        dh = int(cam_h * scale)
        display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        ox = (dw - screen_w) // 2
        oy = (dh - screen_h) // 2
        display = display[oy:oy+screen_h, ox:ox+screen_w]

        # ── Skeleton overlay ──
        if landmarks:
            draw_skeleton(display, landmarks,
                          w=screen_w, h=screen_h,
                          threshold=0.4)

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

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()
