#!/usr/bin/env python3
"""
XL Fitness Smart Mirror — Raspberry Pi 4
=========================================
Live camera + skeleton overlay.
Avatar asks you to strike a pose — when you hit a double bicep flex,
BIG CELEBRATION.

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

ASSET_FILES = {
    "wave":        "XLAvatar-Wave.png",
    "point":       "XLAvatar-Point.png",
    "pose":        "XLAvatar-01Pose.png",
    "thumbsup":    "XLAvatar-ThumbsUp.png",
    "celebrating": "XLAvatar-Celebrating.png",
    "logo":        "SMARTMIRROR.png",
}

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

CELEBRATION_SECONDS     = 3.5
COOLDOWN_SECONDS        = 2.0
STRIKE_PROMPT_SECONDS   = 3.0  # "Strike a pose!" shown for this long
                               # then swaps to "Like this!" + example pose avatar

# ── MoveNet keypoint indices (COCO 17) ────────────────────────────────────────
KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER = 0, 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP     = 11, 12
KP_L_KNEE, KP_R_KNEE   = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# Skeleton bones: (from, to, confidence_threshold).
# Legs use a higher threshold — if they're cut off / off-camera we don't
# want to draw jittery partial legs just because MoveNet guessed something.
SKELETON = [
    (KP_L_SHOULDER, KP_R_SHOULDER, 0.30),
    (KP_L_SHOULDER, KP_L_ELBOW,    0.30),
    (KP_L_ELBOW,    KP_L_WRIST,    0.30),
    (KP_R_SHOULDER, KP_R_ELBOW,    0.30),
    (KP_R_ELBOW,    KP_R_WRIST,    0.30),
    (KP_L_SHOULDER, KP_L_HIP,      0.35),
    (KP_R_SHOULDER, KP_R_HIP,      0.35),
    (KP_L_HIP,      KP_R_HIP,      0.40),
    (KP_L_HIP,      KP_L_KNEE,     0.50),
    (KP_L_KNEE,     KP_L_ANKLE,    0.50),
    (KP_R_HIP,      KP_R_KNEE,     0.50),
    (KP_R_KNEE,     KP_R_ANKLE,    0.50),
    (KP_NOSE,       KP_L_SHOULDER, 0.25),
    (KP_NOSE,       KP_R_SHOULDER, 0.25),
]

# Per-joint confidence thresholds — legs require higher confidence to draw
JOINT_THRESHOLDS = {
    KP_NOSE: 0.25,
    KP_L_SHOULDER: 0.30, KP_R_SHOULDER: 0.30,
    KP_L_ELBOW: 0.30,    KP_R_ELBOW: 0.30,
    KP_L_WRIST: 0.30,    KP_R_WRIST: 0.30,
    KP_L_HIP: 0.40,      KP_R_HIP: 0.40,
    KP_L_KNEE: 0.50,     KP_R_KNEE: 0.50,
    KP_L_ANKLE: 0.50,    KP_R_ANKLE: 0.50,
}

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

def load_assets():
    ensure_dir(ASSETS_DIR)
    assets = {}
    print("Loading assets...")
    for key, fname in ASSET_FILES.items():
        dest = os.path.join(ASSETS_DIR, fname)
        if download_if_missing(f"{GITHUB_RAW}/{fname}", dest, key):
            assets[key] = cv2.imread(dest, cv2.IMREAD_UNCHANGED)
        else:
            assets[key] = None
    return assets

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
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    h, w = inp['shape'][1], inp['shape'][2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, 0).astype(inp['dtype'])
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

def draw_bubble(frame, text, cx, cy, scale=1.2, thickness=2):
    """Semi-transparent rounded rectangle with text inside, pointing down-right."""
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, f, scale, thickness)
    pad = 24
    x1, y1 = cx - tw//2 - pad, cy - th - pad
    x2, y2 = cx + tw//2 + pad, cy + bl + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1]-1, x2)
    y2 = min(frame.shape[0]-1, y2)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text_with_shadow(frame, text, cx - tw//2, cy, scale, (255, 255, 255), thickness)

# ── Skeleton ──────────────────────────────────────────────────────────────────

def hsv_to_bgr(h, s, v):
    """Convert HSV (0-360, 0-1, 0-1) to BGR tuple."""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:    r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return (int((b+m)*255), int((g+m)*255), int((r+m)*255))

def draw_skeleton(frame, kp, w, h, color=(51, 87, 255), rainbow=False, t=0.0):
    """Draw skeleton. Legs use a higher confidence threshold so partial
    off-camera bodies don't get forced, jittery leg lines."""
    if kp is None:
        return
    n_bones = len(SKELETON)
    for idx, (i, j, bone_thr) in enumerate(SKELETON):
        ya, xa, ca = kp[i]
        yb, xb, cb = kp[j]
        if ca < bone_thr or cb < bone_thr:
            continue
        if rainbow:
            hue = ((idx / n_bones) * 360 + t * 400) % 360
            c = hsv_to_bgr(hue, 1.0, 1.0)
        else:
            c = color
        cv2.line(frame, (int(xa*w), int(ya*h)), (int(xb*w), int(yb*h)),
                 c, 5, cv2.LINE_AA)
    for i in range(17):
        y, x, conf = kp[i]
        if conf > JOINT_THRESHOLDS.get(i, 0.3):
            if rainbow:
                hue = ((i / 17) * 360 + t * 400) % 360
                jc = hsv_to_bgr(hue, 1.0, 1.0)
            else:
                jc = color
            cv2.circle(frame, (int(x*w), int(y*h)), 8, jc, -1, cv2.LINE_AA)

# ── Detection ─────────────────────────────────────────────────────────────────

def has_upper_body(kp, threshold=0.3):
    if kp is None:
        return False
    upper = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW]
    return sum(1 for i in upper if kp[i][2] > threshold) >= 3

def is_double_bicep_flex(kp, threshold=0.25):
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

    shoulder_w = abs(ls_x - rs_x)
    if shoulder_w < 0.05:
        return False

    shoulder_mid_y = (ls_y + rs_y) / 2

    # Elbows should be out wide
    left_wide  = le_x < ls_x + shoulder_w * 0.1
    right_wide = re_x > rs_x - shoulder_w * 0.1
    if not (left_wide and right_wide):
        return False

    # Elbows near shoulder height
    if abs(le_y - shoulder_mid_y) > shoulder_w * 1.2: return False
    if abs(re_y - shoulder_mid_y) > shoulder_w * 1.2: return False

    # Wrists above elbows (flexed up)
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

    assets = load_assets()

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

    # Screen
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

    # Pre-scale avatars at screen resolution
    av_h = int(screen_h * 0.55)
    avatars = {k: resize_to_h(img, av_h) for k, img in assets.items()
               if k != "logo" and img is not None}

    # Logo
    logo_img = None
    if assets.get("logo") is not None:
        logo_img = invert_rgb(resize_to_h(assets["logo"], int(screen_h * 0.18)))

    # Camera → screen scaling
    s = max(screen_w / cam_w, screen_h / cam_h)
    sw, sh = int(cam_w * s), int(cam_h * s)
    ox, oy = (sw - screen_w) // 2, (sh - screen_h) // 2

    # Window
    win = "XL Fitness Smart Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # State
    keypoints           = None
    smoothed_kp         = None     # EMA-smoothed keypoints for stable rendering
    kp_last_seen        = np.zeros(17)  # timestamp per joint when last had good confidence
    celebrating_until   = 0.0
    cooldown_until      = 0.0
    current_hype        = ""
    idle_cycle_started  = time.time()  # resets after each celebration
    was_celebrating     = False
    pulse_t             = 0.0

    EMA_ALPHA     = 0.55    # how fast smoothed keypoints track raw (higher = snappier)
    KP_HOLD_SECS  = 0.35    # keep drawing a joint this long after it drops below threshold

    frame_n = 0
    fps_t, fps_count, fps_disp = time.time(), 0, 0
    failed_reads = 0          # consecutive failed camera reads
    last_good_frame = time.time()
    print("\n✓ Running — strike a double bicep flex! Press Q to quit.\n")

    while True:
        # ── Resilient camera read ──────────────────────
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[mirror] camera read error: {e}")
            ret, frame = False, None

        if not ret or frame is None:
            failed_reads += 1
            # If camera has been dead for 3s, try to reopen it
            if time.time() - last_good_frame > 3.0:
                print("[mirror] camera stalled — reopening...")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(0)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS,          30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                last_good_frame = time.time()
            time.sleep(0.05)
            continue

        failed_reads = 0
        last_good_frame = time.time()
        frame_n += 1
        fps_count += 1

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # Inference every frame for smooth tracking (MoveNet Lightning is fast enough on Pi 4)
        try:
            keypoints = run_movenet(interp, frame)
        except Exception as e:
            print(f"[mirror] inference error: {e}")
            keypoints = None

        # ── Smooth keypoints + hold briefly when confidence dips ──
        now_k = time.time()
        if keypoints is not None:
            if smoothed_kp is None:
                smoothed_kp = keypoints.copy()
                kp_last_seen[:] = now_k
            else:
                for i in range(17):
                    thr = JOINT_THRESHOLDS.get(i, 0.3)
                    if keypoints[i][2] >= thr:
                        # good read — EMA-smooth position, refresh last-seen
                        smoothed_kp[i][0] = EMA_ALPHA * keypoints[i][0] + (1 - EMA_ALPHA) * smoothed_kp[i][0]
                        smoothed_kp[i][1] = EMA_ALPHA * keypoints[i][1] + (1 - EMA_ALPHA) * smoothed_kp[i][1]
                        smoothed_kp[i][2] = keypoints[i][2]
                        kp_last_seen[i]   = now_k
                    elif now_k - kp_last_seen[i] < KP_HOLD_SECS:
                        # recently had good read — keep position, decay confidence gently
                        smoothed_kp[i][2] = max(smoothed_kp[i][2] * 0.9, thr)
                    else:
                        # stale — let confidence drop so draw_skeleton skips it
                        smoothed_kp[i][2] = 0.0

        now         = time.time()
        celebrating = now < celebrating_until
        # Use smoothed keypoints for both display and detection — stops flicker
        kp_for_use  = smoothed_kp if smoothed_kp is not None else keypoints
        person_here = has_upper_body(kp_for_use)

        # Restart idle cycle the moment celebration ends
        if was_celebrating and not celebrating:
            idle_cycle_started = now
        was_celebrating = celebrating

        # Flex detection (only when not celebrating + cooldown expired)
        if not celebrating and now >= cooldown_until:
            if is_double_bicep_flex(kp_for_use):
                celebrating_until = now + CELEBRATION_SECONDS
                cooldown_until    = celebrating_until + COOLDOWN_SECONDS
                current_hype      = random.choice(HYPE_MSGS)
                pulse_t           = now
                celebrating       = True
                was_celebrating   = True

        # Scale camera to screen
        display = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        display = display[oy:oy+screen_h, ox:ox+screen_w]

        # Skeleton (rainbow when celebrating) — use smoothed keypoints
        draw_skeleton(display, kp_for_use, screen_w, screen_h,
                      color=(51, 87, 255), rainbow=celebrating, t=now)

        # Logo top-centre (always)
        if logo_img is not None:
            lx = (screen_w - logo_img.shape[1]) // 2
            overlay_png(display, logo_img, lx, 20)

        # ── CELEBRATING ────────────────────────────────
        if celebrating:
            # Pulsing hype text (rainbow color cycles with the skeleton)
            hype_hue = (now * 200) % 360
            hype_color = hsv_to_bgr(hype_hue, 1.0, 1.0)
            pulse = 1.0 + 0.15 * abs(np.sin((now - pulse_t) * 6))
            text_centred(display, current_hype,
                         int(screen_h * 0.48),
                         scale=4.5 * pulse, color=hype_color, thickness=8)

            # Celebrating avatar bottom-right
            av = avatars.get("celebrating")
            if av is not None:
                ax = screen_w - av.shape[1] - 20
                ay = screen_h - av.shape[0] + 40
                overlay_png(display, av, ax, ay)

        # ── IDLE / PROMPTING ───────────────────────────
        else:
            # Two-phase prompt:
            #   Phase 1 (first STRIKE_PROMPT_SECONDS): "Strike a pose!" + wave/point
            #   Phase 2 (after that): "Like this!" + pose-example avatar
            elapsed = now - idle_cycle_started
            if elapsed < STRIKE_PROMPT_SECONDS:
                prompt_text = "Strike a pose!"
                av_key = "point" if person_here else "wave"
            else:
                prompt_text = "Like this!"
                av_key = "pose"

            av = avatars.get(av_key)
            if av is None:
                av = avatars.get("pose")
            if av is not None:
                ax = screen_w - av.shape[1] - 20
                ay = screen_h - av.shape[0] + 40
                overlay_png(display, av, ax, ay)

                # Speech bubble above avatar
                bx = ax + av.shape[1] // 2
                by = ay - 30
                draw_bubble(display, prompt_text, bx, by, scale=1.2, thickness=2)

        # FPS
        if now - fps_t >= 1.0:
            fps_disp, fps_count, fps_t = fps_count, 0, now
        cv2.putText(display, f"{fps_disp}fps",
                    (10, screen_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        try:
            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
        except Exception as e:
            print(f"[mirror] display error: {e}")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    # Watchdog: if main() crashes unexpectedly, restart after a short delay
    # instead of exiting. Keeps the mirror running unattended on the wall.
    while True:
        try:
            main()
            break   # clean exit (user pressed Q) — don't restart
        except KeyboardInterrupt:
            print("\nInterrupted — bye.")
            break
        except Exception as e:
            print(f"\n[mirror] FATAL: {e}")
            print("[mirror] restarting in 3s...")
            time.sleep(3)
