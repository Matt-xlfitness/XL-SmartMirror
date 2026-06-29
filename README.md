# XL Fitness Smart Mirror

Interactive "smart mirror" kiosk for the XL Fitness gym.

A live camera feed runs fullscreen on a wall-mounted TV. A pose-detection model
watches for people stepping up; an on-screen avatar greets them, shows a double
bicep flex, and when they nail the pose — **BIG CELEBRATION**.

## Hardware

- **Raspberry Pi 5** with active cooling
- **Hailo-8L AI Hat** (NPU) — optional, accelerates inference
- USB webcam mounted **low, pointing up ~45°** (detection is upper-body only)
- Landscape TV via HDMI (kiosk — no keyboard/mouse)
- Raspberry Pi OS Bookworm (64-bit), 32GB+ SD card

## Pose backends

Auto-selected at startup, best available first:

| Priority | Backend | Notes |
|----------|---------|-------|
| 1 | **Hailo NPU** | `hailo_platform` SDK + a pose HEF at `assets/yolov8s_pose.hef`. Fastest, multi-person. Finish-tune the output decode on-device. |
| 2 | **MediaPipe Pose** | `mediapipe` (Lite). Needs a **Python 3.11 venv** — MediaPipe has no 3.13 wheels. |
| 3 | **MoveNet Lightning** | `tflite-runtime`. Ships in `assets/`, runs anywhere — the guaranteed fallback. |

## One-line install

SSH into the Pi and run:

```bash
curl -sL https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/setup.sh | bash
```

Installs system + Python deps, the optional Hailo stack, clones the repo, and
sets up autostart on boot.

## Manual install

```bash
pip3 install tflite-runtime opencv-python "numpy<2" --break-system-packages
git clone https://github.com/Matt-xlfitness/XL-SmartMirror.git
cd XL-SmartMirror
python3 smart_mirror.py
```

### MediaPipe fallback (Python 3.11 venv)

```bash
python3.11 -m venv ~/mirror-venv
source ~/mirror-venv/bin/activate
pip install mediapipe opencv-python "numpy<2"
python smart_mirror.py
```

## Run

```bash
cd ~/XL-SmartMirror && python3 smart_mirror.py
DISPLAY=:0 python3 smart_mirror.py     # over SSH
```

Press **Q** or **ESC** to quit.

## State machine

| State | Trigger | Avatar | Copy |
|-------|---------|--------|------|
| **Idle** | no one | Wave | "Step up & strike a pose!" |
| **Greeting** | person held 1.5s | Wave | "Hey! Welcome to XL Fitness!" |
| **Prompt** | after 2.5s | Point | "Can you do THIS? Strike a pose!" |
| **Waiting** | after 2s | 01Pose (example) | "Flex like this! 💪" |
| **Celebrate** | flex held 1.5s | Celebrating | random hype ("BEAST MODE!"…) |
| **Compliment** | after 3.5s | Thumbs Up | "You look incredible! 💪" |
| **Done** | after 3s | Thumbs Up | — (waits for person to leave) |

Hysteresis throughout (presence 1.5s, absence 3s, pose 1.5s) prevents flicker.
Any one person in frame doing the flex triggers the celebration.

## Tuning

All timings and look-and-feel live in clearly-marked constants at the top of
[`smart_mirror.py`](smart_mirror.py) — confirm windows, avatar/logo sizes,
hype messages, and `is_double_bicep_flex()` thresholds.

## Assets

Avatars, logo, and the MoveNet model live in [`assets/`](assets/) next to the
script and are loaded locally (no download). All avatar PNGs are RGBA with real
transparency and are alpha-composited onto the feed.

| File | Used in |
|------|---------|
| `SMARTMIRROR.png` | logo, top-centre, always |
| `XLAvatar-Wave.png` | Idle, Greeting |
| `XLAvatar-Point.png` | Prompt |
| `XLAvatar-01Pose.png` | Waiting (flex example) |
| `XLAvatar-Celebrating.png` | Celebrate |
| `XLAvatar-ThumbsUp.png` | Compliment, Done |
| `movenet_lightning.tflite` | MoveNet backend model |
