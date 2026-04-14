# XL Fitness Smart Mirror

Interactive fitness mirror powered by Raspberry Pi 4 + MediaPipe pose detection.

Detects when someone steps up, shows a reference pose, waits for you to flex, then hypes you up.

## Requirements

- Raspberry Pi 4
- Camera (USB or Pi Camera)
- TV/Monitor (connected via HDMI)
- **Raspberry Pi OS Bullseye (Legacy, 64-bit)** — required for MediaPipe compatibility

## One-Line Install

SSH into your Pi and run:

```bash
curl -sL https://raw.githubusercontent.com/Matt-xlfitness/XL-SmartMirror/main/setup.sh | bash
```

This will:
1. Update system packages
2. Install MediaPipe + OpenCV
3. Clone this repo
4. Download avatar assets
5. Set up autostart on boot

## Manual Install

```bash
pip3 install mediapipe-rpi4 opencv-python --break-system-packages
git clone https://github.com/Matt-xlfitness/XL-SmartMirror.git
cd XL-SmartMirror
python3 smart_mirror.py
```

## Run

```bash
cd ~/XL-SmartMirror
python3 smart_mirror.py
```

Press **Q** or **ESC** to quit.

## How It Works

1. **Idle** — "Step up & strike a pose!"
2. **Greeting** — Detects a person, welcomes them
3. **Show Pose** — Displays a reference flex pose
4. **Prompt** — "Now YOU do it! Arms out & flex!"
5. **Celebrate** — Detects your flex, hypes you up
6. **Compliment** — Random compliment, then resets

## Assets

Avatar images and logo are stored in `/assets` and auto-downloaded to `~/smart_mirror_assets/` on first run.
