#!/bin/bash
# ══════════════════════════════════════════════════
#  XL Fitness Smart Mirror — One-Click Setup
#  Raspberry Pi 5 + Hailo-8L AI Hat (NPU)
#  Raspberry Pi OS Bookworm (64-bit)
# ══════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════════════"
echo "  XL Fitness Smart Mirror — Setup (Pi 5)"
echo "══════════════════════════════════════════════════"
echo ""

# ── System updates ────────────────────────────────
echo "[1/5] Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# ── Core Python deps (MoveNet/OpenCV — always-works backend) ──
echo "[2/5] Installing core Python dependencies..."
pip3 install tflite-runtime opencv-python "numpy<2" --break-system-packages

# ── Optional Hailo NPU stack ──────────────────────
# The Hailo backend auto-activates if these are present AND a pose HEF exists
# at assets/yolov8s_pose.hef. Safe to skip — the app falls back to MoveNet.
echo "[3/5] Installing Hailo NPU stack (optional)..."
sudo apt install -y hailo-all || echo "  · hailo-all not available — skipping (MoveNet fallback will be used)"

# ── Clone / update the repo ───────────────────────
echo "[4/5] Fetching XL Smart Mirror..."
cd ~
if [ -d "XL-SmartMirror" ]; then
    echo "  Repo exists — pulling latest..."
    cd XL-SmartMirror && git pull
else
    git clone https://github.com/Matt-xlfitness/XL-SmartMirror.git
    cd XL-SmartMirror
fi
chmod +x start.sh

# Assets (avatars + MoveNet model) ship in the repo's assets/ folder — no
# download step needed. Verify they're present.
echo "  Checking assets..."
for f in SMARTMIRROR.png XLAvatar-Wave.png XLAvatar-Point.png \
         XLAvatar-01Pose.png XLAvatar-Celebrating.png XLAvatar-ThumbsUp.png \
         movenet_lightning.tflite; do
    [ -f "assets/$f" ] && echo "    ✓ $f" || echo "    ✗ MISSING assets/$f"
done

# ── Autostart on boot ─────────────────────────────
echo "[5/5] Setting up autostart on boot..."
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/smart-mirror.desktop" << EOF
[Desktop Entry]
Type=Application
Name=XL Smart Mirror
Exec=bash -c "sleep 10 && $HOME/XL-SmartMirror/start.sh"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF

echo ""
echo "══════════════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Run now:     ~/XL-SmartMirror/start.sh"
echo "  Or direct:   cd ~/XL-SmartMirror && python3 smart_mirror.py"
echo "  Auto-start:  Enabled (pulls latest + launches on boot)"
echo "  Disable:     rm ~/.config/autostart/smart-mirror.desktop"
echo "  Logs:        tail -f ~/xlf_logs/smart_mirror.log"
echo ""
echo "  Press Q or ESC to quit the mirror."
echo "══════════════════════════════════════════════════"
