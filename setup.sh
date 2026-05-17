#!/bin/bash
# ══════════════════════════════════════════════════
#  XL Fitness Smart Mirror — One-Click Setup
#  Raspberry Pi 4 + Bookworm (Legacy 64-bit)
#  Requires 16GB+ SD card
# ══════════════════════════════════════════════════

set -e

echo "══════════════════════════════════════════════════"
echo "  XL Fitness Smart Mirror — Setup"
echo "══════════════════════════════════════════════════"
echo ""

# ── System updates ────────────────────────────────
echo "[1/5] Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# ── Install Python dependencies ───────────────────
echo "[2/5] Installing Python dependencies..."
pip3 install tflite-runtime opencv-python "numpy<2" --break-system-packages

# ── Clone the repo ────────────────────────────────
echo "[3/5] Downloading XL Smart Mirror..."
cd ~
if [ -d "XL-SmartMirror" ]; then
    echo "  Repo already exists — pulling latest..."
    cd XL-SmartMirror && git pull
else
    git clone https://github.com/Matt-xlfitness/XL-SmartMirror.git
    cd XL-SmartMirror
fi

chmod +x start.sh

# ── Download assets + model ───────────────────────
echo "[4/5] Pre-downloading assets and MoveNet model..."
python3 -c "
import smart_mirror
smart_mirror.load_assets()
smart_mirror.load_movenet()
print('Assets & model ready!')
"

# ── Setup autostart ───────────────────────────────
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
echo "  To run now:    ~/XL-SmartMirror/start.sh"
echo "  Auto-start:    Enabled (pulls latest + launches on boot)"
echo "  To disable:    rm ~/.config/autostart/smart-mirror.desktop"
echo "  Logs:          tail -f ~/xlf_logs/smart_mirror.log"
echo ""
echo "  Press Q or ESC to quit the mirror app."
echo "══════════════════════════════════════════════════"
