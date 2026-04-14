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
pip3 install mediapipe opencv-python --break-system-packages

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

# ── Download assets ───────────────────────────────
echo "[4/5] Pre-downloading avatar assets..."
python3 -c "
import smart_mirror
smart_mirror.load_assets()
print('Assets ready!')
"

# ── Setup autostart (optional) ────────────────────
echo "[5/5] Setting up autostart on boot..."
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/smart-mirror.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=XL Smart Mirror
Exec=bash -c "sleep 10 && cd ~/XL-SmartMirror && python3 smart_mirror.py"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF

echo ""
echo "══════════════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════════════"
echo ""
echo "  To run now:    cd ~/XL-SmartMirror && python3 smart_mirror.py"
echo "  Auto-start:    Enabled (runs on boot after 10s)"
echo "  To disable:    rm ~/.config/autostart/smart-mirror.desktop"
echo ""
echo "  Press Q or ESC to quit the mirror app."
echo "══════════════════════════════════════════════════"
