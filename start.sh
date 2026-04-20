#!/bin/bash
# One command to start the Smart Mirror

# Kill any existing instance
pkill -f smart_mirror.py 2>/dev/null
sleep 1

# Pull latest
cd ~/XL-SmartMirror && git pull --quiet

# Launch
echo "[mirror] ═══════════════════════════════════════════════════════"
echo "[mirror]   XL FITNESS SMART MIRROR — STARTING"
echo "[mirror] ═══════════════════════════════════════════════════════"
echo "[mirror]"

DISPLAY=:0 python3 smart_mirror.py
