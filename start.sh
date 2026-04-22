#!/bin/bash
# XL Fitness Smart Mirror — one-command launcher.
# Usage:  ~/XL-SmartMirror/start.sh

BRAND="[mirror]"
REPO_DIR="$HOME/XL-SmartMirror"
LOG_DIR="$HOME/xlf_logs"
LOG_FILE="$LOG_DIR/smart_mirror.log"

mkdir -p "$LOG_DIR"

# Kill any existing instance
pkill -f smart_mirror.py 2>/dev/null
sleep 1

# Pull latest (quietly — errors surface in the log)
cd "$REPO_DIR" && git pull --quiet 2>>"$LOG_FILE"

# Launch in background, keep PID
DISPLAY=:0 python3 "$REPO_DIR/smart_mirror.py" >>"$LOG_FILE" 2>&1 &
MIRROR_PID=$!
sleep 1

# Runtime banner
echo "$BRAND ═══════════════════════════════════════════════════════"
echo "$BRAND   XL FITNESS SMART MIRROR — RUNNING"
echo "$BRAND ═══════════════════════════════════════════════════════"
echo "$BRAND"
echo "$BRAND   💪  DETECTS:"
echo "$BRAND       double bicep flex → rainbow celebration"
echo "$BRAND"
echo "$BRAND   smart_mirror.py  PID $MIRROR_PID"
echo "$BRAND   Logs   →  tail -f $LOG_FILE"
echo "$BRAND   Stop   →  pkill -f smart_mirror.py"
echo "$BRAND"
echo "$BRAND ═══════════════════════════════════════════════════════"
