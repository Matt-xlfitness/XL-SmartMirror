#!/bin/bash
# ══════════════════════════════════════════════════
#  XL Smart Mirror — install as an always-on service
#  Starts on boot, auto-restarts on crash, runs until
#  you explicitly stop it. Run once:  ./install_service.sh
# ══════════════════════════════════════════════════
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_DIR="$HOME/.config/systemd/user"
UNIT="$SERVICE_DIR/xl-mirror.service"

echo "Repo: $REPO_DIR"
mkdir -p "$SERVICE_DIR"

# Kill any manual / nohup instance so it doesn't fight over the camera.
pkill -f "smart_mirror.py" 2>/dev/null || true
# Remove the old desktop-autostart entry (superseded by this service).
rm -f "$HOME/.config/autostart/smart-mirror.desktop" 2>/dev/null || true
sleep 1

cat > "$UNIT" <<EOF
[Unit]
Description=XL Fitness Smart Mirror
After=graphical-session.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=DISPLAY=:0
Environment=WAYLAND_DISPLAY=wayland-0
ExecStart=/usr/bin/python3 $REPO_DIR/smart_mirror.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
echo "Wrote $UNIT"

# Let the user's services run at boot without an interactive login.
sudo loginctl enable-linger "$USER" || true

chmod +x "$REPO_DIR/healthcheck.sh" 2>/dev/null || true

# ── Daily 07:00 health check (timer + oneshot service) ──
cat > "$SERVICE_DIR/xl-mirror-healthcheck.service" <<EOF
[Unit]
Description=XL Smart Mirror daily health check

[Service]
Type=oneshot
ExecStart=$REPO_DIR/healthcheck.sh
EOF

cat > "$SERVICE_DIR/xl-mirror-healthcheck.timer" <<EOF
[Unit]
Description=Run XL Smart Mirror health check daily at 07:00

[Timer]
OnCalendar=*-*-* 07:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
echo "Wrote health-check timer (daily 07:00)"

systemctl --user daemon-reload
systemctl --user enable xl-mirror.service
systemctl --user restart xl-mirror.service
systemctl --user enable --now xl-mirror-healthcheck.timer
sleep 4

echo
systemctl --user status xl-mirror.service --no-pager || true
echo
echo "-- health-check timer --"
systemctl --user list-timers xl-mirror-healthcheck.timer --no-pager || true
echo
echo "══════════════════════════════════════════════════"
echo "  Installed. It now starts on boot & auto-restarts."
echo "══════════════════════════════════════════════════"
echo "  Stop (until you restart it): systemctl --user stop xl-mirror"
echo "  Start again:                 systemctl --user start xl-mirror"
echo "  Restart:                     systemctl --user restart xl-mirror"
echo "  Live logs:                   journalctl --user -u xl-mirror -f"
echo "  Turn OFF boot start:         systemctl --user disable xl-mirror"
echo "  Turn ON  boot start:         systemctl --user enable  xl-mirror"
echo "  Run health check now:        $REPO_DIR/healthcheck.sh"
echo "  Health-check history:        cat ~/xlf_logs/healthcheck.log"
echo "══════════════════════════════════════════════════"
