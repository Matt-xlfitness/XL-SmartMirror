#!/bin/bash
# ══════════════════════════════════════════════════
#  XL Smart Mirror — health check (run daily by the timer)
#  Confirms the service is active AND actually rendering
#  (fresh heartbeat). Auto-restarts + logs if not.
# ══════════════════════════════════════════════════
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

LOG="$HOME/xlf_logs/healthcheck.log"
HB="$HOME/.cache/xl-mirror/heartbeat"
mkdir -p "$HOME/xlf_logs"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

need_restart=0
reason=""

# 1) Is the service active?
if ! systemctl --user is-active --quiet xl-mirror; then
    need_restart=1
    reason="service not active"
fi

# 2) Is it actually alive? (heartbeat fresh within 60s)
if [ "$need_restart" -eq 0 ]; then
    if [ -f "$HB" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
        if [ "$age" -gt 60 ]; then
            need_restart=1
            reason="frozen — heartbeat stale (${age}s)"
        fi
    fi
    # No heartbeat file yet (e.g. just updated) is not itself a failure.
fi

if [ "$need_restart" -eq 1 ]; then
    echo "$(ts) UNHEALTHY: $reason -> restarting" >> "$LOG"
    systemctl --user restart xl-mirror
    sleep 8
    if systemctl --user is-active --quiet xl-mirror; then
        echo "$(ts) RECOVERED: service active after restart" >> "$LOG"
    else
        echo "$(ts) FAILED to recover — still not active" >> "$LOG"
    fi
else
    echo "$(ts) OK: service active, heartbeat fresh" >> "$LOG"
fi
