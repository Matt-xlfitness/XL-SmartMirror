#!/bin/bash
# ══════════════════════════════════════════════════
#  XL Smart Mirror — save Wi-Fi permanently (kiosk mode)
#  Usage:  ./setup_wifi.sh "SSID" "PASSWORD"
#  Auto-reconnects on boot, retries forever, never prompts again.
# ══════════════════════════════════════════════════
SSID="$1"
PSK="$2"
if [ -z "$SSID" ] || [ -z "$PSK" ]; then
    echo "Usage: $0 \"SSID\" \"PASSWORD\""
    exit 1
fi

# Connect (creates the profile with the password stored).
nmcli device wifi connect "$SSID" password "$PSK" \
    || { echo "Connect failed — check SSID/password/range."; exit 1; }

# Harden for an always-on kiosk:
#   autoconnect yes            -> comes up on every boot
#   autoconnect-retries 0      -> retry forever if the router drops
#   autoconnect-priority 100   -> prefer this network
#   psk-flags 0                -> secret stored by the system (no agent popup, ever)
nmcli connection modify "$SSID" \
    connection.autoconnect yes \
    connection.autoconnect-retries 0 \
    connection.autoconnect-priority 100 \
    802-11-wireless-security.psk-flags 0

nmcli connection up "$SSID" >/dev/null 2>&1

echo "✓ '$SSID' saved. It will auto-reconnect on boot and never prompt again."
echo "  Internet check:"; ping -c2 -W2 8.8.8.8 >/dev/null 2>&1 && echo "  ✓ online" || echo "  ✗ still offline"
