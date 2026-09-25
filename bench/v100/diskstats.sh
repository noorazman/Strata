#!/usr/bin/env bash
# diskstats.sh <label> [--drop-caches]
#
# Stage 1.2A: capture nvme0n1 counters around a bench run so the SSD vs RAM
# comparison can show the actual device-level I/O reduction (Phase 9).
#
#   bench/v100/diskstats.sh <label> --drop-caches
#   python3 bench/v100/bench.py <label> ...
#   bench/v100/diskstats.sh <label>
#
# /proc/diskstats fields: 1 major, 2 minor, 3 name, 4 reads completed,
# 5 reads merged, 6 sectors read, 8 writes completed, 10 sectors written
# (1 sector = 512 B).
set -euo pipefail
REPO="/home/noorazman/dsh/strata/Strata"
SUDO_PW="mustoe8"
label="$1"
snap() { awk '$3 == "nvme0n1" {print $6, $10}' /proc/diskstats; }
if [ "${2:-}" = "--drop-caches" ]; then
    echo "$SUDO_PW" | sudo -S -p "" sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 1
    snap > "$REPO/Logs/cpu/$label.diskstats-before"
else
    snap > "$REPO/Logs/cpu/$label.diskstats-after"
    # derive the delta in the same call for convenience
    {
        read -r rb rbw < "$REPO/Logs/cpu/$label.diskstats-before"
        read -r ra raw < "$REPO/Logs/cpu/$label.diskstats-after"
        echo "$label: NVMe read delta = $((ra - rb)) sectors (=$(( (ra - rb) * 512 / 1048576 )) MiB), write delta = $((raw - rbw)) sectors"
    } >> "$REPO/Logs/cpu/$label.diskstats-after"
    cat "$REPO/Logs/cpu/$label.diskstats-after"
fi
