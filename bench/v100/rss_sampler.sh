#!/usr/bin/env bash
# rss_sampler.sh <label>
#
# Stage 1.2A: sample the `strata` engine's VmRSS (kB) at 1 Hz into
# Logs/cpu/<label>.rss while a bench run is in flight.
#
#   bash bench/v100/rss_sampler.sh <label> &  SAMPLER=$!
#   python3 bench/v100/bench.py <label> ...
#   kill $SAMPLER
REPO="/home/noorazman/dsh/strata/Strata"
label="$1"
out="$REPO/Logs/cpu/$label.rss"
: > "$out"
while sleep 1; do
    for pid in $(pgrep -x strata 2>/dev/null); do
        rss=$(awk '/^VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -n "$rss" ] && echo "$(date +%s.%N) $rss" >> "$out"
    done
done
