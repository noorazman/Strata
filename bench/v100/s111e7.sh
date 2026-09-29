#!/bin/bash
# Stage 1.11 E7 A/B: two-stream expert staging DMA (STRATA_TWO_STREAM_DMA=1, opt-in).
#
#   s111e7.sh gate 16384     engine-level gate at 16K with two-stream-DMA ON (32-tok golden prefix +
#                            256-tok determinism x2, int8 KV, production flags) + decode rate
#   s111e7.sh ctx <C>        serve-path TTFT/prefill/decode point at max-context C, two-stream-DMA ON
#                            (s110.sh ctx with EXTRA_ENV=STRATA_TWO_STREAM_DMA=1)
#
# The OFF side is the fresh s110.sh ctx <C> baseline (same build, default env).
set -u
cd /home/noorazman/dsh/strata/Strata

die() { echo "[$(date +%H:%M:%S)] FATAL: $*"; exit 1; }
pgrep -x strata >/dev/null && die "a strata engine is already running"
mib=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$mib" -gt 2048 ] && die "GPU0 not idle (used $mib MiB)"
[ "$(systemctl is-active strata.service 2>/dev/null)" = "active" ] && die "strata.service is active"

case "${1:-}" in
  gate)
    c="${2:-16384}"
    echo "[$(date +%H:%M:%S)] E7 gate ctx=$c (STRATA_TWO_STREAM_DMA=1)"
    python3 bench/v100/bench.py "s111e7-g${c}-g32" --gpu 0 --workers 24 --max-new 32 --stats \
      --kv int8 --max-context "$c" --env STRATA_TWO_STREAM_DMA=1 > "Logs/gpu/s111e7-g${c}-g32.out" 2>&1
    echo "g32 rc=$?"
    python3 bench/v100/bench.py "s111e7-g${c}-det1" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv int8 --max-context "$c" --env STRATA_TWO_STREAM_DMA=1 > "Logs/gpu/s111e7-g${c}-det1.out" 2>&1
    echo "det1 rc=$?"
    python3 bench/v100/bench.py "s111e7-g${c}-det2" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv int8 --max-context "$c" --env STRATA_TWO_STREAM_DMA=1 > "Logs/gpu/s111e7-g${c}-det2.out" 2>&1
    echo "det2 rc=$?"
    python3 bench/v100/s18check.py "Logs/benchmarks/s111e7-g${c}-det1.json" "Logs/benchmarks/s111e7-g${c}-det2.json"
    python3 bench/v100/s18check.py --prefix "Logs/benchmarks/s111e7-g${c}-g32.json" 32
    python3 - "$c" <<'EOF'
import json, sys
c = sys.argv[1]
d = json.load(open(f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s111e7-g{c}-det1.json"))
for k in ("prefill_tps", "decode_tps", "spec_accept", "spec_tpr", "mtp_draft_ms_round", "gpu_only_full_ms"):
    print(f"{k}: {d.get(k)}")
EOF
    echo "[$(date +%H:%M:%S)] E7 GATE DONE"
    ;;
  ctx)
    c="${2:-}"
    [ -n "$c" ] || die "usage: s111e7.sh ctx <max-context>"
    EXTRA_ENV="STRATA_TWO_STREAM_DMA=1" bash bench/v100/s110.sh ctx "$c"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
