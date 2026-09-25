#!/bin/sh
# Phase 7: VRAM / expert-cache / KV experiments. 256-token decode per config,
# 32-token golden check where the resident set or numerics change.
cd /home/noorazman/dsh/strata/Strata || exit 1
run() { # label extra-flags workers
  W=${3:-28}
  python3 bench/v100/bench.py "$1" --gpu 0 --workers "$W" --max-new 256 --stats \
    --strata-flags="$2" 2>&1 | grep -E "^\[bench\]|cache hits|pool " | head -7
  python3 bench/v100/bench.py "${1}-32tok" --gpu 0 --workers "$W" --max-new 32 \
    --strata-flags="$2" 2>&1 | grep -E "^\[bench\]" | tail -1
}
run "opt-reserve300"      "--vram-reserve-mib 300"
run "opt-perlayer"        "--expert-cache-per-layer"
run "opt-kvint8"          "--kv int8"
run "opt-workers24"       "" 24
echo "VRAM SWEEP DONE"
