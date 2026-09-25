#!/bin/sh
# Speculative-window sweep: max window and acceptance threshold.
# 256-token decode per config; 32-token run for the golden prefix check.
cd /home/noorazman/dsh/strata/Strata || exit 1
run() { # label spec minp
  python3 bench/v100/bench.py "$1" --gpu 0 --workers 28 --max-new 256 --stats \
    --strata-flags="--spec $2 --spec-min-p $3" 2>&1 | grep -E "^\[bench\]|spec:" | head -6
  python3 bench/v100/bench.py "${1}-32tok" --gpu 0 --workers 28 --max-new 32 \
    --strata-flags="--spec $2 --spec-min-p $3" 2>&1 | grep -E "^\[bench\]" | tail -1
}
run "opt-spec2"     2 0.5
run "opt-spec3"     3 0.5
run "opt-spec4-p7"  4 0.7
echo "SPEC SWEEP DONE"
