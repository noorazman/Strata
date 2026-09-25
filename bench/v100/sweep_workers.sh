#!/bin/sh
# Stage 1.1 pool-worker sweep: 256-token sustained decode per worker count.
# One engine run per config, sequential on GPU0. ~40 s per run.
cd /home/noorazman/dsh/strata/Strata || exit 1
for w in 8 12 16 20 24 28 32 40; do
  echo "=== sweep workers=$w ==="
  python3 bench/v100/bench.py "sweep-w${w}-256tok" --gpu 0 --workers "$w" \
    --max-new 256 --stats 2>&1 | grep -E "^\[bench\]" | head -8
done
echo "SWEEP DONE"
