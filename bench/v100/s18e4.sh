#!/bin/bash
# Stage 1.8 — QSA attention A/B campaign on the E4 plumbing (commit 00e1609).
#
# Arms (per cycle, round-robin interleaved, drop_caches between runs):
#   base         fast attention + fast selection (production default)
#   nofastattn   --no-fast-attn      slow (P2.S2) gather + one-block-per-head attention
#   nofastsel    --no-fast-select    slow (P2.S2) per-cell scores + bit-serial cell top-k
#
# Config: Stage 1.7 protocol (canonical 28-token prompt, --kv fp16 --max-context 8192,
# --spec 4 --spec-min-p 0.5 --expert-cache auto --mtp mtp/rt --pool-workers 24,
# 256-token decode, GPU0).  A second campaign variant runs the production config
# (--kv int8 --max-context 32768).
#
# Usage: s18e4.sh [cycles] [kv] [max-context] [max-new]
set -u
cd /home/noorazman/dsh/strata/Strata
CYCLES="${1:-3}"
KV="${2:-fp16}"
MAXCTX="${3:-8192}"
MAXNEW="${4:-256}"
i=0
for c in $(seq 1 "$CYCLES"); do
  for arm in base nofastattn nofastsel; do
    i=$((i+1))
    case "$arm" in
      base) extra="";;
      nofastattn) extra="--no-fast-attn";;
      nofastsel) extra="--no-fast-select";;
    esac
    echo "=== [run $i/$((CYCLES*3))] cycle $c arm $arm ($KV ctx$MAXCTX $extra) ==="
    if [ -n "$extra" ]; then
      python3 bench/v100/bench.py "s18e4-${KV}-${arm}-$c" --gpu 0 --workers 24 \
        --max-new "$MAXNEW" --kv "$KV" --max-context "$MAXCTX" --stats \
        --strata-flags="$extra"
    else
      python3 bench/v100/bench.py "s18e4-${KV}-${arm}-$c" --gpu 0 --workers 24 \
        --max-new "$MAXNEW" --kv "$KV" --max-context "$MAXCTX" --stats
    fi
    echo "rc=$?"
    echo mustoe8 | sudo -S -p '' sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
    sleep 3
  done
done
echo "S18E4 DONE ($KV ctx$MAXCTX cycles=$CYCLES)"
