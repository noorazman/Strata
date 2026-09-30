#!/usr/bin/env python3
"""bench/v100/s114_pool_sim.py - Stage 1.14: fp16 dequant-reuse pool hit-rate simulation.

Reads the Stage 1.13 STRATA_MOE_ROUTE_DUMP files ("<chunk> <layer> <nr> <e1> <c1> ...", one
line per (chunk, MoE layer) section = one expert-set execution, experts processed in id
order) and simulates, for pool sizes P, the fraction of expert RUNS whose (layer, expert)
pair is already dequantized in an fp16 weight pool:

  - static top-P : the pair set is fixed from the profile (never evicted) -> hit iff the
    pair is in the global top-P by total decision count. This is the implementable policy
    (the STRP profile mechanism already ships such a ranked list).
  - LRU-P        : a real LRU pool of P entries over the exact execution sequence.
  - per-layer    : each layer gets an independent static top-P_l set (P_l = P / n_layers),
    which fits the access structure (layer c recurs 48 sections after layer c-1, so a
    global LRU of P < ~17K evicts the same layer's set before it returns).

The dequant cost is per RUN (resident or streamed, same kernels), so a pool hit removes
one dequant pair (gu + flat) per hit run.
"""
import sys
from collections import Counter, defaultdict

def load(path):
    seq = []          # execution order: (layer, expert)
    cnt = Counter()   # (layer, expert) -> runs
    lines = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            chunk, layer = int(p[0]), int(p[1])
            nr = int(p[2])
            for i in range(nr):
                e = int(p[3 + 2 * i])
                seq.append((layer, e))
                cnt[(layer, e)] += 1
            lines += 1
    return seq, cnt, lines

def static_top(seq, cnt, P):
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    top = set(p for p, _ in ranked[:P])
    hits = sum(1 for p in seq if p in top)
    return hits, len(ranked)

def lru(seq, P):
    from collections import OrderedDict
    od = OrderedDict()
    hits = 0
    for p in seq:
        if p in od:
            od.move_to_end(p)
            hits += 1
        else:
            if len(od) == P:
                od.popitem(last=False)
            od[p] = 1
    return hits

def per_layer_static(seq, cnt, P):
    n_l = max(l for l, _ in cnt) + 1
    Pl = max(1, P // n_l)
    per = defaultdict(Counter)
    for (l, e), c in cnt.items():
        per[l][e] = c
    hits = 0
    for l in range(n_l):
        top = set(e for e, _ in sorted(per[l].items(), key=lambda kv: (-kv[1], kv[0]))[:Pl])
        hits += sum(1 for (ll, e) in seq if ll == l and e in top)
    return hits

def main():
    total = 0
    for path in sys.argv[1:]:
        seq, cnt, lines = load(path)
        n = len(seq)
        total += n
        uniq = len(cnt)
        # skew stats
        ranked = sorted(cnt.values(), reverse=True)
        def frac(k):
            return sum(ranked[:k]) / n
        print(f"== {path}")
        print(f"   sections={lines} runs={n} unique_pairs={uniq} re-routing={n/uniq:.2f}x")
        print(f"   top-1% pairs cover {frac(max(1,uniq//100))*100:.1f}% of runs; "
              f"top-5% {frac(uniq*5//100)*100:.1f}%; top-10% {frac(uniq//10)*100:.1f}%")
        print(f"   {'P':>6} | {'static top-P':>12} | {'LRU-P':>8} | {'per-layer P':>12} | GB @9.83MB/entry")
        for P in (128, 256, 512, 1024, 2048, 4096, 8192):
            s, _ = static_top(seq, cnt, P)
            l = lru(seq, P)
            pl = per_layer_static(seq, cnt, P)
            gb = P * 9.83 / 1024
            print(f"   {P:6d} | {s/n*100:11.1f}% | {l/n*100:7.1f}% | {pl/n*100:11.1f}% | {gb:6.2f}")
    print(f"total runs across dumps: {total}")

if __name__ == "__main__":
    main()
