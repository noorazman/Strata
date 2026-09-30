#!/usr/bin/env python3
"""Stage 1.13 E8 - expert-profile builder.

Aggregates STRATA_MOE_ROUTE_DUMP files (one or more workloads) into a token-count-weighted
ranking of (layer, expert) pairs and writes a STRP profile file for `--expert-profile`.

Route dump line formats (both accepted):
  new (Stage 1.13):  "<chunk> <layer> <nr> <e1> <c1> ... <en> <cn>"   (per-expert token counts)
  old (Stage 1.12):  "<chunk> <layer> <nr> <e1> ... <en>"             (bare routed set, weight 1)
Lines starting with '#' are comments.

Usage:
  e8_profile.py -o data/expert-profile-14k.bin --slots 14000 \
      data/expert-profile-dump-16k.txt data/expert-profile-dump-32k.txt
  e8_profile.py --check data/expert-profile.bin     # report geometry + size of an existing file

The ranking is count-descending with a deterministic (layer, expert) tie-break.  The coverage
curve reported is IN-SAMPLE (top-C of the same data); per-workload coverage of the union
ranking is the honest out-of-sample-ish number for the A/B workloads.
"""
import argparse
import os
import struct
import sys

MAGIC = b"STRP"


def parse_dump(path):
    """Return {(layer, expert): total_count} for one dump file."""
    counts = {}
    max_layer, max_expert = -1, -1
    n_lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            chunk, layer = int(toks[0]), int(toks[1])
            nr = int(toks[2])
            rest = toks[3:]
            if len(rest) == 2 * nr:
                max_layer = max(max_layer, layer)
                for i in range(nr):
                    e, c = int(rest[2 * i]), int(rest[2 * i + 1])
                    max_expert = max(max_expert, e)
                    counts[(layer, e)] = counts.get((layer, e), 0) + c
            elif len(rest) == nr:  # old (Stage 1.12) format: bare set, weight 1
                max_layer = max(max_layer, layer)
                for i in range(nr):
                    e = int(rest[i])
                    max_expert = max(max_expert, e)
                    counts[(layer, e)] = counts.get((layer, e), 0) + 1
            else:
                # incomplete line (the engine died mid-dump at shutdown): skip it
                continue
            n_lines += 1
    return counts, max_layer, max_expert, n_lines


def rank(counts, slots, n_layers, n_expert):
    pairs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return pairs[:slots]


def coverage(ranked_counts, ranked_pairs, c):
    """In-sample: share of total count covered by the top-c of `ranked_pairs`."""
    top = set(p for p, _ in ranked_pairs[:c])
    tot = sum(ranked_counts.values())
    cov = sum(ranked_counts[p] for p in top if p in ranked_counts)
    return cov / tot if tot else 0.0


def write_strp(path, n_layers, n_expert, want, ranked):
    """ranked = [((layer, expert), count), ...]; n_ranked = len(ranked) <= want."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5I", 1, n_layers, n_expert, want, len(ranked)))
        for pr in ranked:
            l, e = pr[0]
            f.write(struct.pack("<HH", l, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="*", help="route dump file(s)")
    ap.add_argument("-o", "--out", help="output STRP path (write mode)")
    ap.add_argument("--slots", type=int, default=14000)
    ap.add_argument("--check", metavar="PATH", help="report geometry/size of an existing STRP")
    a = ap.parse_args()

    if a.check:
        with open(a.check, "rb") as f:
            magic = f.read(4)
            version, nl, ne, want, n_ranked = struct.unpack("<5I", f.read(20))
        size = os.path.getsize(a.check)
        print(f"{a.check}: magic={magic!r} version={version} {nl}x{ne} built-for={want} ranked={n_ranked} size={size} B")
        return 0

    if not a.dumps:
        ap.error("route dump file(s) required")

    merged = {}
    per_file = {}
    for p in a.dumps:
        c, ml, me, nl = parse_dump(p)
        per_file[p] = (c, ml, me, nl)
        for k, v in c.items():
            merged[k] = merged.get(k, 0) + v
        print(f"  {p}: {nl} dump lines, {len(c)} unique (layer, expert), max layer {ml}, max expert {me}")

    max_layer = max(ml for _, ml, _, _ in per_file.values())
    max_expert = max(me for _, _, me, _ in per_file.values())
    n_layers, n_expert = max_layer + 1, max_expert + 1
    if (n_layers, n_expert) != (48, 512):
        print(f"WARNING: model geometry from dumps is {n_layers}x{n_expert}, expected 48x512")

    ranked = rank(merged, max(a.slots, 20000), n_layers, n_expert)
    total = sum(merged.values())
    print(f"union: {len(merged)} unique pairs, {total} routing decisions across {len(a.dumps)} dump(s)")
    print("in-sample top-C coverage of the union data:")
    for c in (8000, 10000, 12000, 14000, 16000, 20000):
        print(f"  top-{c:>6}: {coverage(merged, ranked, c) * 100:6.2f} %")
    print("per-workload coverage of the UNION ranking (the A/B-relevant numbers):")
    for p, (c, ml, me, nl) in per_file.items():
        for s in (8000, 14000):
            top = set(pair for pair, _ in ranked[:s])
            cov = sum(c.get(k, 0) for k in top) / sum(c.values())
            print(f"  {os.path.basename(p)}: top-{s} of union -> {cov * 100:.2f} %")

    if a.out:
        write_strp(a.out, n_layers, n_expert, a.slots, ranked[:a.slots])
        print(f"wrote {a.out}: {n_layers}x{n_expert} built-for={a.slots} ranked={min(len(ranked), a.slots)} "
              f"({os.path.getsize(a.out)} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
