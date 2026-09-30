#!/usr/bin/env python3
"""Stage 1.8 — golden / determinism checker for strata bench JSONs.

The Stage 1.7 golden is the 256-token sequence of the fp16/8192 production-default
run, md5 (space-joined tokens + newline) = c1517d02473fbc06b5cf415ea1f8be63.

Usage:
  s18check.py <run1.json> [run2.json ...]   # report each run's md5 + token count
  s18check.py --prefix <run.json> N         # first N tokens vs the golden prefix
  s18check.py --diff <a.json> <b.json>      # first divergence point between two runs
"""
import hashlib
import json
import sys

GOLDEN_MD5 = "c1517d02473fbc06b5cf415ea1f8be63"
GOLDEN_32 = [271, 248068, 198, 760, 1156, 369, 30869, 5402, 430, 328, 23202, 1288,
             264, 11952, 5617, 303, 220, 17, 15, 17, 21, 11, 321, 369, 883, 310,
             3184, 728, 883, 836, 1118, 1834]


def md5_of(toks):
    return hashlib.md5((" ".join(map(str, toks)) + "\n").encode()).hexdigest()


def toks_of(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("output_tokens", [])


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--prefix":
        toks = toks_of(sys.argv[2])
        n = int(sys.argv[3])
        ok = toks[:n] == GOLDEN_32[:n] if n <= 32 else None
        print(f"{sys.argv[2]}: n={len(toks)} first-{min(n,32)} golden-prefix: "
              f"{'MATCH' if ok else ('DIVERGE' if ok is False else 'n>32')}")
        if ok is False:
            for i, (a, b) in enumerate(zip(toks, GOLDEN_32[:n])):
                if a != b:
                    print(f"  first divergence at token {i}: run={a} golden={b}")
                    break
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--diff":
        a, b = toks_of(sys.argv[2]), toks_of(sys.argv[3])
        if a == b:
            print(f"{sys.argv[2]} == {sys.argv[3]} (byte-identical, n={len(a)})")
        else:
            n = min(len(a), len(b))
            for i in range(n):
                if a[i] != b[i]:
                    print(f"first divergence at token {i}: {sys.argv[2]}={a[i]} {sys.argv[3]}={b[i]} "
                          f"(lens {len(a)}/{len(b)})")
                    return
            print(f"prefix identical up to min length {n} (lens {len(a)}/{len(b)})")
        return
    for p in sys.argv[1:]:
        toks = toks_of(p)
        m = md5_of(toks) if toks else "n/a"
        tag = ""
        if toks:
            tag = " GOLDEN" if m == GOLDEN_MD5 else ""
            if len(toks) >= 32:
                tag += " prefix32=" + ("ok" if toks[:32] == GOLDEN_32 else "DIVERGE")
        print(f"{p}: n={len(toks)} md5={m}{tag}")


if __name__ == "__main__":
    main()
