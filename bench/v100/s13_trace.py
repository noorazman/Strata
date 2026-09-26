#!/usr/bin/env python3
"""Stage 1.3 wait_flag_ge attribution for an nsys cuda_gpu_trace CSV.

Usage: s13_trace.py <trace_cuda_gpu_trace.csv> [gpu-stream]

Classifier (v3, corrected): a wait_flag_ge_kernel event is attributed to the flag
it gates = the NEXT event on the SAME stream:
  - next is copy_i32_from_mapped_kernel  -> flagA (the plan copy; the plan is ready)
  - next is copy_from_mapped_kernel      -> flag_ (the CPU rows are ready)
  - next is a grouped kernel (native_gu / native_down) -> flagB (staging DMA done)

Also reports: H2D copy stats (staging DMA volume), and the GPU-idle gaps on the
verify stream that end at the window-start copy_i32_from_mapped (round boundaries).
"""
import csv
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]),
                         int(r["Strm"] or 0), (r["Name"] or ""),
                         float(r["Bytes (MB)"] or 0)))
    rows.sort()
    n = len(rows)
    kinds = {"A": [0, 0.0], "B": [0, 0.0], "_": [0, 0.0], "U": [0, 0.0]}
    for i, (s, d, st, nm, _) in enumerate(rows):
        if "wait_flag_ge_kernel" not in nm:
            continue
        k = "_"
        if i + 1 < n and rows[i + 1][2] == st:
            nn = rows[i + 1][3]
            if "copy_i32_from_mapped_kernel" in nn:
                k = "A"
            elif "copy_from_mapped_kernel" in nn:
                k = "_"
            elif "native_gu_kernel" in nn or "native_down" in nn:
                k = "B"
            else:
                k = "U"
        kinds[k][0] += 1
        kinds[k][1] += d
    tot = sum(v[1] for v in kinds.values())
    print("== wait_flag_ge attribution (same-stream next-event classifier, v3) ==")
    for k in ("A", "B", "_", "U"):
        c, us = kinds[k]
        pct = (100 * us / tot) if tot else 0.0
        print(f"  flag{k or ' '}: n={c:>6}  {us/1e6:>9.1f} ms  ({pct:5.1f}%)")
    print(f"  TOTAL waits: {tot/1e6:.1f} ms")

    # A-wait size buckets
    a = [d for (s, d, st, nm, _) in rows if "wait_flag_ge_kernel" in nm]
    big = sorted(d for d in a if d > 1_000_000)
    print(f"  waits >1ms: n={len(big)}  total {sum(big)/1e6:.1f} ms  "
          f"max {max(big)/1e3:.1f} ms" if big else "  waits >1ms: none")

    # H2D copies (staging DMA)
    h2d = [(s, d, st, nm, b) for (s, d, st, nm, b) in rows
           if "Host-to-Device" in nm]
    vol = sum(b for *_, b in h2d)
    busy = sum(d for _, d, _, _, _ in h2d)
    print(f"\n== H2D copies: n={len(h2d)}  volume {vol:.1f} MB  copy-engine busy {busy/1e6:.1f} ms")
    sizes = Counter()
    for _, d, _, _, b in h2d:
        sizes[round(b * 10) / 10] += 1
    for sz, c in sorted(sizes.items())[:12]:
        print(f"  ~{sz:5.1f} MB: {c:>6}")

    # round-boundary gaps on the verify stream (default stream 18 if not given)
    gst = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    ev = sorted((s, d) for (s, d, st, _, _) in rows if st == gst)
    span = (ev[-1][0] + ev[-1][1] - ev[0][0]) / 1e6
    gaps = []
    for i in range(1, len(ev)):
        gap = ev[i][0] - (ev[i - 1][0] + ev[i - 1][1])
        if gap >= 500_000:
            gaps.append((gap, ev[i][0]))
    print(f"\n== stream {gst}: span {span:.1f} ms, GPU-idle gaps >=500us: "
          f"n={len(gaps)} total {sum(g for g, _ in gaps)/1e6:.1f} ms "
          f"avg {sum(g for g, _ in gaps)/1e3/max(1, len(gaps)):.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
