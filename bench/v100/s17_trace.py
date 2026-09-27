#!/usr/bin/env python3
"""Stage 1.7 kernel-level analysis of an nsys sqlite capture (cuda-graph-trace=node).

Usage: s17_trace.py <capture.sqlite>

Reports, for the DECODE window (identified as the activity after the largest
inter-kernel gap, i.e. after prefill + host-side round setup):
  - per-kernel aggregation (count, total ms, mean us, share of busy)
  - top kernels with grid/block/registers/shared-memory config
  - wait_flag_ge attribution (same-stream next-event classifier)
  - per-graph (gridId) busy totals -> verify-window vs token graph vs MTP
  - H2D/DtoH memcpy volume
  - GPU busy % of the decode span
  - decode-window span, per-token wall check
"""
import sqlite3
import sys
from collections import defaultdict


def load(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    k = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, s.value AS name, gridId, graphNodeId,"
        " gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread,"
        " staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted,"
        " localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.demangledName"
        " ORDER BY start")]
    m = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, bytes, copyKind"
        " FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")]
    return k, m, db


def find_decode_window(k):
    """Return (start_ns, end_ns) of the decode window: after the largest gap
    between consecutive kernel completions (>= 2 ms), taking the largest gap
    among the first 3 such gaps (skips tiny early gaps)."""
    gaps = []
    prev_end = None
    for e in k:
        if prev_end is not None:
            g = e["start"] - prev_end
            if g >= 2_000_000:
                gaps.append((g, e["start"]))
        prev_end = max(prev_end or 0, e["end"])
    if not gaps:
        return k[0]["start"], k[-1]["end"]
    # the prefill->decode boundary is the largest gap
    gaps.sort(reverse=True)
    start = gaps[0][1]
    return start, k[-1]["end"]


def busy_union(events):
    """events: list of (start, end). Returns union length in ns."""
    ev = sorted(events)
    tot = 0
    cs, ce = ev[0]
    for s, e in ev[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    tot += ce - cs
    return tot


def main():
    k, m, db = load(sys.argv[1])
    d0, d1 = find_decode_window(k)
    span = d1 - d0
    dec = [e for e in k if e["start"] >= d0]
    for e in dec:
        if e["name"] is None:
            e["name"] = "<null>"
    mem = [e for e in m if e["start"] >= d0]
    print(f"decode window: {(span / 1e6):.1f} ms span, {len(dec)} kernels, {len(mem)} memcpy")

    # ---- per-kernel aggregation
    agg = defaultdict(lambda: [0, 0, None])  # name -> [n, total_ns, config]
    for e in dec:
        a = agg[e["name"]]
        a[0] += 1
        a[1] += e["end"] - e["start"]
        if a[2] is None:
            a[2] = (e["gridX"], e["gridY"], e["gridZ"], e["blockX"], e["blockY"],
                    e["blockZ"], e["registersPerThread"],
                    e["staticSharedMemory"] + e["dynamicSharedMemory"],
                    e["sharedMemoryExecuted"], e["localMemoryPerThread"])
    tot = sum(a[1] for a in agg.values())
    print(f"\nkernel busy (union of all streams): {busy_union([(e['start'], e['end']) for e in dec]) / 1e6:.1f} ms "
          f"= {100.0 * busy_union([(e['start'], e['end']) for e in dec]) / span:.1f} % of span")
    print(f"kernel busy (sum, may overlap across streams): {tot / 1e6:.1f} ms")
    print(f"\n== per-kernel (decode window), top 30 by total time ==")
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"{'kernel':58s} {'n':>7s} {'total ms':>9s} {'share':>6s} {'mean us':>8s}  grid/block  regs smem")
    for name, (n, t, cfg) in rows[:30]:
        gx, gy, gz, bx, by, bz, regs, smem, smem_ex, lmem = cfg
        grid = f"{gx}x{gy}x{gz}" if (gy > 1 or gz > 1) else str(gx)
        blk = f"{bx}x{by}x{bz}" if (by > 1 or bz > 1) else str(bx)
        print(f"{name[:58]:58s} {n:>7d} {t / 1e6:>9.1f} {100.0 * t / tot:>5.1f}% {t / n / 1e3:>8.2f}  "
              f"{grid:>10s}/{blk:>5s}  r={regs:<3d} sm={smem_ex or smem:<5d} lm={lmem}")

    # ---- wait_flag_ge attribution
    rows2 = sorted(dec, key=lambda e: e["start"])
    kinds = defaultdict(lambda: [0, 0])
    for i, e in enumerate(rows2):
        if "wait_flag_ge" not in e["name"]:
            continue
        knd = "U"
        if i + 1 < len(rows2) and rows2[i + 1]["streamId"] == e["streamId"]:
            nn = rows2[i + 1]["name"]
            if "copy_i32_from_mapped" in nn:
                knd = "A"
            elif "copy_from_mapped" in nn:
                knd = "C"
            elif "native_gu" in nn or "native_down" in nn or "native_mmvq" in nn:
                knd = "B"
        dt = e["end"] - e["start"]
        kinds[knd][0] += 1
        kinds[knd][1] += dt
    wt = sum(v[1] for v in kinds.values())
    print(f"\n== wait_flag_ge attribution (total {wt / 1e6:.1f} ms) ==")
    for knd in "ACBU":
        n, t = kinds[knd]
        print(f"  flag{knd}: n={n:>6d}  {t / 1e6:>9.1f} ms  ({100.0 * t / wt if wt else 0:5.1f}%)")
    waits = sorted([e["end"] - e["start"] for e in dec if "wait_flag_ge" in e["name"]], reverse=True)
    if waits:
        print(f"  wait>1ms: n={sum(1 for w in waits if w > 1e6)} total {sum(w for w in waits if w > 1e6) / 1e6:.1f} ms max {waits[0] / 1e6:.2f} ms")

    # ---- per-graph (gridId) totals
    g = defaultdict(lambda: [0, 0, set()])
    for e in dec:
        gg = g[e["gridId"] or 0]
        gg[0] += 1
        gg[1] += e["end"] - e["start"]
        gg[2].add(e["name"])
    print(f"\n== per-graph busy (gridId) ==")
    for gid, (n, t, names) in sorted(g.items(), key=lambda kv: -kv[1][1])[:15]:
        top = sorted(names)
        sample = next((x for x in top if "wait_flag" not in x), top[0])
        print(f"  grid {gid}: n={n:>6d}  {t / 1e6:>9.1f} ms   e.g. {sample[:44]}")

    # ---- memcpy
    h2d = [e for e in mem if e["copyKind"] == 1]
    d2h = [e for e in mem if e["copyKind"] == 2]
    p2p = [e for e in mem if e["copyKind"] not in (1, 2)]
    print(f"\n== memcpy (decode) ==")
    print(f"  H2D: n={len(h2d)}  {sum(e['bytes'] for e in h2d) / 1e6:.1f} MB")
    print(f"  D2H: n={len(d2h)}  {sum(e['bytes'] for e in d2h) / 1e6:.1f} MB")
    print(f"  P2P/other: n={len(p2p)}  {sum(e['bytes'] for e in p2p) / 1e6:.1f} MB")
    sizes = defaultdict(int)
    for e in h2d:
        sizes[round(e["bytes"] / 1e6 * 10) / 10] += 1
    for sz, c in sorted(sizes.items())[:10]:
        print(f"    H2D ~{sz:6.1f} MB: {c}")


if __name__ == "__main__":
    main()
