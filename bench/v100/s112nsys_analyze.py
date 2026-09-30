#!/usr/bin/env python3
"""bench/v100/s112nsys_analyze.py - Stage 1.12 E5: kernel-level A/B from nsys sqlite captures.

Compares the 16 K prefill windows of the OFF (s110nsys-16k-off.sqlite, committed in Stage 1.11)
and E5 (s112nsys-16k-e5.sqlite) captures.  The prefill window is anchored on the main-expert H2D
burst (stream 15), which spans the prefill in both captures.  nsys sqlite timestamps are ns.
  - kernel launch counts and time, grouped by category (dequant / GEMM / fused / others)
  - H2D byte volume by stream (E5 must not change DMA: staging ring untouched)

Usage: python3 s112nsys_analyze.py <off.sqlite> <e5.sqlite>
"""
import sqlite3, sys

def load(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db

def window(db):
    # The prefill window = the main-expert H2D span on stream 15 (16 K prefill).
    r = db.execute("""SELECT min(start) s, max(end) e FROM CUPTI_ACTIVITY_KIND_MEMCPY
                      WHERE streamId = 15 AND copyKind = 1""").fetchone()
    return r["s"], r["e"]

def kernels(db, s, e):
    # kernel name columns are integer FKs into StringIds in this nsys export format.
    rows = db.execute("""SELECT COALESCE(s.value, '<unnamed>') name, count(*) n, sum(k.end-k.start) ns
                         FROM CUPTI_ACTIVITY_KIND_KERNEL k
                         LEFT JOIN StringIds s ON k.demangledName = s.id
                         WHERE k.start >= ? AND k.end <= ?
                         GROUP BY name ORDER BY ns DESC""", (s, e)).fetchall()
    return rows

def h2d(db):
    return db.execute("""SELECT streamId, count(*) n, sum(bytes) b
                         FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1
                         GROUP BY streamId ORDER BY b DESC""").fetchall()

def classify(name):
    n = (name or "<unnamed>").lower()
    if "moe_fused" in n or "fused_dq_gemm" in n:
        return "E5 fused dequant+gemm"
    if "dequant" in n:
        return "dequant (OFF path)"
    if any(k in n for k in ("gemm", "gemv", "sgemmx", "hgemm", "sm35", "sm75", "cutlass",
                            "magma", "cublas", "ampere", "volta", "hmma")):
        return "GEMM (cuBLAS/expert+dense)"
    if "swiglu" in n:
        return "swiglu"
    if "moe_combine" in n or "moe_hit" in n:
        return "moe combine/hit"
    return "other"

def main():
    off, e5 = load(sys.argv[1]), load(sys.argv[2])
    for tag, db in (("OFF", off), ("E5", e5)):
        s, e = window(db)
        print(f"\n===== {tag}: prefill window {(e-s)/1e9:.2f} s (H2D span on stream 15) =====")
        rows = kernels(db, s, e)
        tot_n = sum(r["n"] for r in rows)
        tot_ns = sum(r["ns"] for r in rows)
        print(f"total: {tot_n} kernel launches, {tot_ns/1e6:.1f} ms kernel time")
        agg = {}
        for r in rows:
            agg.setdefault(classify(r["name"]), [0, 0])
            agg[classify(r["name"])][0] += r["n"]
            agg[classify(r["name"])][1] += r["ns"]
        for k, (n, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f"  {k:32s} {n:8d} launches {ns/1e6:10.1f} ms")
        print("  top individual kernels:")
        for r in rows[:8]:
            print(f"    {r['ns']/1e6:10.1f} ms {r['n']:6d}x {(r['name'] or '<unnamed>')[:96]}")
        print("  H2D by stream:")
        for r in h2d(db):
            print(f"    stream {r['streamId']:2d}: {r['n']:8d} transfers {r['b']/1e9:8.2f} GB")

if __name__ == "__main__":
    main()
