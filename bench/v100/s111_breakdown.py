#!/usr/bin/env python3
"""Stage 1.11 — prefill cost breakdown into the 10 task categories (v2).

Usage: s111_breakdown.py <capture.sqlite> [prompt_tokens]

copyKind: 1=H2D, 2=D2H, 8=D2D.  MemKind: 0=Pageable 1=Pinned 2=Device.
Expert pipeline attribution on the main stream:
  [dequant_gu][GEMM][swiglu_il][dequant_flat(expert)][GEMM] per expert use.
dequant_gu_kernel count == total expert uses (streamed + resident).
"""
import bisect
import sqlite3
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    con = sqlite3.connect(path)
    cur = con.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))

    krows = [list(r) for r in cur.execute(
        "SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()]
    for r in krows:
        r[3] = strings.get(r[3], str(r[3]))
    k = [{"s": r[0], "e": r[1], "st": r[2], "n": r[3]} for r in krows]
    k.sort(key=lambda x: x["s"])

    dcnt = defaultdict(int)
    for e in k:
        if e["n"].startswith("doorbell_publish"):
            dcnt[e["st"]] += 1
    vstream = max(dcnt, key=dcnt.get)
    r0 = min(e["s"] for e in k if e["n"].startswith("doorbell_publish") and e["st"] == vstream)
    st_busy = defaultdict(int)
    for e in k:
        if e["s"] < r0:
            st_busy[e["st"]] += e["e"] - e["s"]
    main_stream = max(st_busy, key=st_busy.get)
    seq = [e for e in k if e["st"] == main_stream and e["s"] < r0]
    t0, t1 = seq[0]["s"], r0
    span = t1 - t0

    # ---- memcpys in window
    mrows = [list(r) for r in cur.execute(
        "SELECT start, end, streamId, bytes, copyKind, srcKind, dstKind FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()]
    win_m = [r for r in mrows if t0 <= r[0] < t1]
    h2d = [r for r in win_m if r[4] == 1]
    d2h = [r for r in win_m if r[4] == 2]
    d2d = [r for r in win_m if r[4] == 8]
    blob = [r for r in h2d if r[3] >= 512 * 1024]
    h2d_bytes, h2d_busy = sum(r[3] for r in h2d), sum(r[1] - r[0] for r in h2d)
    blob_bytes, blob_busy = sum(r[3] for r in blob), sum(r[1] - r[0] for r in blob)
    d2h_bytes, d2h_busy = sum(r[3] for r in d2h), sum(r[1] - r[0] for r in d2h)
    d2d_bytes, d2d_busy = sum(r[3] for r in d2d), sum(r[1] - r[0] for r in d2d)
    # overlap of blob H2D with main-stream kernel activity
    mbusy = [(e["s"], e["e"]) for e in seq]
    tot_overlap = 0
    for r in blob:
        a, b = r[0], r[1]
        j = bisect.bisect_left(mbusy, (b, 1 << 62)) - 1
        while j >= 0:
            iv = mbusy[j]
            if iv[1] <= a:
                break
            tot_overlap += min(b, iv[1]) - max(a, iv[0])
            j -= 1
    # per-stream memcpy busy
    st_m = defaultdict(int)
    for r in win_m:
        st_m[r[2]] += r[1] - r[0]

    # ---- main-stream timeline incl. memcpys (for gaps + MoE sections)
    tl = []
    for e in seq:
        tl.append([e["s"], e["e"], "K", e["n"]])
    for r in win_m:
        if r[2] == main_stream:
            tl.append([r[0], r[1], "M", f"cpy{r[4]}:{r[3]}"])
    tl.sort(key=lambda x: x[0])

    # gaps
    gaps = []
    for a, b in zip(tl, tl[1:]):
        if b[0] > a[1]:
            gaps.append((a[1], b[0], b[0] - a[1], a[3], b[3]))
    gap_tot = sum(g[2] for g in gaps)
    gap_small = sum(g[2] for g in gaps if g[2] < 50_000)
    gap_big = [g for g in gaps if g[2] >= 50_000]
    gap_after_d2h = sum(g[2] for g in gaps if g[3].startswith("cpy2"))
    main_busy = sum(e["e"] - e["s"] for e in seq)

    # MoE sections: route_kernel ... moe_combine_kernel on the main stream
    moe_secs = []
    open_s = None
    for e in seq:
        if "route_kernel" in e["n"] and open_s is None:
            open_s = e["s"]
        elif "moe_combine" in e["n"] and open_s is not None:
            moe_secs.append((open_s, e["e"]))
            open_s = None
    moe_wall = sum(b - a for a, b in moe_secs)
    # per-expert wait: gap immediately before each dequant_gu launch
    dqs = [e for e in seq if "dequant_gu" in e["n"]]
    waits = []
    prev_end = {}
    for i, e in enumerate(seq):
        prev_end[i] = seq[i - 1]["e"] if i > 0 else e["s"]
    for i, e in enumerate(seq):
        if "dequant_gu" in e["n"]:
            waits.append(e["s"] - prev_end[i])
    w_big = [w for w in waits if w > 20_000]

    # ---- classification
    def is_gemm(n):
        return any(x in n for x in ("volta", "magma", "splitKreduce", "gemm", "gemv", "cutlass", "Kernel2"))
    CAT = {"expert_dq": [], "expert_dq_down": [], "expert_gemm": [], "route": [], "dense_dq": [], "other": []}
    pending = 0
    for e in seq:
        n = e["n"]
        if "dequant_gu" in n:
            CAT["expert_dq"].append(e)
            pending = 2
        elif "dequant_flat" in n or "dequant_kernel" in n:
            if pending > 0:
                CAT["expert_dq_down"].append(e)
            else:
                CAT["dense_dq"].append(e)
        elif is_gemm(n):
            if pending > 0:
                CAT["expert_gemm"].append(e)
                pending -= 1
            else:
                CAT["other"].append(e)
        elif "route" in n and "topk" not in n:
            CAT["route"].append(e)
        elif "swiglu_il" in n or "swiglu_pair" in n or "gather_rows16" in n or "moe_combine" in n:
            CAT["other"].append(e)
        else:
            CAT["other"].append(e)

    def tot(lst):
        return sum(e["e"] - e["s"] for e in lst)

    # ---- host runtime API in window
    rts = cur.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    names = defaultdict(lambda: [0, 0])
    sync_wait = ev_sync = launch_api = memcpy_api = 0
    host_api_busy = 0
    for r in rts:
        if not (t0 <= r[0] < t1):
            continue
        nm = strings.get(r[2], str(r[2]))
        d = r[1] - r[0]
        names[nm][0] += 1
        names[nm][1] += d
        host_api_busy += d
        if "StreamSynchronize" in nm or "DeviceSynchronize" in nm:
            sync_wait += d
        elif "EventSynchronize" in nm:
            ev_sync += d
        elif "LaunchKernel" in nm or "GraphLaunch" in nm:
            launch_api += d
        elif "Memcpy" in nm:
            memcpy_api += d
    srows = cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start >= ? AND start < ?",
                         (t0, t1)).fetchall()
    gpu_sync = sum(r[1] - r[0] for r in srows)

    # ---- per-kernel table
    by_name = defaultdict(lambda: [0, 0])
    for e in seq:
        by_name[e["n"]][0] += 1
        by_name[e["n"]][1] += e["e"] - e["s"]
    for r in win_m:
        tag = f"memcpy-c{r[4]}:{r[3]}"
        by_name[tag][0] += 1
        by_name[tag][1] += r[1] - r[0]

    # ---- HBM estimate for the top kernels (bytes = known shapes; V100 HBM2 = 900 GB/s)
    HBM = 900.0  # GB/s
    hbm_notes = [
        # (match, bytes per call estimate, note)
        ("dequant_gu", 819200 + 1280 * 2560 * 2, "IQ3 gu codes in, fp16 1280x2560 out"),
        ("dequant_flat", 409600 + 2560 * 640 * 2, "down codes in, fp16 2560x640 out"),
        ("attn_chunk", 8 * 1024 * 1024, "chunked sparse attn (context dependent)"),
        ("gdn_rec", 36 * 4 * 10240 + 2048 * 4 * 10240, "state 36x128x128x2 + h/y io per token"),
        ("route", 2048 * 512 * 4, "logits read"),
        ("moe_combine", 2048 * 2560 * 4 * 11, "10 rows + shared + out"),
        ("swiglu_il", 20480 * 1280 * 2 // 2, "gu rows io (avg ne~40)"),
    ]

    # ---------------------------------------------------------------- report
    ms = span / 1e6
    print(f"== s111 prefill breakdown v2 ({path.split('/')[-1]}) ==")
    print(f"prefill window {ms:.1f} ms | {ntok} tokens = {span/ntok/1e3:.3f} ms/token | "
          f"main stream {main_stream} busy {main_busy/1e6:.1f} ms ({100*main_busy/span:.1f}%)")
    print()
    print(f"{'category':60s}{'ms':>9s}{'%span':>8s}{'ms/tok':>9s}")

    def line(tag, msval, note=""):
        print(f"{tag[:60]:60s}{msval:9.1f}{100*msval/ms:8.1f}{msval/ntok:9.4f}  {note}")

    print("-- GPU kernel work (main stream) --")
    line("2. expert dequant gu (dequant_gu)", tot(CAT["expert_dq"]) / 1e6, f"n={len(CAT['expert_dq'])}")
    line("2. expert dequant down (dequant_flat@expert)", tot(CAT["expert_dq_down"]) / 1e6, f"n={len(CAT['expert_dq_down'])}")
    line("3. expert GEMM (cuBLAS f16, per-expert)", tot(CAT["expert_gemm"]) / 1e6, f"n={len(CAT['expert_gemm'])}")
    line("4. routing", tot(CAT["route"]) / 1e6, f"n={len(CAT['route'])}")
    line("8. dense/native dequant (dequant_flat@dense, dequant_kernel)", tot(CAT["dense_dq"]) / 1e6, f"n={len(CAT['dense_dq'])}")
    line("8. other kernels (atttn/GDN/dense GEMM/PLE/norms/MoE glue)", tot(CAT["other"]) / 1e6, f"n={len(CAT['other'])}")
    line("   [main-stream kernel total]", main_busy / 1e6)
    line("   [MoE section wall (route..moe_combine)]", moe_wall / 1e6, f"{len(moe_secs)} sections")
    swiglu_busy = sum(e["e"] - e["s"] for e in seq if "swiglu" in e["n"])
    moe_compute = tot(CAT["expert_dq"]) + tot(CAT["expert_dq_down"]) + tot(CAT["expert_gemm"]) + swiglu_busy
    print(f"   [MoE compute (dequant+GEMM+swiglu, main stream)] {moe_compute/1e6:.1f} ms  "
          f"[DMA blob H2D busy {blob_busy/1e6:.1f} ms] -> floor = max(DMA, compute) + glue; "
          f"section wall {moe_wall/1e6:.1f} ms => serialization ~{(moe_wall - max(blob_busy, moe_compute))/1e6:.0f} ms")
    print()
    print("-- memory traffic (all streams) --")
    line("1. expert blob H2D (staging ring)", blob_busy / 1e6,
         f"n={len(blob)} {blob_bytes/1e9:.1f} GB @ {blob_bytes/blob_busy/1e9 if blob_busy else 0:.2f} GB/s")
    line("   all H2D", h2d_busy / 1e6, f"{h2d_bytes/1e9:.2f} GB; overlapped under main-stream kernels: {tot_overlap/1e6:.1f} ms")
    line("   D2H (ids round-trips)", d2h_busy / 1e6, f"n={len(d2h)} {d2h_bytes/1e6:.0f} KB")
    line("   D2D", d2d_busy / 1e6, f"n={len(d2d)} {d2d_bytes/1e6:.0f} MB")
    print()
    print("-- host / gaps / syncs --")
    line("5. gaps after D2H memcpys (host grouping)", gap_after_d2h / 1e6, f"n={len([g for g in gaps if g[3].startswith('cpy2')])}")
    line("6. main-stream gaps < 50 us (launch class)", gap_small / 1e6, f"n={len([g for g in gaps if g[2] < 50_000])}")
    line("   gaps >= 50 us", sum(g[2] for g in gap_big) / 1e6, f"n={len(gap_big)}")
    line("7. host time in Stream/Device/Event Synchronize", (sync_wait + ev_sync) / 1e6)
    line("   GPU-side sync events", gpu_sync / 1e6)
    print(f"   host CUDA API time in window: {host_api_busy/1e6:.1f} ms "
          f"(launch {launch_api/1e6:.1f}, memcpy {memcpy_api/1e6:.1f})")
    host_non_api = ms - host_api_busy / 1e6
    print(f"   host non-API CPU time (grouping/PLE/etc.): ~{host_non_api:.1f} ms")
    print()
    print("-- per-expert H2D wait (gap before each dequant_gu) --")
    if waits:
        ws = sorted(waits)
        import statistics
        print(f"  n={len(waits)} mean {statistics.mean(waits)/1e3:.1f} us  p50 {ws[len(ws)//2]/1e3:.1f}  "
              f"p99 {ws[int(len(ws)*0.99)]/1e3:.1f}  max {ws[-1]/1e3:.1f}  "
              f">20us: n={len(w_big)} total {sum(w_big)/1e6:.1f} ms")
    print()
    print("-- streams (memcpy busy) --")
    for st, b in sorted(st_m.items(), key=lambda x: -x[1]):
        print(f"  stream {st}: memcpy busy {b/1e6:.1f} ms")
    print()
    print("-- top kernels by GPU time --")
    for nm, (c, t) in sorted(by_name.items(), key=lambda x: -x[1][1])[:30]:
        print(f"  {nm[:72]:72s} {c:8d}  {t/1e6:9.1f} ms  {100*t/span:5.1f} %  mean {t/max(c,1)/1e3:8.1f} us")
    print()
    print("-- gap histogram (main stream, incl. memcpys) --")
    bins = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 1000), (1000, 10**12)]
    for lo, hi in bins:
        sel = [g for g in gaps if lo <= g[2] / 1e3 < hi]
        lab = f"{lo}-{hi}" if hi < 10**12 else f"{lo}+"
        print(f"  {lab:>10s} us: n={len(sel):7d} total {sum(g[2] for g in sel)/1e6:8.1f} ms")
    worst = sorted(gaps, key=lambda g: -g[2])[:10]
    for a, b, d, na, nb in worst:
        print(f"    worst {d/1e3:9.1f} us  after [{na[:44]}] before [{nb[:44]}]")
    print()
    print("-- HBM bandwidth estimate for top kernels (bytes from known shapes) --")
    for pat, bpc, note in hbm_notes:
        sel = [e for e in seq if pat in e["n"]]
        if not sel:
            continue
        t = sum(e["e"] - e["s"] for e in sel)
        bw = t and (len(sel) * bpc) / t  # bytes / ns  ==  GB/s
        print(f"  {pat:20s} {len(sel):7d}  {t/1e6:8.1f} ms  ~{bw:6.1f} GB/s ({100*bw/HBM:4.1f}% of 900)  {note}")
    print()
    print("-- host sync calls in window --")
    for nm, (c, t) in sorted(names.items(), key=lambda x: -x[1][1])[:12]:
        print(f"  {nm[:44]:44s} n={c:8d} total {t/1e6:9.1f} ms mean {t/max(c,1)/1e3:8.1f} us")


if __name__ == "__main__":
    main()
