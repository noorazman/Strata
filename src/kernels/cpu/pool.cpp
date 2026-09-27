// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>   // SYS_futex (Stage 1.5 hybrid park)
#include <linux/futex.h>   // FUTEX_WAIT / FUTEX_WAKE
#include <unistd.h>
#endif

namespace strata::kernels::cpu {

// Stage 1.5: the worker park policy.  The park is where a worker waits between batches; for most of the
// program it waits with NOTHING to do, and a pure `_mm_pause` spin (the Stages 1.3/1.4 behavior) then costs
// one whole pinned core PER WORKER at idle - 24 cores at 100 % on a box that is otherwise free.  The hybrid
// park instead spins for a short, configurable window (fast wake for the common small inter-dispatch gaps)
// and then FUTEX-parks on the `epoch_` word itself, so an idle pool costs ~0 CPU.  The publish paths (run,
// run_phase, stop) bump the epoch AND futex_wake the word, which is what wakes the parked workers; a worker
// that is woken by anything else (signal, spurious) just re-reads the epoch and keeps sleeping if it is
// unchanged, so there is no lost-wakeup path.
//
//   STRATA_POOL_PARK unset           -> the DEFAULT hybrid: spin 2048us then futex (Stage 1.5 winner)
//   STRATA_POOL_PARK "spin"          -> pure spin (legacy, byte-for-byte the old behavior; the fallback)
//   STRATA_POOL_PARK "futex" / "0"   -> one epoch check, then futex (always-futex baseline)
//   STRATA_POOL_PARK "<N>"           -> hybrid: spin up to N microseconds, then futex
//
// The default threshold (2 ms) is the measured winner of the Stage 1.5 sweep on the V100 box (24 workers,
// E5-2680 v4): the inter-dispatch park gap is ~256us-4ms (90% in 256us-1ms), and a core that futex-parks
// across a gap drops to the idle P-state (1.2 GHz) so its next drain runs ~2x slower (the always-futex
// arm costs -14% decode for it).  A 2 ms spin window keeps the pinned cores at full P-state through the
// decode gaps (decode: 48.95 tok/s vs 48.89 spin, pool 13.5 vs 13.3 ms/tok) while a 60 s idle costs
// 24.0 -> 0.0 cores; gaps >= 2 ms (3.1% of parks) and the between-request idle still futex-park.
//
// Parsed once per process (the pool is constructed once); the value is process-wide, not per-pool.
namespace {

constexpr int kDefaultParkSpinUs = 2048;  // Stage 1.5 measured default (see the block above)

struct ParkCfg {
    int spin_us;        // -1: pure spin (legacy), 0: always futex, >0: hybrid (microseconds)
    const char* desc;   // startup-log description
};

const ParkCfg& park_cfg() {
    static const ParkCfg cfg = [] {
        const char* e = std::getenv("STRATA_POOL_PARK");
        if (e && std::strcmp(e, "spin") == 0) return ParkCfg{-1, "spin (legacy, opt-in)"};
        if (e && std::strcmp(e, "futex") == 0) return ParkCfg{0, "always-futex"};
        const bool def = !(e && *e);
        int v = def ? kDefaultParkSpinUs : std::atoi(e);
        if (v <= 0) return ParkCfg{0, "always-futex"};
        if (v > 1000000) v = 1000000;
        static char buf[64];
        std::snprintf(buf, sizeof buf, "hybrid spin-%dus+futex%s", v, def ? " (default)" : "");
        return ParkCfg{v, buf};
    }();
    return cfg;
}

bool park_futex_enabled() { return park_cfg().spin_us >= 0; }

// Gap-histogram diagnostics, on only when STRATA_POOL_PARK_DIAG=1 (Stage 1.5 task 1): how long workers
// actually wait in the park.  The bucket edges (us) are chosen around the inter-dispatch gap distribution,
// so a threshold sweep can be read straight off the histogram.
const bool& park_diag() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_POOL_PARK_DIAG");
        return e && e[0] == '1';
    }();
    return on;
}

// Park-gap histogram: 9 buckets over these us edges; bucket 8 is ">= the last edge".
constexpr uint64_t kGapEdges[8] = {1, 4, 16, 64, 256, 1024, 4096, 16384};
constexpr const char* kGapNames[9] = {"<1us", "1-4us", "4-16us", "16-64us", "64-256us",
                                      "0.25-1ms", "1-4ms", "4-16ms", ">=16ms"};

// The ONLY place the epoch moves.  `release` makes the jobs_/njobs_ writes visible before the bump (the
// workers read them after seeing the new epoch); the futex wake then releases any worker parked on the word.
// A wake with no waiters is ~100 ns, so it is unconditional whenever the futex park is in effect.
//
// The glibc `futex()` wrapper is not in this system's libc symbol table, so the raw syscall is used
// (SYS_futex = 202 on x86_64); the FUTEX_WAIT/FUTEX_WAKE constants come from <linux/futex.h>.
inline void publish_epoch(std::atomic<uint32_t>& epoch) {
    epoch.fetch_add(1, std::memory_order_release);
    if (park_futex_enabled()) {
#if !defined(_WIN32)
        ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch), FUTEX_WAKE, INT_MAX,
                  nullptr, nullptr, 0);
#endif
    }
}

// Park on the epoch word until it differs from `seen`.  Fast path: the epoch already moved -> return without
// a syscall.  Slow path: futex_wait (the kernel re-checks the word under its lock, so a publish that lands
// between our check and the wait is not lost).  `stop` is re-checked after each wake; the stop path bumps the
// epoch through publish_epoch, so a parked worker is woken for it.
inline void park_on_epoch(std::atomic<uint32_t>& epoch, const std::atomic<bool>& stop,
                          uint32_t seen) {
    for (;;) {
        const uint32_t e = epoch.load(std::memory_order_acquire);
        if (e != seen || stop.load(std::memory_order_relaxed)) return;
#if !defined(_WIN32)
        ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch), FUTEX_WAIT, (unsigned) e,
                  nullptr, nullptr, 0);
#else
        _mm_pause();
#endif
    }
}

}  // namespace

const char* ExpertPool::park_mode() const { return park_cfg().desc; }

std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
    } else {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            const char* p = buf.data();
            const char* end = p + len;
            while (p < end) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
                if (e->Relationship == RelationProcessorCore) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                    for (int bit = 0; bit < 64; ++bit)
                        if (g.Mask & (1ull << bit)) { cores.push_back((int) (g.Group * 64 + bit)); break; }
                }
                p += e->Size;
            }
        }
    }
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    // The `else` MUST brace the outer `if`: unbraced, it dangles off the inner `if (CPU_ISSET...)` and the
    // fallback list is pushed once per UNSET mask bit (measured: 54,263 "cores" on a 56-CPU machine - the
    // pool then spawns that many 8 MB-stack workers and the main thread spends minutes in the ctor).
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) cores.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
    }
#endif
    if (skip_first && !cores.empty()) cores.erase(cores.begin());
    return cores;
}

namespace {

void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

long long pin_current_thread(int core) {
    if (core < 0) return -1;
#if defined(_WIN32)
    // `SetThreadAffinityMask` RETURNS the previous mask, or 0 on failure - so 0 doubles as the error, which is
    // why the caller must not treat it as a restorable value.
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
    return prev == 0 ? -1 : (long long) prev;
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return -1;
    unsigned long mask = 0;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i)
        if (CPU_ISSET(i, &prev)) mask |= 1ul << i;
    pin_this_thread(core);
    return (long long) mask;
#endif
}

void restore_thread_affinity(long long previous) {
    if (previous <= 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) previous);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; ++i)
        if ((previous >> i) & 1) CPU_SET(i, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works) : host_works_(host_works) {
    const std::vector<int> cores = physical_cores(true);
    n_ = n_workers > 0 ? n_workers : (int) cores.size();
    if (n_ < 1) n_ = 1;
    scratch_.resize((size_t) n_);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    fgate_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    stop_.store(true, std::memory_order_release);
    // Bump the epoch (and wake any futex-parked worker) so a PARKED worker notices the stop flag rather
    // than sleeping through it.
    publish_epoch(epoch_);
    for (auto& t : threads_) t.join();
    if (park_diag()) {
        uint64_t n = park_count_.load(std::memory_order_relaxed);
        uint64_t us = gap_total_us_.load(std::memory_order_relaxed);
        std::fprintf(stderr, "strata pool park-gaps (diagnostics): %llu parks, %.1f ms total parked; ",
                     (unsigned long long) n, us / 1000000.0);
        for (int i = 0; i < 9; ++i)
            std::fprintf(stderr, "%s:%llu%s", kGapNames[i],
                         (unsigned long long) gap_hist_[(size_t) i].load(std::memory_order_relaxed),
                         i < 8 ? " " : "\n");
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    const int spin_us = park_cfg().spin_us;   // -1: pure spin (legacy), 0: always futex, >0: hybrid (us)
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // Stage 1.5: the spin can be BOUNDED (STRATA_POOL_PARK=<N>us) or SKIPPED ("futex"): after the spin
        // window the worker parks on the epoch WORD itself via futex_wait, so an idle pool costs ~0 CPU
        // instead of one core per worker.  The publish paths wake it (publish_epoch); the park protocol
        // (parked_/seen/drain) is unchanged, so nothing downstream of the wait is affected.
        const auto park_t0 = park_diag() ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
        if (spin_us < 0) {
            while (epoch_.load(std::memory_order_acquire) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
        } else {
            if (spin_us > 0) {
                // Spin for up to `spin_us` microseconds.  The clock is checked every 1024 iterations (~2-3 us
                // at park-loop speed) so small thresholds stay honest without paying the clock read per
                // iteration.
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::microseconds(spin_us);
                int64_t it = 0;
                for (;;) {
                    if (epoch_.load(std::memory_order_acquire) != seen) break;
                    if (stop_.load(std::memory_order_relaxed)) return;
                    _mm_pause();
                    if (++it == 1024) {
                        it = 0;
                        if (std::chrono::steady_clock::now() >= deadline) break;
                    }
                }
            }
            park_on_epoch(epoch_, stop_, seen);
        }
        if (park_diag()) {
            const uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - park_t0).count();
            int b = 8;
            for (int i = 0; i < 8; ++i)
                if (us < kGapEdges[i]) { b = i; break; }
            gap_hist_[(size_t) b].fetch_add(1, std::memory_order_relaxed);
            gap_total_us_.fetch_add(us, std::memory_order_relaxed);
            park_count_.fetch_add(1, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = epoch_.load(std::memory_order_relaxed);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.
        drain(id, scratch_[(size_t) id]);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch) {
    (void) id;
    for (;;) {
        const uint32_t i = head_.fetch_add(1, std::memory_order_relaxed);
        if (i >= (uint32_t) njobs_) break;
        const auto w0 = std::chrono::steady_clock::now();   // Stage 1.4: busy time inside the task body
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ == 7 && i < (uint32_t) mtasks_) {
            // Stage 1.4 fused phase: the gate/up row chunks of ONE barrier phase (the down rows follow, gated).
            // Identical row work to mode 5; the per-expert gate accumulates finished rows (release), so the
            // quantization task knows when the whole expert's gate/up output is in memory.
            const int per = FF;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            int e_cur = -1, cnt = 0;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                if (e != e_cur) {
                    if (e_cur >= 0) fgate_[(size_t) e_cur].v.fetch_add((uint32_t) cnt, std::memory_order_release);
                    e_cur = e;
                    cnt = 0;
                }
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (nfmt_->gu_type == 42) {
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r2 = r0; r2 < r1; ++r2)
                            sb.ff[t][r2] = (gbuf[t][r2] / (1.f + std::exp(-gbuf[t][r2]))) * ubuf[t][r2];
                } else {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                }
                cnt += r1 - r0;
                r += r1 - r0;
            }
            if (e_cur >= 0) fgate_[(size_t) e_cur].v.fetch_add((uint32_t) cnt, std::memory_order_release);
        } else if (mode_ == 7 && i < (uint32_t) (mtasks_ + fuse_nb_)) {
            // the fused phase's quantization task for expert e: every gate/up row of e is finished (the gate
            // sums to exactly FF across its chunks), so the intermediate is complete; quantize all of its
            // tokens, then publish FF+1 for the down chunks.
            const int e = (int) (i - (uint32_t) mtasks_);
            while (fgate_[(size_t) e].v.load(std::memory_order_acquire) != FF) _mm_pause();
            SplitBufMulti& sb = split_multi_[(size_t) e];
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (nfmt_->d_type == 42) act_quant_any(sb.ff[t], FF, sb.a2[t]);
                else native_quant_h(*nfmt_, sb.ff[t], sb.hq[t]);
            fgate_[(size_t) e].v.store(FF + 1, std::memory_order_release);
        } else if (mode_ == 7) {
            // the fused phase's down row chunks: work to mode 6, except a chunk first waits (at expert
            // boundaries only - at most two per chunk) for its expert's gate to pass FF.
            const int per = H;
            const uint32_t i2 = (uint32_t) i - (uint32_t) (mtasks_ + fuse_nb_);
            const int64_t g0 = mrows_dn_ * (int64_t) i2 / mtasks_, g1 = mrows_dn_ * (int64_t) (i2 + 1) / mtasks_;
            int e_cur = -1;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                if (e != e_cur) {
                    while (fgate_[(size_t) e].v.load(std::memory_order_acquire) <= FF) _mm_pause();
                    e_cur = e;
                }
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (nfmt_->d_type == 42) {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt_->d_type == 42) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        busy_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - w0).count(), std::memory_order_relaxed);
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    const auto p0 = std::chrono::steady_clock::now();
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) _mm_pause();
    const auto p1 = std::chrono::steady_clock::now();
    busy_ns_.store(0, std::memory_order_relaxed);
    if (mode == 7)
        for (int e = 0; e < fuse_nb_; ++e) fgate_[(size_t) e].v.store(0, std::memory_order_relaxed);
    mode_ = mode;
    njobs_ = n_tasks;
    head_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    publish_epoch(epoch_);   // bump + wake any futex-parked worker (Stage 1.5)
    const auto t_pub = std::chrono::steady_clock::now();
    if (host_works_) drain(-1, host_scratch_);
    const uint32_t d0 = done_.load(std::memory_order_relaxed);
    bool first_seen = (d0 > 0);
    while (done_.load(std::memory_order_acquire) != (uint32_t) n_tasks) {
        if (!first_seen && done_.load(std::memory_order_relaxed) > 0) {
            first_seen = true;
            proto_first_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_pub).count();
        }
        _mm_pause();
    }
    const auto t_last = std::chrono::steady_clock::now();
    if (first_seen) proto_tail_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t_last - t_pub).count();
    const auto p2 = std::chrono::steady_clock::now();
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) _mm_pause();
    const auto p3 = std::chrono::steady_clock::now();
    ms_phase_wait_ += std::chrono::duration<double, std::milli>(p1 - p0).count();
    ms_phase_wall_ += std::chrono::duration<double, std::milli>(p2 - p1).count();
    ms_phase_repark_ += std::chrono::duration<double, std::milli>(p3 - p2).count();
    proto_n_++;
    ms_phase_busy_ += (double) busy_ns_.load(std::memory_order_relaxed) * 1e-6;   // ns -> ms (per phase)
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    // Stage 1.4 (default): the gate/up rows, the per-expert quantization and the down rows of one dispatch run
    // in ONE barrier phase (mode 7) instead of two (modes 5+6 with a host-only quantization between them).
    // Same row work, same kernels, same per-row numerics - one fewer worker wake/re-park cycle per dispatch:
    // row-production wall -11%, Flag C -19% (nsys), golden 256/256. STRATA_POOL_UNFUSE=1 restores the two-phase
    // path; STRATA_POOL_FUSE=1 is a no-op alias kept for the A/B scripts.
    static const bool fuse = std::getenv("STRATA_POOL_UNFUSE") == nullptr;
    // experiment knob: STRATA_POOL_TASKS=N multiplies the per-phase row-chunk count (default 3, as before).
    static const int ftask = std::getenv("STRATA_POOL_TASKS") ? std::atoi(std::getenv("STRATA_POOL_TASKS")) : 3;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
        mtasks_ = (ftask > 0 ? ftask : 3) * threads;
        const auto a = std::chrono::steady_clock::now();
        if (fuse) {
            mrows_ = (int64_t) nb * FF;
            mrows_dn_ = (int64_t) nb * H;
            fuse_nb_ = nb;
            run_phase(7, 2 * mtasks_ + nb);
            const auto d = std::chrono::steady_clock::now();
            ms_multi_gu += std::chrono::duration<double, std::milli>(d - a).count();
            continue;
        }
        fuse_nb_ = 0;
        mrows_ = (int64_t) nb * FF;
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (f.d_type == 42) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * H;
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) _mm_pause();
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    head_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    publish_epoch(epoch_);   // release: jobs_/njobs_ visible before the bump + wake futex-parked workers

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const uint32_t i = head_.fetch_add(1, std::memory_order_relaxed);
            if (i >= (uint32_t) n) break;
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    while (done_.load(std::memory_order_acquire) != (uint32_t) n) _mm_pause();
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) _mm_pause();
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
