// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <sys/syscall.h>   // SYS_futex (Stage 1.5 hybrid park)
#include <linux/futex.h>   // FUTEX_WAIT / FUTEX_WAKE
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
//   STRATA_POOL_PARK unset           -> the DEFAULT hybrid: spin 2048us then futex (Stage 1.5 winner);
//                                       STRATA_POOL_SPIN_US, if set, overrides the window (upstream #4 knob)
//   STRATA_POOL_PARK "spin"          -> pure spin (legacy; the fallback)
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
        int v;
        if (def) {
            const char* sp = std::getenv("STRATA_POOL_SPIN_US");   // the upstream #4 test knob, as the window
            v = (sp && *sp) ? std::atoi(sp) : kDefaultParkSpinUs;
        } else {
            v = std::atoi(e);
        }
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

}  // namespace

const char* ExpertPool::park_mode() const { return park_cfg().desc; }

// Stage 1.6 placement experiment: STRATA_POOL_CORES overrides the worker pin list.
// Format: comma-separated logical CPUs, ranges allowed ("1-13,29-41" = the 24 SMT pair of node 0's
// cores 1-13). Empty/unset -> the default physical-core list. The count must cover n_workers; a
// clear startup error (not a silent re-pin) on mismatch so an experiment config cannot quietly
// fall back to a different placement than the one being measured.
std::vector<int> pool_cores_override() {
    std::vector<int> out;
    const char* e = std::getenv("STRATA_POOL_CORES");
    if (!e || !*e) return out;
    char* endp = nullptr;
    for (const char* p = e;; p = endp) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        char* q = nullptr;
        const long a = std::strtol(p, &q, 10);
        if (q == p) break;
        long b = a;
        if (*q == '-') { b = std::strtol(q + 1, &endp, 10); } else { endp = q; }
        for (long c = a; c <= b; ++c) out.push_back((int) c);
        if (a < 0 || b > 1023 || a > b) break;
    }
    return out;
}


namespace {
// Stage 1.6 probe: per-site max/avg duration of the pool's post-store sfences (WC drains).
// site 0: worker drain-tail flush, 1: the flag-C sfence in async_participant_done, 2: the host dispatch-end flush.
struct SfenDiag {
    static const int kSites = 3;
    mutable std::atomic<long long> max_ns[kSites] = {};
    mutable std::atomic<long long> total_ns[kSites] = {};
    mutable std::atomic<long> count[kSites] = {};
    bool on = false;
    SfenDiag() { on = std::getenv("STRATA_SFENCE_DIAG") != nullptr; }
    void note(int site, long long ns) const {
        if (!on) return;
        count[site].fetch_add(1, std::memory_order_relaxed);
        total_ns[site].fetch_add(ns, std::memory_order_relaxed);
        long long m = max_ns[site].load(std::memory_order_relaxed);
        while (ns > m && !max_ns[site].compare_exchange_weak(m, ns, std::memory_order_relaxed)) {}
    }
    void report() const {
        if (!on) return;
        for (int i = 0; i < kSites; ++i) {
            const long c = count[i].load(std::memory_order_relaxed);
            if (c == 0) continue;
            const long long mn = max_ns[i].load(std::memory_order_relaxed);
            const long long to = total_ns[i].load(std::memory_order_relaxed);
            std::fprintf(stderr, "pool sfence site %d: n=%ld max=%.3f ms avg=%.4f ms total=%.2f ms\n", i, c,
                         mn / 1e6, (double) to / (double) c / 1e6, to / 1e6);
        }
    }
};
const SfenDiag& sfen_diag_pool() {
    static SfenDiag d;
    return d;
}
long long sfen_t0() { return (long long) std::chrono::steady_clock::now().time_since_epoch().count(); }
// Stage 1.6: the worker-measured wall of the async rows phases (publish to flagC), for the pool report.
struct AsyncDiag {
    long long n = 0;
    double ms_total = 0;
    void note(long long wall_ns) {
        ++n;
        ms_total += wall_ns * 1e-6;
    }
    void report() {
        if (n == 0) return;
        std::fprintf(stderr, "strata pool async rows: %lld phases, %.1f ms total wall, %.2f ms/phase\n",
                     (long long) n, ms_total, ms_total / (double) n);
    }
};
AsyncDiag& async_diag() {
    static AsyncDiag d;
    return d;
}
}   // namespace


namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity) {
    CpuTopology topo;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) topo.worker_cores.push_back((int) i);
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }
    std::vector<char> buf(len);
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        const char* p = buf.data();
        const char* end = p + len;
        struct CoreDesc {
            uint8_t efficiency = 0;
            bool has_smt = false;
            std::vector<int> lps;
        };
        std::vector<CoreDesc> descs;
        while (p < end) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            if (e->Relationship == RelationProcessorCore) {
                CoreDesc cd;
                cd.efficiency = e->Processor.EfficiencyClass;
                cd.has_smt = (e->Processor.Flags & LTP_PC_SMT) != 0;
                const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                for (int bit = 0; bit < 64; ++bit) {
                    if (g.Mask & (1ull << bit)) {
                        cd.lps.push_back((int) (g.Group * 64 + bit));
                    }
                }
                if (!cd.lps.empty()) {
                    descs.push_back(std::move(cd));
                }
            }
            p += e->Size;
        }

        uint8_t min_eff = 255, max_eff = 0;
        for (const auto& c : descs) {
            min_eff = (std::min)(min_eff, c.efficiency);
            max_eff = (std::max)(max_eff, c.efficiency);
        }

        topo.is_hybrid = (max_eff > min_eff);
        if (topo.is_hybrid) {
            for (const auto& c : descs) {
                if (c.efficiency == max_eff) {
                    topo.p_cores++;
                    topo.p_threads += (int) c.lps.size();
                } else {
                    topo.e_cores++;
                }
            }
        } else {
            topo.p_cores = (int) descs.size();
            for (const auto& c : descs) topo.p_threads += (int) c.lps.size();
        }

        if (affinity == PoolAffinity::All || !topo.is_hybrid) {
            for (const auto& c : descs) topo.worker_cores.push_back(c.lps[0]);
            if (skip_first && !topo.worker_cores.empty()) {
                topo.host_core = topo.worker_cores.front();
                topo.worker_cores.erase(topo.worker_cores.begin());
            }
            return topo;
        }

        // Hybrid CPU with Auto or PCores affinity:
        // Prioritize Performance cores:
        // 1. Primary logical processor of each P-core (avoids SMT resource contention)
        // 2. SMT sibling logical processors of P-cores
        // 3. E-cores (only as overflow in Auto mode)
        std::vector<int> p_primaries;
        std::vector<int> p_siblings;
        std::vector<int> e_cores;

        for (const auto& c : descs) {
            if (c.efficiency == max_eff) {
                p_primaries.push_back(c.lps[0]);
                for (size_t s = 1; s < c.lps.size(); ++s) {
                    p_siblings.push_back(c.lps[s]);
                }
            } else {
                for (int lp : c.lps) e_cores.push_back(lp);
            }
        }

        if (skip_first && !p_primaries.empty()) {
            topo.host_core = p_primaries.front();
            p_primaries.erase(p_primaries.begin());
        }

        for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
        for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
        if (affinity != PoolAffinity::PCores) {
            for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
        }
        return topo;
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo_read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    auto cap_read = [](int cpu) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };

    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) allowed.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }

    struct CoreLinux {
        int cpu = -1;
        long pkg = -1;
        long core = -1;
        long cap = -1;
        bool is_sibling = false;
    };
    std::vector<CoreLinux> all_cpus;
    std::vector<std::pair<long, long>> seen_phys;
    long max_cap = 0, min_cap = 1000000;

    for (int cpu : allowed) {
        CoreLinux cl;
        cl.cpu = cpu;
        cl.pkg = topo_read(cpu, "physical_package_id");
        cl.core = topo_read(cpu, "core_id");
        cl.cap = cap_read(cpu);
        if (cl.cap > 0) {
            max_cap = (std::max)(max_cap, cl.cap);
            min_cap = (std::min)(min_cap, cl.cap);
        }
        if (cl.pkg >= 0 && cl.core >= 0) {
            const std::pair<long, long> key{cl.pkg, cl.core};
            if (std::find(seen_phys.begin(), seen_phys.end(), key) != seen_phys.end()) {
                cl.is_sibling = true;
            } else {
                seen_phys.push_back(key);
            }
        }
        all_cpus.push_back(cl);
    }

    topo.is_hybrid = (max_cap > 0 && max_cap > min_cap);
    if (topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (cl.cap == max_cap) {
                if (!cl.is_sibling) topo.p_cores++;
                topo.p_threads++;
            } else {
                if (!cl.is_sibling) topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) seen_phys.size();
        topo.p_threads = (int) all_cpus.size();
    }

    if (affinity == PoolAffinity::All || !topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (!cl.is_sibling) topo.worker_cores.push_back(cl.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }

    // Hybrid CPU on Linux:
    std::vector<int> p_primaries;
    std::vector<int> p_siblings;
    std::vector<int> e_cores;

    for (const auto& cl : all_cpus) {
        if (cl.cap == max_cap) {
            if (!cl.is_sibling) p_primaries.push_back(cl.cpu);
            else p_siblings.push_back(cl.cpu);
        } else {
            e_cores.push_back(cl.cpu);
        }
    }

    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }

    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != PoolAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
#endif
    return topo;
}

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
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

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, PoolAffinity affinity)
    : host_works_(host_works), affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    // STRATA_POOL_SPIN_US (the #4 test knob) is honored through park_cfg() as the hybrid park's window.
    if (n_workers > 0) {
        n_ = n_workers;
    } else if (topo_.is_hybrid && affinity_ != PoolAffinity::All) {
        n_ = (std::max)(1, topo_.p_cores - 1);
    } else {
        n_ = (int) topo_.worker_cores.size();
    }
    if (n_ < 1) n_ = 1;
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    fgate_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    // Stage 1.6: STRATA_POOL_CORES overrides the pin list (placement experiments).
    std::vector<int> ocores;
    if (pin) {
        ocores = pool_cores_override();
        if (!ocores.empty() && (int) ocores.size() < n_) {
            std::fprintf(stderr, "strata: STRATA_POOL_CORES lists %d cores for %d pool workers\n",
                         (int) ocores.size(), n_);
            std::exit(1);
        }
    }
    for (int i = 0; i < n_; ++i) {
        int core = -1;
        if (pin) {
            if (!ocores.empty()) core = ocores[(size_t) i];
            else core = (i < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) i] : -1);
        }
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    async_diag().report();
    sfen_diag_pool().report();
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
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

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
    if (park_futex_enabled()) {
        // A futex wake with no waiters is ~100 ns, so it is unconditional whenever the futex park is in
        // effect (the spin-only "legacy" mode never parks on the word, so it never needs one).
#if !defined(_WIN32)
        ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch_), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
#endif
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
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
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        // Stage 1.5: the park policy (STRATA_POOL_PARK).  The default is the measured V100 winner: spin for
        // 2 ms (keeps the pinned cores at full P-state across the ~256us-1ms inter-dispatch gaps), then
        // futex-park on the epoch word itself (an idle pool costs ~0 CPU; publish() wakes with FUTEX_WAKE).
        // "spin" keeps the legacy pure spin; "futex" parks after one epoch check.
        const int spin_us = park_cfg().spin_us;   // -1: pure spin, 0: always futex, >0: hybrid (microseconds)
        const auto parked_at = std::chrono::steady_clock::now();
        if (spin_us < 0) {
            while (epoch_.load(std::memory_order_acquire) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
        } else {
            while (epoch_.load(std::memory_order_acquire) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
                if (spin_us > 0 &&
                    std::chrono::steady_clock::now() - parked_at < std::chrono::microseconds(spin_us))
                    continue;
                if (spin_us == 0) break;
#if !defined(_WIN32)
                ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch_), FUTEX_WAIT, (unsigned) seen,
                          nullptr, nullptr, 0);
#endif
            }
        }
        if (park_diag()) {
            const uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - parked_at).count();
            park_count_.fetch_add(1, std::memory_order_relaxed);
            gap_total_us_.fetch_add(us, std::memory_order_relaxed);
            int b = 8;
            for (int i = 0; i < 8; ++i)
                if (us < kGapEdges[i]) { b = i; break; }
            gap_hist_[(size_t) b].fetch_add(1, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "strata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        const auto w0 = std::chrono::steady_clock::now();   // Stage 1.4: busy time inside the task body
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
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
    // Stage 1.6: every participant (worker or host) that leaves its share of an async fused phase counts in;
    // the last one raises the layer's flag-C word.
    if (async_armed_.load(std::memory_order_acquire)) async_participant_done();
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    // Stage 1.4: the phase's synchronization is timed separately from the row work (publish-to-first-done and
    // publish-to-last-done are the critical path the fused mode 7 attacks; `busy` is the kernel time inside
    // the task bodies, accumulated by the drain).
    const auto p0 = std::chrono::steady_clock::now();
    wait_parked("before a phase");
    const auto p1 = std::chrono::steady_clock::now();
    busy_ns_.store(0, std::memory_order_relaxed);
    if (mode == 7)
        for (int e = 0; e < fuse_nb_; ++e) fgate_[(size_t) e].v.store(0, std::memory_order_relaxed);
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    const auto t_pub = std::chrono::steady_clock::now();
    if (host_works_) drain(-1, host_scratch_, e);
    uint32_t d = done_.load(std::memory_order_relaxed);
    if (d > 0) {
        proto_first_ += (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t_pub).count();
    } else {
        // wait for the FIRST done with the same stall bound as wait_done, then hand the rest to it
        uint32_t spins = 0;
        std::chrono::steady_clock::time_point t0{};
        for (;;) {
            d = done_.load(std::memory_order_acquire);
            if (d > 0) {
                proto_first_ += (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - t_pub).count();
                break;
            }
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            if (spins == 1024u) t0 = now;
            else if (now - t0 > kStall) {
                std::fprintf(stderr, "strata: the CPU expert pool stalled in a phase: %u of %d jobs done - "
                                     "stopping the engine so the server can start it again\n",
                             d, n_tasks);
                strata::core::release_gpu_waits(stderr);   // #267
                std::fflush(stderr);
                std::abort();
            }
        }
    }
    wait_done(n_tasks);
    const auto t_last = std::chrono::steady_clock::now();
    proto_tail_ += (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t_last - t_pub).count();
    const auto p2 = std::chrono::steady_clock::now();
    wait_parked("after a phase");
    const auto p3 = std::chrono::steady_clock::now();
    ms_phase_wait_ += std::chrono::duration<double, std::milli>(p1 - p0).count();
    ms_phase_wall_ += std::chrono::duration<double, std::milli>(p2 - p1).count();
    ms_phase_repark_ += std::chrono::duration<double, std::milli>(p3 - p2).count();
    ms_phase_busy_ += (double) busy_ns_.load(std::memory_order_relaxed) * 1e-6;   // ns -> ms (per phase)
    proto_n_++;
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
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
    // row-production wall -11%, Flag C -19% (nsys), golden 256/256. STRATA_POOL_UNFUSE=1 restores the
    // two-phase path.
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


bool ExpertPool::dispatch_multi_native_async(const NativeFmt& f, ExpertJobMulti* jobs, int n,
                                             volatile uint32_t* flagC, uint32_t want) {
    if (n <= 0 || n > kMaxSplitMulti) return false;
    // The unfused two-phase path has a host-serial quantization between its phases - it stays synchronous.
    if (std::getenv("STRATA_POOL_UNFUSE") != nullptr) return false;
    if (async_armed_.load(std::memory_order_acquire)) return false;   // a phase is still in flight
    // The workers must be parked (the previous phase fully drained, rows visible) before the shared phase
    // state is reset.  A worker that re-parked has finished its claimed rows, so `parked_ == n_` here is the
    // completion guarantee this dispatch relies on.
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) _mm_pause();
    static const int ftask = std::getenv("STRATA_POOL_TASKS") ? std::atoi(std::getenv("STRATA_POOL_TASKS")) : 3;
    mjobs_ = jobs;
    nfmt_ = &f;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = (ftask > 0 ? ftask : 3) * threads;
    mrows_ = (int64_t) n * FF;
    mrows_dn_ = (int64_t) n * H;
    fuse_nb_ = n;
    mode_ = 7;
    for (int e = 0; e < fuse_nb_; ++e) fgate_[(size_t) e].v.store(0, std::memory_order_relaxed);
    const uint32_t ntasks = 2 * mtasks_ + n;
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, ntasks, 0), std::memory_order_release);
    async_flagC_ = flagC;
    async_want_ = want;
    async_participants_ = (uint32_t) threads;
    async_finished_.store(0, std::memory_order_relaxed);
    async_t0_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()).count(),
                       std::memory_order_relaxed);
    async_armed_.store(true, std::memory_order_release);
    publish();   // seq_cst epoch bump + futex/cv wake
    if (host_works_) {
        drain(-1, host_scratch_, e);   // the host's share, alongside the workers (same path as run_phase)
        async_participant_done();
    }
    {
        const long long t0 = sfen_t0();
        _mm_sfence();   // Stage 1.6: the host zeroed this layer's hit rows into the write-combined h_ymiss_ (and
                        // wrote its own rows share, if any) before the phase fired; flush the host core's WC lines
                        // so they are in DRAM by the time the last participant's flag-C lands.  (The workers flush
                        // their own WC lines at the end of their drain.)
        sfen_diag_pool().note(2, sfen_t0() - t0);
    }
    ++async_phases_;
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    // `mode_` stays 7 until the NEXT dispatch: the workers still read it (and mjobs_/mtasks_/nfmt_) while this
    // phase is in flight, and the next dispatch - sync or async - sets it again after its parked_ == n_ wait.
    // The synchronous run_split_multi_native resets it on return because its phase is complete by then.
    return true;
}

void ExpertPool::async_participant_done() {
    if (!async_armed_.load(std::memory_order_acquire)) return;
    const uint32_t done = async_finished_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (done == async_participants_) {
        const long long t0 = async_t0_ns_.load(std::memory_order_relaxed);
        const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count();
        const long long wall = t0 > 0 ? now - t0 : 0;
        if (t0 > 0)
            async_wall_ns_.fetch_add(wall, std::memory_order_relaxed);
        {
            // Stage 1.6: publish flag C as a NON-POSTED write.  A plain store (even to write-combined memory
            // followed by sfence) is a posted write - it is ISSUED but the GPU's PCIe read of the line keeps
            // returning the stale DRAM value until it is absorbed (measured 6-17 ms at the round head).  A
            // lock-prefixed RMW retires only once the line's write-back has completed, so the value is in DRAM
            // by the time this returns.
            const long long t0 = sfen_t0();
            *async_flagC_ = async_want_;
            __asm__ volatile ("lock addl $0, %0" : "+m" (*async_flagC_) : : "memory");   // non-posted: DRAM on return
            sfen_diag_pool().note(1, sfen_t0() - t0);
        }
        async_armed_.store(false, std::memory_order_release);
        async_diag().note(wall);
    }
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
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

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
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
