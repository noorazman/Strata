// src/kernels/ngram.cpp - P2.S4: the PLE n-gram hash and the IQ4_NL table read.
//
// See include/strata/kernels/ngram.hpp for the semantics, the rival readings, and the note on MADV_RANDOM.
#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
#include <stdexcept>

#if defined(_WIN32)
// `PrefetchVirtualMemory` (memoryapi.h, Windows 8+) is the whole point of the change in `gather` below.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::kernels {

namespace {
/// The A/B arm.  Host-token-path only, so it needs no atomics; see the note on `ple_prefetch_enable`.
bool g_ple_prefetch = true;
}  // namespace

void ple_prefetch_enable(bool on) { g_ple_prefetch = on; }
bool ple_prefetch_enabled() { return g_ple_prefetch; }

PleConsts ple_artifact_consts() {
    // docs/gguf-dump-shard1.txt, verbatim.  Written once here rather than derived, because `head_offsets` is
    // ALSO in the metadata as its own array - and deriving one from the other would hide a mismatch between
    // them instead of surfacing it.  The parity test checks the two agree.
    PleConsts c{};
    c.mult[0] = 23703573157769ull;
    c.mult[1] = 20109073645365ull;
    c.mult[2] = 8052911324071ull;
    const uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t offset[PLE_N_HEADS] = {
        0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    for (int i = 0; i < PLE_N_HEADS; ++i) {
        c.vocab[i] = vocab[i];
        c.offset[i] = offset[i];
    }
    return c;
}

uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    // The first term is an ASSIGNMENT and the rest are XORed into it, which is how the source writes it
    // (`uint64_t mixed = ctx[0]*m[0]; for j=1.. mixed ^= ctx[j]*m[j];`).  Every product wraps mod 2^64,
    // which is what the `(uint64_t)` casts in the source make explicit.
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is OLDEST FIRST, so predecessor `s` positions back is entry (n_prev - s): s=1 reads the
            // NEWEST.  Reading index (s-1) instead walks the window backwards, which still produces indices
            // in range and so cannot be caught by a range check - only by an oracle.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is
            // itself EOS.  Storing first and then cutting would leave position s holding the real token while
            // position s+1 became EOS - one token of history too much.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

namespace {
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

int iq4nl_code(int code) { return kIq4Nl[code & 15]; }

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j+16, not 2j and 2j+1.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

// ---------------------------------------------------------------------------------------------------
struct PleTable::Impl {
    GgufFile* file = nullptr;
    const uint8_t* data = nullptr;
    uint64_t n_rows = 0;
    mutable uint64_t bytes_read = 0;
    // Direct mode (plan v0.3 P2): the mapping above is released after the header parse and every row comes
    // from an unbuffered SSD read into `raw`.
    // Ram mode (Stage 1.2A): `ram_base` holds the WHOLE table in anonymous RAM after a measured preload; the
    // mapping is released like in Direct, so nothing else in the process touches the file during inference.
    // `ram_requests` counts rows served from RAM (the Ram analogue of `requests` in ReaderStats).
    PleIo mode = PleIo::Mmap;
    strata::ngram::PleReader reader;
    strata::ngram::PleReader::Ticket ticket;
    bool pending = false;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES] = {};
    uint8_t* ram_base = nullptr;
    uint64_t ram_bytes = 0;
    uint64_t ram_requests = 0;
    double preload_s = 0;
};

// Stage 1.2B: total system RAM in bytes from /proc/meminfo (MemTotal kB). 0 when unavailable (e.g. non-Linux),
// in which case the insufficient-RAM guard is skipped and the mmap itself remains the last line of defense.
static uint64_t system_total_ram_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            long long kb = 0;
            if (std::sscanf(line.c_str(), "MemTotal: %lld kB", &kb) == 1 && kb > 0)
                return (uint64_t) kb << 10;
        }
    }
    return 0;
}

static std::string gib_str(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f GiB", (double) bytes / (double) (1ULL << 30));
    return buf;
}

PleTable::PleTable() : impl_(new Impl) {}
PleTable::~PleTable() { close(); delete impl_; }

bool PleTable::open(const std::string& gguf_path, std::string& err) {
    return open(gguf_path, err, PleIoOptions{});
}

bool PleTable::open(const std::string& gguf_path, std::string& err, const PleIoOptions& io) {
    close();
    try {
        impl_->file = new GgufFile(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const TensorInfo* t = impl_->file->find("per_layer_token_embd.weight");
    if (t == nullptr) {
        err = "per_layer_token_embd.weight is not in " + gguf_path;
        close();
        return false;
    }
    // [160, 320001536]: ne0 = 160 is the FAST axis, so the ROW index is shape[1] and a row is contiguous.
    if (t->shape.size() != 2 || t->shape[0] != (uint64_t) PLE_HEAD_DIM) {
        err = "per_layer_token_embd.weight has an unexpected shape";
        close();
        return false;
    }
    if (std::strcmp(t->type_name(), "IQ4_NL") != 0) {
        err = std::string("per_layer_token_embd.weight is ") + t->type_name() + ", not IQ4_NL";
        close();
        return false;
    }
    impl_->n_rows = t->shape[1];
    impl_->data = impl_->file->tensor_data(*t);

    // THE CHECK THAT MAKES THE OFFSET FALSIFIABLE.  The manifest's `shard2_tensor.offset` is 0, but that is
    // the offset within the GGUF's DATA SECTION: the file's first 192 bytes are a header, and reading at 0
    // would decode the header plus 192 bytes of shifted rows - still plausible IQ4_NL, and wrong for every
    // row.  `GgufFile` parses the header, so `tensor_data` is already correct; this asserts the tensor
    // exactly fills the file from there, which is what makes the whole arrangement checkable rather than
    // assumed.  A wrong data offset would leave a different remainder.
    // A shard may hold other tensors too (Swift 1.5's shard 1 holds layers 0-12 and the table): the table must
    // then fit inside the file at its own offset; alone in its shard (the original's shard 2) it fills it exactly.
    const uint64_t need = impl_->n_rows * (uint64_t) PLE_ROW_BYTES;
    const uint64_t have = impl_->file->file_size() - impl_->file->data_start();
    const bool alone = impl_->file->tensors().size() == 1;
    if (alone ? need != have : t->offset + need > have) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "PLE table size mismatch: %llu rows x %d B = %llu at offset %llu, but the file holds %llu from "
                      "data_start %llu",
                      (unsigned long long) impl_->n_rows, PLE_ROW_BYTES, (unsigned long long) need,
                      (unsigned long long) t->offset, (unsigned long long) have,
                      (unsigned long long) impl_->file->data_start());
        err = buf;
        close();
        return false;
    }
    if (io.mode == PleIo::Direct) {
        // The parse above is the validated source of the offset; the mapping itself is not kept, so no page of
        // the table can enter this process's working set or the file cache through it.
        const uint64_t table_offset = impl_->file->data_start() + t->offset;
        const uint64_t n_rows = impl_->n_rows;
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        if (!impl_->reader.open(gguf_path, table_offset, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread)) {
            close();
            return false;
        }
        impl_->n_rows = n_rows;
    }
    if (io.mode == PleIo::Ram) {
        // Stage 1.2A: the parse above is the validated source of the offset; the preload copies the table
        // region into anonymous RAM, and the mapping is released afterwards, exactly as in Direct mode.
        // Buffered preads (page-cache-backed): the data is resident in THIS process afterwards, so there is
        // nothing the cache copy can compete with here; the cache pages themselves are reclaimable.
        const uint64_t table_offset = impl_->file->data_start() + t->offset;
        const uint64_t n_rows = impl_->n_rows;
        const uint64_t bytes = n_rows * (uint64_t) PLE_ROW_BYTES;
        // Stage 1.2B insufficient-RAM guard: run BEFORE the preload. If the caller told us how much RAM the
        // rest of the engine needs (io.ram_rest_bytes), require the total to fit on this machine and fail
        // clearly - never silently switch to another mode, which would make the performance behavior
        // surprising on a low-RAM box.
        if (io.ram_rest_bytes > 0) {
            const uint64_t total = system_total_ram_bytes();
            if (total > 0 && total < bytes + io.ram_rest_bytes) {
                err = "PLE ram mode needs " + gib_str(bytes + io.ram_rest_bytes) +
                      " of system RAM (PLE table " + gib_str(bytes) + " + the rest of the engine " +
                      gib_str(io.ram_rest_bytes) + "), but this system has " + gib_str(total) +
                      " total; use --ple-io direct on lower-RAM systems";
                close();
                return false;
            }
        }
        const int fd = ::open(gguf_path.c_str(), O_RDONLY);
        if (fd < 0) { err = "PleTable: cannot open " + gguf_path + " for the RAM preload"; close(); return false; }
        void* base = mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            ::close(fd);
            err = "PleTable: cannot allocate " + gib_str(bytes) +
                  " for the RAM table (use --ple-io direct on lower-RAM systems)";
            close();
            return false;
        }
        std::atomic<bool> bad{false};
        auto fill_slice = [&](uint64_t lo, uint64_t hi) {
            for (uint64_t off = lo; off < hi && !bad; off += ((uint64_t) 1) << 26) {
                const uint64_t n = std::min<uint64_t>(((uint64_t) 1) << 26, hi - off);
                if (pread(fd, (uint8_t*) base + off, (size_t) n, (off_t) (table_offset + off)) != (ssize_t) n)
                    bad = true;
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t nthreads = std::max<uint64_t>(1, std::min<uint64_t>((uint64_t) io.ram_threads, 64));
        if (nthreads == 1) {
            fill_slice(0, bytes);
        } else {
            const uint64_t slice = (bytes + nthreads - 1) / nthreads;
            std::vector<std::thread> pool;
            for (uint64_t i = 1; i < nthreads; ++i)
                pool.emplace_back([&, i] { fill_slice(i * slice, std::min(bytes, (i + 1) * slice)); });
            fill_slice(0, slice);
            for (auto& th : pool) th.join();
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (bad) {
            munmap(base, (size_t) bytes);
            ::close(fd);
            err = "PleTable: the RAM preload read short of the table region";
            close();
            return false;
        }
        ::close(fd);
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        impl_->ram_base = (uint8_t*) base;
        impl_->ram_bytes = bytes;
        impl_->preload_s = secs;
        impl_->n_rows = n_rows;
    }
    impl_->mode = io.mode;
    return true;
}

void PleTable::close() {
    impl_->reader.close();
    impl_->pending = false;
    impl_->mode = PleIo::Mmap;
    delete impl_->file;
    impl_->file = nullptr;
    impl_->data = nullptr;
    impl_->n_rows = 0;
    if (impl_->ram_base != nullptr) {
        munmap(impl_->ram_base, (size_t) impl_->ram_bytes);
        impl_->ram_base = nullptr;
        impl_->ram_bytes = 0;
        impl_->ram_requests = 0;
        impl_->preload_s = 0;
    }
}

bool PleTable::is_open() const {
    // Ram mode holds no mapping and no reader: the resident buffer itself is the open state.
    return impl_->data != nullptr || impl_->reader.is_open() || impl_->ram_base != nullptr;
}
PleIo PleTable::mode() const { return impl_->mode; }
uint64_t PleTable::rows() const { return impl_->n_rows; }
uint64_t PleTable::bytes_read() const { return impl_->bytes_read; }

void PleTable::read_row(uint32_t row, float* out160) const {
    if (impl_->mode == PleIo::Direct && impl_->reader.is_open()) {
        uint8_t raw[PLE_ROW_BYTES];
        std::string err;
        const auto t = impl_->reader.issue(&row, 1, raw);
        if (!impl_->reader.collect(t, err)) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        iq4nl_dequant_row(raw, out160);
        impl_->bytes_read += PLE_ROW_BYTES;
        return;
    }
    if (impl_->mode == PleIo::Ram && impl_->ram_base != nullptr) {
        if (row >= impl_->n_rows) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        iq4nl_dequant_row(impl_->ram_base + (size_t) row * PLE_ROW_BYTES, out160);
        impl_->bytes_read += PLE_ROW_BYTES;
        ++impl_->ram_requests;
        return;
    }
    if (impl_->data == nullptr || row >= impl_->n_rows) {
        std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        return;
    }
    iq4nl_dequant_row(impl_->data + (size_t) row * PLE_ROW_BYTES, out160);
    impl_->bytes_read += PLE_ROW_BYTES;
}

bool PleTable::issue(const uint32_t* rows16) {
    std::memcpy(impl_->rows, rows16, sizeof impl_->rows);
    if (impl_->mode == PleIo::Direct) {
        if (impl_->pending) return false;              // one token in flight per table
        impl_->ticket = impl_->reader.issue(impl_->rows, PLE_N_HEADS, impl_->raw);
        impl_->pending = true;
        return true;
    }
    if (impl_->mode == PleIo::Ram) {
        // Synchronous: the rows come out of the resident table, so `issue` is done by the time it returns and
        // `collect` only dequantizes. Out-of-range rows are zeros, the Direct and Mmap behaviour.
        if (impl_->pending) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            const uint32_t r = impl_->rows[h];
            if (r >= impl_->n_rows) std::memset(impl_->raw + (size_t) h * PLE_ROW_BYTES, 0, PLE_ROW_BYTES);
            else std::memcpy(impl_->raw + (size_t) h * PLE_ROW_BYTES,
                             impl_->ram_base + (size_t) r * PLE_ROW_BYTES, PLE_ROW_BYTES);
            ++impl_->ram_requests;
        }
        impl_->pending = true;
        return true;
    }
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * PLE_ROW_BYTES);
            ranges[n].NumberOfBytes = PLE_ROW_BYTES;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    impl_->pending = true;
    return true;
}

bool PleTable::collect(float* out2560, std::string& err) {
    if (!impl_->pending) { err = "PleTable::collect without issue"; return false; }
    impl_->pending = false;
    if (impl_->mode == PleIo::Direct) {
        if (!impl_->reader.collect(impl_->ticket, err)) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h)
            iq4nl_dequant_row(impl_->raw + (size_t) h * PLE_ROW_BYTES, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * PLE_ROW_BYTES;
        return true;
    }
    if (impl_->mode == PleIo::Ram) {
        // `issue` already copied the rows out of the resident table; only the dequantize remains.
        for (int h = 0; h < PLE_N_HEADS; ++h)
            iq4nl_dequant_row(impl_->raw + (size_t) h * PLE_ROW_BYTES, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * PLE_ROW_BYTES;
        return true;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
    return true;
}

bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err) {
    if (impl_->pending) { err = "PleTable::gather_batch while a token is in flight"; return false; }
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    if (impl_->mode == PleIo::Direct) {
        std::vector<uint8_t> raw(n * PLE_ROW_BYTES);
        const auto ticket = impl_->reader.issue(rows, n, raw.data());
        if (!impl_->reader.collect(ticket, err)) return false;
        for (size_t i = 0; i < n; ++i) iq4nl_dequant_row(raw.data() + i * PLE_ROW_BYTES, out + i * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) n * PLE_ROW_BYTES;
        return true;
    }
    if (impl_->mode == PleIo::Ram) {
        // The whole chunk's rows out of the resident table: one memcpy per row, then the same dequantize the
        // Direct path runs. This is the prefill/verify replacement for the batched O_DIRECT ticket.
        std::vector<uint8_t> raw(n * PLE_ROW_BYTES);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t r = rows[i];
            if (r >= impl_->n_rows) std::memset(raw.data() + i * PLE_ROW_BYTES, 0, PLE_ROW_BYTES);
            else std::memcpy(raw.data() + i * PLE_ROW_BYTES, impl_->ram_base + (size_t) r * PLE_ROW_BYTES,
                             PLE_ROW_BYTES);
            ++impl_->ram_requests;
        }
        for (size_t i = 0; i < n; ++i) iq4nl_dequant_row(raw.data() + i * PLE_ROW_BYTES, out + i * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) n * PLE_ROW_BYTES;
        return true;
    }
    for (size_t i = 0; i < n; ++i) read_row(rows[i], out + i * PLE_HEAD_DIM);
    return true;
}

void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

std::string PleTable::io_report() const {
    if (impl_->mode == PleIo::Ram) {
        char buf[256];
        const double gib = (double) impl_->ram_bytes / (double) (1ull << 30);
        std::snprintf(buf, sizeof buf,
                      "ple ram: %llu rows resident (%.2f GiB), preloaded in %.2f s (%.2f GiB/s); served %llu rows (%.1f MB) from RAM",
                      (unsigned long long) impl_->n_rows, gib, impl_->preload_s,
                      impl_->preload_s > 0 ? gib / impl_->preload_s : 0.0, (unsigned long long) impl_->ram_requests,
                      (double) impl_->ram_requests * PLE_ROW_BYTES / 1e6);
        return buf;
    }
    if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats& s = impl_->reader.stats();
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                  "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
                  (unsigned long long) s.requests, s.requests ? 100.0 * (double) s.cache_hits / (double) s.requests : 0.0,
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
    return buf;
}

std::string PleTable::preload_report() const {
    if (impl_->mode != PleIo::Ram || impl_->ram_base == nullptr) return {};
    char buf[200];
    const double gib = (double) impl_->ram_bytes / (double) (1ull << 30);
    std::snprintf(buf, sizeof buf,
                  "PLE ram: preloaded %llu rows (%.2f GiB) in %.2f s (%.2f GiB/s); RAM-resident for the whole run",
                  (unsigned long long) impl_->n_rows, gib, impl_->preload_s,
                  impl_->preload_s > 0 ? gib / impl_->preload_s : 0.0);
    return buf;
}

void PleTable::gather(const uint32_t* rows16, float* out2560) const {
    if (impl_->mode == PleIo::Direct || impl_->mode == PleIo::Ram) {
        // `gather` stays const for its existing callers; the reader's (or the RAM table's) state is the table's
        // I/O state. Ram's `issue`/`collect` are synchronous copies, so the split costs nothing here.
        PleTable* self = const_cast<PleTable*>(this);
        std::string err;
        if (!self->issue(rows16) || !self->collect(out2560, err)) {
            std::fprintf(stderr, "PleTable::gather: %s\n", err.empty() ? "a token is already in flight" : err.c_str());
            std::memset(out2560, 0, (size_t) NG_N_EMBD * sizeof(float));
        }
        return;
    }
    // ================================ SIXTEEN SERIAL PAGE FAULTS, MEASURED ================================
    //
    // **THIS COST 2.10-2.61 ms PER TOKEN AND HAD NEVER BEEN IN THE PLAN'S BUDGET AT ALL.**  The round-309
    // `token host phases` line put it second behind the layer loop among avoidable terms, and the arithmetic
    // says why: the table is 320,001,536 rows of `PLE_ROW_BYTES` = 90 B in a 26.8 GB mapping, so the sixteen
    // rows a token needs are 1,440 B - **0.5 MB/s**.  That is not bandwidth, it is latency: sixteen reads into
    // sixteen different 4 KB pages scattered across 26.8 GB, taken ONE AT A TIME, and on this machine the PLE
    // shard is 26.8 GB against 63 GB of RAM that the 31.6 GB expert arena is also competing for, so they are
    // not in the OS cache.  Sixteen serial NVMe reads at ~150 us is 2.4 ms, which is the measurement.
    //
    // `PrefetchVirtualMemory` issues all sixteen in ONE call and lets them complete in parallel.  It is a hint
    // and cannot change the answer - a range it does not fetch is simply faulted in by the read that follows -
    // so the only risk is that it does nothing.
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            // Out-of-range rows are handled by `read_row` as zeros and have no address to prefetch.
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * PLE_ROW_BYTES);
            ranges[n].NumberOfBytes = PLE_ROW_BYTES;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    // HEAD-SLOWEST, which is what `ggml_get_rows` does and what the source's own comment says: head h's 160
    // values occupy [h*160, (h+1)*160).  A head-fastest layout would put element (d, h) at d*16 + h and needs
    // a real transpose - a reshape of the same flat buffer compares equal and would make the check vacuous.
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace strata::kernels
