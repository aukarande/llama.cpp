#define CPU_WARMUP_ITERS  2
// timed passes: at least CPU_TIMED_ITERS, more until the timed work spans CPU_TIMED_MIN_S
// or CPU_TIMED_ITERS_MAX passes, so a sub-millisecond op is averaged over dozens of samples
#define CPU_TIMED_ITERS      2
#define CPU_TIMED_ITERS_MAX  32
#define CPU_TIMED_MIN_S      0.010
// cold cache by rotation: consecutive passes read different copies of the operand (or a
// different expert set) until at least BENCH_COLD_BYTES of traffic (operands read, outputs
// written) lie between two reads of the same bytes, more than any CPU cache holds; no pass
// sees what the previous one left in cache
#define BENCH_COLD_BYTES  (512ULL * 1024 * 1024)
#define BENCH_COPY_MAX    4096

#include "profiler-common.h"

#include "common.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#include <intrin.h>
#endif

// pin-ceiling probe: physical-memory query
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <cstdlib>
#endif

// copies a rotation needs so that BENCH_COLD_BYTES of traffic separate two reads of one
// copy; bytes_per_pass = what one pass reads of its copy plus the output it writes
static int bench_n_copies(size_t bytes_per_pass) {
    if (bytes_per_pass == 0) {
        return 1;
    }
    const size_t n = (BENCH_COLD_BYTES + bytes_per_pass - 1) / bytes_per_pass;
    return (int) std::min<size_t>(BENCH_COPY_MAX, std::max<size_t>(1, n));
}

// prepare(pass) points a pass at operands no recent pass has read (a weight copy, a KV
// block, an expert set) and returns the graph to run; the first CPU_WARMUP_ITERS passes are
// not timed, a small op then repeats until its samples span the minimum duration
static double time_op(ggml_backend_t be, const std::function<ggml_cgraph *(int)> & prepare) {
    int pass = 0;
    for (; pass < CPU_WARMUP_ITERS; pass++) {
        ggml_backend_graph_compute_async(be, prepare(pass));
    }
    double total_time = 0.0;
    std::vector<double> samples;
    bench_timer t;
    while (samples.size() < CPU_TIMED_ITERS || (total_time < CPU_TIMED_MIN_S && samples.size() < CPU_TIMED_ITERS_MAX)) {
        ggml_cgraph * gf = prepare(pass++);
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        samples.push_back(t.stop());
        total_time += samples.back();
    }
    // the median once there are enough samples: one stalled pass must not price a fast op
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    return n >= 3 ? 0.5 * (samples[(n - 1) / 2] + samples[n / 2]) : total_time / n;
}

static double benchmark_cpu_dram_bandwidth(int threads) {
    const size_t pool_bytes = 1024ULL * 1024 * 1024;
    const size_t chunk_per_thread = pool_bytes / threads;
    const int iterations = 10;

    std::vector<uint8_t> pool(pool_bytes);
    for (size_t i = 0; i < pool.size(); i += 4096) {
        pool[i] = (uint8_t)(i & 0xFF);
    }

    std::vector<std::thread> workers;
    std::vector<double> thread_bytes(threads, 0.0);

    bench_timer t;
    t.start();

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&pool, &thread_bytes, tid, chunk_per_thread, iterations, pool_bytes]() {
            const size_t start = tid * chunk_per_thread;
            const size_t end_pos = (tid == (int)(pool_bytes / chunk_per_thread) - 1) ? pool_bytes : (start + chunk_per_thread);
            // four independent sums over every word: a streaming read, not one dependent
            // load per line that waits on the previous one
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            double local_bytes = 0.0;
            for (int iter = 0; iter < iterations; ++iter) {
                const uint64_t * p   = (const uint64_t *) (pool.data() + start);
                const uint64_t * end = (const uint64_t *) (pool.data() + (end_pos & ~(size_t) 31));
                for (; p + 4 <= end; p += 4) {
                    s0 += p[0]; s1 += p[1]; s2 += p[2]; s3 += p[3];
                }
                local_bytes += (double) ((const uint8_t *) end - (pool.data() + start));
            }
            volatile uint64_t sink = s0 + s1 + s2 + s3;
            (void) sink;
            thread_bytes[tid] = local_bytes;
        });
    }
    for (auto & w : workers) w.join();

    double elapsed = t.stop();
    double total_bytes = 0.0;
    for (int i = 0; i < threads; i++) total_bytes += thread_bytes[i];
    return total_bytes / elapsed / 1e9;
}

struct pcie_stress_ctx {
    std::atomic<bool> active{false};
    std::atomic<bool> stop{false};

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_buffer_t host_buf = nullptr;
    ggml_backend_buffer_t dev_buf  = nullptr;
    ggml_tensor * h_tensor = nullptr;
    ggml_tensor * d_tensor = nullptr;
    ggml_context * ctx = nullptr;
    size_t transfer_size = 256 * 1024 * 1024;
    double calibrated_bw_gb_s = 0.0;
    // bytes the stress loop moved and the time it ran: the concurrent PCIe rate is computed from these
    std::atomic<uint64_t> stress_bytes{0};
    std::atomic<uint64_t> stress_ns{0};
};

static void pcie_stress_loop(pcie_stress_ctx * pcie) {
    bench_timer t;
    t.start();
    uint64_t bytes = 0;
    pcie->active.store(true, std::memory_order_release);
    while (!pcie->stop.load(std::memory_order_acquire)) {
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        bytes += 2 * (uint64_t) pcie->transfer_size;
    }
    pcie->stress_bytes.fetch_add(bytes, std::memory_order_relaxed);
    pcie->stress_ns.fetch_add((uint64_t)(t.stop() * 1e9), std::memory_order_relaxed);
    pcie->active.store(false, std::memory_order_release);
}

// ---- machine calibrations (the profile header block) ----------------------------------------------------
// Every machine-specific number the planner prices with is calibrated here and written as a header line; the
// planner keeps no fallback constants for them. Kernel copies (transfers performed by kernels through the
// device mapping of pinned host memory, no copy-engine transition) are the expert pool's per-token path, so
// the gathered-upload curve is taken for the copy engine AND the two kernel paths, and the pool's per-layer
// fixed costs (host round trip, CPU-route handoff) are timed the way the runtime issues them.
#define CPU_PROFILE_SCHEMA 2
static const char * CPU_PROFILE_COLUMNS =
    "# op_name quant threads AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) Concurrent_GFLOP/s PCIe_Concurrent_BW N K B n_tokens ctx_len n_heads head_dim n_elements";

struct machine_id {
    std::string gpu = "none";
    std::string cpu = "unknown";
    std::string os  = "unknown";
    size_t      vram_mib = 0;
};

static std::string cpu_brand_string() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    char brand[49] = { 0 };
    for (int i = 0; i < 3; i++) {
        int r[4] = { 0, 0, 0, 0 };
#if defined(_MSC_VER)
        __cpuid(r, 0x80000002 + i);
#else
        __cpuid(0x80000002 + i, r[0], r[1], r[2], r[3]);
#endif
        memcpy(brand + 16 * i, r, 16);
    }
    std::string s;
    for (const char * p = brand; *p; p++) {
        if (*p == ' ' && (s.empty() || s.back() == ' ')) continue;
        s += *p;
    }
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s.empty() ? std::string("unknown") : s;
#else
    return "unknown";
#endif
}

static machine_id identify_machine(ggml_backend_t gpu) {
    machine_id m;
    m.cpu = cpu_brand_string();
#if defined(_WIN32)
    m.os = "windows";
#elif defined(__APPLE__)
    m.os = "macos";
#elif defined(__linux__)
    m.os = "linux";
#endif
    if (gpu) {
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
        if (dev) {
            const char * d = ggml_backend_dev_description(dev);
            if (d) m.gpu = d;
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(dev, &fr, &tot);
            m.vram_mib = tot >> 20;
        }
    }
    return m;
}

// backend procs the calibrations drive (the CUDA backend implements them; elsewhere only the copy-engine
// paths are calibrated and the kernel-copy lines are omitted)
struct gpu_procs {
    ggml_backend_copy_segments_async_t copy_segments       = nullptr;
    ggml_backend_kernel_copy_set_t     kernel_copy_set     = nullptr;
    ggml_backend_kernel_copy_max_set_t kernel_copy_max_set = nullptr;
    bool kernel_copies = false;   // the backend takes the switch
};

static gpu_procs lookup_gpu_procs(ggml_backend_t gpu) {
    gpu_procs p;
    ggml_backend_dev_t dev = gpu ? ggml_backend_get_device(gpu) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return p;
    p.copy_segments       = (ggml_backend_copy_segments_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_segments_async");
    p.kernel_copy_set     = (ggml_backend_kernel_copy_set_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_kernel_copy_set");
    p.kernel_copy_max_set = (ggml_backend_kernel_copy_max_set_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_kernel_copy_max_set");
    if (p.kernel_copy_set && p.kernel_copy_max_set) {
        p.kernel_copies = p.kernel_copy_set(true);
        p.kernel_copy_set(false);
    }
    return p;
}

// kernel copies on for a scope with the size cap lifted so every chunk takes the kernel path; restored on exit
struct kernel_copy_scope {
    const gpu_procs & p;
    bool   on;
    size_t prev_cap = 0;
    kernel_copy_scope(const gpu_procs & procs, bool enable) : p(procs), on(enable) {
        if (p.kernel_copy_set) p.kernel_copy_set(on);
        if (on && p.kernel_copy_max_set) prev_cap = p.kernel_copy_max_set(SIZE_MAX);
    }
    ~kernel_copy_scope() {
        if (on && p.kernel_copy_max_set) p.kernel_copy_max_set(prev_cap);
        if (p.kernel_copy_set) p.kernel_copy_set(false);
    }
};

// concurrent host DRAM load (decode runs the CPU expert chain while the pool uploads): threads-1 readers
struct dram_stress {
    std::atomic<bool>        stop{false};
    std::vector<std::thread> workers;
    std::vector<uint8_t>     pool;
    void start(int threads) {
        const size_t pool_bytes = 512ULL * 1024 * 1024;
        pool.resize(pool_bytes);
        for (size_t i = 0; i < pool.size(); i += 4096) pool[i] = (uint8_t)(i & 0xFF);
        const int n = std::max(1, threads - 1);   // one core stays with the enqueue thread
        for (int tid = 0; tid < n; ++tid) {
            workers.emplace_back([this, tid, n, pool_bytes]() {
                const size_t chunk = pool_bytes / n;
                const size_t start = tid * chunk;
                const size_t limit = start + chunk - sizeof(uint64_t);
                volatile uint64_t sink = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    for (size_t off = start; off + 64 <= limit; off += 64) {
                        sink += *(const uint64_t *)(pool.data() + off);
                    }
                }
                (void)sink;
            });
        }
    }
    void finish() {
        stop.store(true, std::memory_order_release);
        for (auto & w : workers) w.join();
        workers.clear();
    }
};

struct calib_results {
    bool       has_gpu = false;
    machine_id machine;
    double dram_bw = 0.0, pcie_standalone = 0.0, pcie_concurrent = 0.0;
    double cpu_eff = -1.0;                                  // CPU efficiency under PCIe load, from the op tables (-1 = not run)
    double sliced_bw[4]        = { 0.0, 0.0, 0.0, 0.0 };    // gathered uploads, copy engine
    double sliced_kernel_bw[4] = { 0.0, 0.0, 0.0, 0.0 };    // gathered uploads, one copy kernel per chunk
    double segs_kernel_bw[4]   = { 0.0, 0.0, 0.0, 0.0 };    // gathered uploads, one segment-batch launch per burst
    double segs_kernel_idle_bw[4] = { 0.0, 0.0, 0.0, 0.0 }; // the same with the CPU idle (fetch-only pool: no CPU chain)
    double staged_bw = 0.0;                                 // pageable source through the staging ring
    static const int n_cross = 9;
    static constexpr double cross_mb[n_cross] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    double kernel_bw[n_cross] = { 0 };                      // one transfer ordered behind a kernel, copy kernel
    double dma_bw[n_cross]    = { 0 };                      // the same on the copy engine
    double kernel_cap_mb = -1.0;                            // largest size at which the kernel copy still wins (-1 = not calibrated)
    double engine_switch_us = 0.0;
    double pool_serve_us = 0.0, pool_split_us = 0.0;
    double pin_ceiling_gb = 0.0;
};

enum sliced_mode { SLICED_DMA = 0, SLICED_KERNEL = 1, SLICED_SEGS = 2 };
static const double sliced_chunk_mb[4] = { 0.5, 2.0, 8.0, 32.0 };

// gathered-slice upload bandwidth: many small strided host->device chunks per burst (the runtime's
// sliced-by-used-ids expert copies: ~top-k experts x 3 expert tensors enqueued back-to-back, one synchronize
// per split), under concurrent CPU DRAM load. Three paths: the copy engine (legacy sliced tiers), one copy
// kernel per chunk, and the segment-batch kernel (the pool's per-layer upload). Small chunks run far below
// peak PCIe; the predictor interpolates these curves to price sliced uploads.
static void calibrate_pcie_sliced(pcie_stress_ctx * pcie, const gpu_procs & procs, int threads, int mode, bool loaded, double out_bw[4]) {
    static const char * names[3] = { "copy engine", "copy kernels", "segment-batch kernel" };
    printf("Calibrating gathered-slice upload bandwidth, %s (%s)...\n", names[mode], loaded ? "concurrent CPU load" : "CPU idle");
    kernel_copy_scope kc(procs, mode != SLICED_DMA);
    dram_stress stress;
    if (loaded) stress.start(threads);
    for (int c = 0; c < 4; c++) {
        const size_t chunk  = (size_t)(sliced_chunk_mb[c] * 1024.0 * 1024.0);
        const int    burst  = 24;                      // ~ top-k(8) experts x gate/up/down
        const size_t stride = chunk + 1024 * 1024;     // gathered: non-adjacent sources
        const size_t span   = pcie->transfer_size - chunk;
        const int    iters  = std::max(2, (int)(3.0e9 / ((double)burst * chunk)));
        std::vector<ggml_backend_copy_segment> segs(burst);
        bool taken = true;
        bench_timer t;
        t.start();
        double bytes = 0.0;
        for (int it = 0; it < iters && taken; ++it) {
            for (int b = 0; b < burst; b++) {
                const size_t off = ((size_t)(it * burst + b) * stride) % span;
                if (mode == SLICED_SEGS) {
                    segs[b] = { (char *)pcie->d_tensor->data + off, (const char *)pcie->h_tensor->data + off, chunk };
                } else {
                    ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor,
                        (const char *)pcie->h_tensor->data + off, off, chunk);
                }
            }
            if (mode == SLICED_SEGS) {
                taken = procs.copy_segments(pcie->gpu_backend, segs.data(), burst);
            }
            ggml_backend_synchronize(pcie->gpu_backend);
            bytes += (double)burst * chunk;
        }
        const double elapsed = t.stop();
        out_bw[c] = taken ? bytes / elapsed / 1e9 : 0.0;
        if (taken) printf("  %4.1fMB chunks: %.1f GB/s\n", sliced_chunk_mb[c], out_bw[c]);
        else       printf("  %4.1fMB chunks: segment kernel not taken\n", sliced_chunk_mb[c]);
    }
    if (loaded) stress.finish();
    printf("\n");
}

// PCIe rate while the CPU streams DRAM. A full run measures it per op while the tables run; the
// calibration-only modes measure it against the same DRAM readers the sliced curves use.
static double calibrate_pcie_concurrent(pcie_stress_ctx * pcie, int threads) {
    printf("Calibrating concurrent PCIe bandwidth (CPU DRAM load)...\n");
    dram_stress stress;
    stress.start(threads);
    const uint64_t b0 = pcie->stress_bytes.load(), n0 = pcie->stress_ns.load();
    pcie->stop.store(false, std::memory_order_release);
    std::thread th(pcie_stress_loop, pcie);
    while (!pcie->active.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    pcie->stop.store(true, std::memory_order_release);
    th.join();
    stress.finish();
    const uint64_t db = pcie->stress_bytes.load() - b0, dn = pcie->stress_ns.load() - n0;
    const double bw = dn > 0 ? (double)db / ((double)dn / 1e9) / 1e9 : 0.0;
    printf("  Concurrent PCIe BW: %.1f GB/s\n\n", bw);
    return bw;
}

// small graphs to order transfers against and to stand in for the pool's router / expert / join kernels
struct small_graph {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buf    = nullptr;
    ggml_tensor *         x      = nullptr;
    ggml_tensor *         y      = nullptr;
    ggml_tensor *         ids    = nullptr;   // 16 x i32: a router's top-k ids
    ggml_tensor *         ids2   = nullptr;
    ggml_cgraph *         g_x    = nullptr;   // scale(x)
    ggml_cgraph *         g_y    = nullptr;   // scale(y)
    ggml_cgraph *         g_join = nullptr;   // add(x, y)
    bool init(ggml_backend_t be, int64_t n) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 3 + 4096, nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        x    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        y    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
        ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
        ggml_tensor * sx = ggml_scale(ctx, x, 1.0f);
        ggml_tensor * sy = ggml_scale(ctx, y, 1.0f);
        ggml_tensor * j  = ggml_add(ctx, x, y);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) return false;
        g_x = ggml_new_graph(ctx);    ggml_build_forward_expand(g_x, sx);
        g_y = ggml_new_graph(ctx);    ggml_build_forward_expand(g_y, sy);
        g_join = ggml_new_graph(ctx); ggml_build_forward_expand(g_join, j);
        return true;
    }
    void release() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

// copy kernel vs copy engine for one transfer ordered behind a kernel: on WDDM a copy-engine transfer ordered
// against a kernel idles the GPU for the engine transition, a copy kernel does not. The cap is the largest size
// at which the kernel still wins; the transition is timed on its own.
static void calibrate_copy_crossover(pcie_stress_ctx * pcie, const gpu_procs & procs, calib_results & cr) {
    printf("Calibrating copy kernel vs copy engine for transfers ordered behind a kernel...\n");
    ggml_backend_t gpu = pcie->gpu_backend;
    small_graph sg;
    if (!sg.init(gpu, 4096)) { printf("  could not build the probe graph - skipped\n\n"); return; }
    for (int pass = 0; pass < 2; pass++) {
        kernel_copy_scope kc(procs, pass == 1);
        double * out = pass == 1 ? cr.kernel_bw : cr.dma_bw;
        for (int i = 0; i < calib_results::n_cross; i++) {
            const size_t size  = (size_t)(calib_results::cross_mb[i] * 1024.0 * 1024.0);
            const int    iters = std::max(4, (int)(1.5e9 / (double)size));
            auto body = [&]() {
                ggml_backend_graph_compute_async(gpu, sg.g_x);
                ggml_backend_tensor_set_async(gpu, pcie->d_tensor, pcie->h_tensor->data, 0, size);
            };
            for (int w = 0; w < 2; w++) body();
            ggml_backend_synchronize(gpu);
            bench_timer t;
            t.start();
            for (int it = 0; it < iters; it++) body();
            ggml_backend_synchronize(gpu);
            out[i] = (double)iters * size / t.stop() / 1e9;
        }
    }
    printf("  %8s %12s %12s\n", "size", "kernel GB/s", "engine GB/s");
    for (int i = 0; i < calib_results::n_cross; i++) {
        printf("  %6.0fMB %12.1f %12.1f\n", calib_results::cross_mb[i], cr.kernel_bw[i], cr.dma_bw[i]);
    }
    cr.kernel_cap_mb = 0.0;
    for (int i = 0; i < calib_results::n_cross; i++) {
        if (cr.kernel_bw[i] > cr.dma_bw[i]) cr.kernel_cap_mb = calib_results::cross_mb[i]; else break;
    }
    // the transition itself: a small readback behind a kernel, synchronized each time, copy engine vs kernel
    double tt[2] = { 0.0, 0.0 };
    for (int pass = 0; pass < 2; pass++) {
        kernel_copy_scope kc(procs, pass == 1);
        const int iters = 1000;
        auto body = [&]() {
            ggml_backend_graph_compute_async(gpu, sg.g_x);
            ggml_backend_tensor_get_async(gpu, pcie->d_tensor, pcie->h_tensor->data, 0, 8192);
            ggml_backend_synchronize(gpu);
        };
        for (int w = 0; w < 20; w++) body();
        bench_timer t;
        t.start();
        for (int it = 0; it < iters; it++) body();
        tt[pass] = t.stop() / iters * 1e6;
    }
    cr.engine_switch_us = std::max(0.0, tt[0] - tt[1]);
    printf("  kernel -> 8 KB readback -> sync: copy engine %.1f us, copy kernel %.1f us: engine switch %.1f us\n",
        tt[0], tt[1], cr.engine_switch_us);
    printf("  kernel copies win up to %.0f MB\n\n", cr.kernel_cap_mb);
    sg.release();
}

// the expert pool's per-layer fixed costs, issued the way the runtime issues them (kernel copies on):
//   serve: router kernel -> ids readback (64 B) -> event sync -> decision -> ids upload (64 B) -> expert launch
//   split: router kernel -> activation download (8 KB) -> event sync -> CPU graph on a persistent thread pool
//          -> partial upload (8 KB) -> join kernel
// each reported net of the same kernels launched back to back (the GPU with nothing in between)
static void calibrate_pool_latencies(pcie_stress_ctx * pcie, ggml_backend_t cpu_be, int threads, const gpu_procs & procs, calib_results & cr) {
    printf("Calibrating the expert pool's per-layer host round trips (kernel copies on)...\n");
    kernel_copy_scope kc(procs, true);
    ggml_backend_t gpu = pcie->gpu_backend;
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    small_graph sg;
    if (!sg.init(gpu, 2048)) { printf("  could not build the probe graph - skipped\n\n"); return; }
    ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * cctx = ggml_init(ip);
    ggml_tensor * xc = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, 2048);
    ggml_tensor * sc = ggml_scale(cctx, xc, 1.0f);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu_be);
    ggml_cgraph * g_cpu = ggml_new_graph(cctx);
    ggml_build_forward_expand(g_cpu, sc);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_n_threads(cpu_be, threads);
    ggml_backend_cpu_set_threadpool(cpu_be, tp);
    ggml_backend_event_t ev = ggml_backend_event_new(dev);
    char * host = (char *)pcie->h_tensor->data;   // pinned and device-mapped: the kernel-copy paths apply
    char * ids_host = host, * ids_host2 = host + 4096, * x_host = host + 8192, * y_host = host + 65536;

    auto timed = [&](const std::function<void()> & body) {
        const int iters = 1000;
        for (int w = 0; w < 50; w++) body();
        ggml_backend_synchronize(gpu);
        bench_timer t;
        t.start();
        for (int it = 0; it < iters; it++) body();
        ggml_backend_synchronize(gpu);
        return t.stop() / iters * 1e6;
    };
    const double t_kk = timed([&]() {
        ggml_backend_graph_compute_async(gpu, sg.g_x);
        ggml_backend_graph_compute_async(gpu, sg.g_y);
    });
    const double t_serve = timed([&]() {
        ggml_backend_graph_compute_async(gpu, sg.g_x);
        ggml_backend_tensor_get_async(gpu, sg.ids, ids_host, 0, 64);
        ggml_backend_event_record(ev, gpu);
        ggml_backend_event_synchronize(ev);
        ggml_backend_tensor_set_async(gpu, sg.ids2, ids_host2, 0, 64);
        ggml_backend_graph_compute_async(gpu, sg.g_y);
    });
    const double t_kj = timed([&]() {
        ggml_backend_graph_compute_async(gpu, sg.g_x);
        ggml_backend_graph_compute_async(gpu, sg.g_join);
    });
    const double t_split = timed([&]() {
        ggml_backend_graph_compute_async(gpu, sg.g_x);
        ggml_backend_tensor_get_async(gpu, sg.x, x_host, 0, 8192);
        ggml_backend_event_record(ev, gpu);
        ggml_backend_event_synchronize(ev);
        ggml_backend_graph_compute_async(cpu_be, g_cpu);
        ggml_backend_tensor_set_async(gpu, sg.y, y_host, 0, 8192);
        ggml_backend_graph_compute_async(gpu, sg.g_join);
    });
    cr.pool_serve_us = std::max(0.0, t_serve - t_kk);
    cr.pool_split_us = std::max(0.0, t_split - t_kj);
    printf("  two kernels back to back %.1f us; with the ids round trip %.1f us: serve %.1f us per layer\n",
        t_kk, t_serve, cr.pool_serve_us);
    printf("  kernel + join %.1f us; with the CPU-route handoff %.1f us: split %.1f us per route\n\n",
        t_kj, t_split, cr.pool_split_us);

    ggml_backend_event_free(ev);
    ggml_backend_cpu_set_threadpool(cpu_be, nullptr);
    ggml_threadpool_free(tp);
    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    sg.release();
}

// pageable upload through the staging ring: the loader's path for mmap mappings past the host pin ceiling
static double calibrate_staged_upload(pcie_stress_ctx * pcie) {
    printf("Calibrating pageable upload through the staging ring...\n");
    std::vector<uint8_t> src(pcie->transfer_size);
    for (size_t i = 0; i < src.size(); i += 4096) src[i] = (uint8_t)(i & 0xFF);
    const size_t chunk = 64ULL << 20;   // the loader's chunking
    auto pass = [&]() {
        for (size_t off = 0; off < pcie->transfer_size; off += chunk) {
            ggml_backend_tensor_set(pcie->d_tensor, src.data() + off, off, std::min(chunk, pcie->transfer_size - off));
        }
        ggml_backend_synchronize(pcie->gpu_backend);
    };
    pass();
    const int iters = 4;
    bench_timer t;
    t.start();
    for (int it = 0; it < iters; it++) pass();
    const double bw = (double)iters * pcie->transfer_size / t.stop() / 1e9;
    printf("  Staged (pageable) upload BW: %.1f GB/s\n\n", bw);
    return bw;
}

// the profile header: every calibrated machine number as one parseable line
static void write_profile_header(FILE * f, const calib_results & cr, int threads, const std::vector<int32_t> * batch_sizes, const char * first_line) {
    if (first_line) {
        fprintf(f, "%s\n", first_line);
    } else {
        fprintf(f, "# Concurrent Profiling (threads=%d, batch_sizes=[", threads);
        if (batch_sizes) {
            for (size_t i = 0; i < batch_sizes->size(); i++) {
                fprintf(f, "%d%s", (*batch_sizes)[i], i + 1 < batch_sizes->size() ? "," : "");
            }
        }
        fprintf(f, "])\n");
    }
    fprintf(f, "#   Machine: gpu=\"%s\" vram_mib=%zu cpu=\"%s\" threads=%d os=%s schema=%d\n",
        cr.machine.gpu.c_str(), cr.machine.vram_mib, cr.machine.cpu.c_str(), threads, cr.machine.os.c_str(), CPU_PROFILE_SCHEMA);
    fprintf(f, "# Measured Bandwidths Per Thread Count:\n");
    if (!cr.has_gpu) {
        fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s\n", threads, cr.dram_bw);
        return;
    }
    fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s, PCIe_Standalone=%.1f GB/s, PCIe_Concurrent=%.1f GB/s (CPU_Eff=%.1f%%)\n",
        threads, cr.dram_bw, cr.pcie_standalone, cr.pcie_concurrent, cr.cpu_eff >= 0.0 ? cr.cpu_eff : 100.0);
    auto curve = [&](const char * name, const double bw[4]) {
        if (bw[0] > 0.0) {
            fprintf(f, "#   %s: 0.5MB=%.1f 2MB=%.1f 8MB=%.1f 32MB=%.1f GB/s\n", name, bw[0], bw[1], bw[2], bw[3]);
        }
    };
    curve("PCIe_Sliced", cr.sliced_bw);
    curve("PCIe_Sliced_Kernel", cr.sliced_kernel_bw);
    curve("PCIe_Segs_Kernel", cr.segs_kernel_bw);
    curve("PCIe_Segs_Kernel_Idle", cr.segs_kernel_idle_bw);
    if (cr.staged_bw > 0.0) fprintf(f, "#   PCIe_Staged: %.1f GB/s\n", cr.staged_bw);
    if (cr.kernel_cap_mb >= 0.0) {
        auto line = [&](const char * name, const double bw[]) {
            fprintf(f, "#   %s:", name);
            for (int i = 0; i < calib_results::n_cross; i++) fprintf(f, " %.0fMB=%.1f", calib_results::cross_mb[i], bw[i]);
            fprintf(f, " GB/s\n");
        };
        line("Kernel_Copy", cr.kernel_bw);
        line("Engine_Copy", cr.dma_bw);
        fprintf(f, "#   Kernel_Copy_Cap_MB: %.0f\n", cr.kernel_cap_mb);
        fprintf(f, "#   Engine_Switch_us: %.1f\n", cr.engine_switch_us);
    }
    if (cr.pool_serve_us > 0.0 || cr.pool_split_us > 0.0) {
        fprintf(f, "#   Pool_Serve_us: %.1f\n", cr.pool_serve_us);
        fprintf(f, "#   Pool_Split_us: %.1f\n", cr.pool_split_us);
    }
    if (cr.pin_ceiling_gb > 0.0) fprintf(f, "#   Host_Pin_Ceiling: %.1f GB\n", cr.pin_ceiling_gb);
}

// replace the header block of an existing profile with freshly calibrated lines, keeping its op tables (they
// take the long run; the calibrations take seconds). Values the calibration modes do not produce (CPU_Eff,
// the pin ceiling when its probe was skipped) are carried over from the old header. The old file is kept
// as <path>.bak.
static bool splice_profile_header(const char * path, calib_results & cr, int threads) {
    FILE * in = fopen(path, "r");
    if (!in) { fprintf(stderr, "cannot open %s\n", path); return false; }
    std::vector<std::string> lines;
    char buf[4096];
    while (fgets(buf, sizeof(buf), in)) lines.push_back(buf);
    fclose(in);
    size_t cols = lines.size();
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].compare(0, 9, "# op_name") == 0) { cols = i; break; }
    }
    if (cols == lines.size() || lines.empty()) {
        fprintf(stderr, "%s: no column header line - not a CPU profile\n", path);
        return false;
    }
    std::string first = lines[0];
    while (!first.empty() && (first.back() == '\n' || first.back() == '\r')) first.pop_back();
    for (size_t i = 0; i < cols; i++) {
        double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
        int tc = 0;
        if (sscanf(lines[i].c_str(), "#   Threads=%d: DRAM_BW=%lf GB/s, PCIe_Standalone=%lf GB/s, PCIe_Concurrent=%lf GB/s (CPU_Eff=%lf%%)",
                   &tc, &a, &b, &c, &d) == 5) {
            if (cr.cpu_eff < 0.0) cr.cpu_eff = d;
            if (cr.pcie_concurrent <= 0.0) cr.pcie_concurrent = c;
            if (tc != threads) {
                printf("WARNING: the op tables in %s were measured at %d threads, this splice ran at %d: the planner matches the\n"
                       "         thread count and will refuse the file - rerun with --threads %d\n", path, tc, threads, tc);
            }
        }
        double pc = 0.0;
        if (sscanf(lines[i].c_str(), "#   Host_Pin_Ceiling: %lf GB", &pc) == 1 && cr.pin_ceiling_gb <= 0.0) {
            cr.pin_ceiling_gb = pc;
        }
    }
    const std::string bak = std::string(path) + ".bak";
    FILE * fb = fopen(bak.c_str(), "w");
    if (fb) {
        for (auto & l : lines) fputs(l.c_str(), fb);
        fclose(fb);
    }
    FILE * out = fopen(path, "w");
    if (!out) { fprintf(stderr, "cannot write %s\n", path); return false; }
    write_profile_header(out, cr, threads, nullptr, first.c_str());
    for (size_t i = cols; i < lines.size(); i++) fputs(lines[i].c_str(), out);
    fclose(out);
    printf("Header replaced in %s (previous copy in %s)\n", path, bak.c_str());
    return true;
}

// host pin ceiling: how many bytes of ordinary process memory the driver will
// page-lock (cudaHostRegister). pshard's streamed weights source from mmap
// regions registered exactly this way; mappings past the ceiling stay pageable
// and stage through a pinned ring at a host-DRAM-bound rate. Register 2 GiB
// anonymous chunks until the driver refuses (or RAM runs short), report the total.
static double calibrate_pin_ceiling(pcie_stress_ctx * pcie) {
    printf("Probing host pin ceiling (cudaHostRegister until refusal)...\n");
    ggml_backend_dev_t dev = ggml_backend_get_device(pcie->gpu_backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto register_fn = reg ? (bool (*)(void *, size_t))
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer") : nullptr;
    auto unregister_fn = reg ? (void (*)(void *))
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer") : nullptr;
    if (!register_fn || !unregister_fn) {
        printf("  backend does not expose host registration - skipping\n\n");
        return 0.0;
    }
    // stop short of AVAILABLE memory: past it the probe faults into swap and
    // measures the pager (or feeds the OOM killer), not the driver
    size_t avail = 0;
#ifdef _WIN32
    {
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) avail = (size_t) ms.ullAvailPhys;
    }
#else
    {
        long pages = sysconf(_SC_AVPHYS_PAGES), psize = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && psize > 0) avail = (size_t) pages * (size_t) psize;
    }
#endif
    const size_t cap = avail > 0 ? (size_t)(0.90 * avail) : (256ull << 30);
    // descending chunk sizes: after a refusal, smaller chunks tighten the floor to
    // 256 MiB granularity (a sub-2 GiB ceiling would otherwise read as 0 = probe
    // skipped, and the planner would price everything at the pinned rate)
    static const size_t chunk_sizes[] = { 2ull << 30, 1ull << 30, 512ull << 20, 256ull << 20 };
    std::vector<std::pair<void *, size_t>> chunks;
    size_t total = 0;
    bool cap_hit = false;
    for (size_t chunk : chunk_sizes) {
        while (true) {
            if (total + chunk > cap) { cap_hit = true; break; }
            void * p = malloc(chunk);
            if (!p) { cap_hit = true; break; }
            memset(p, 1, chunk);   // fault the pages in - the driver locks real pages
            if (!register_fn(p, chunk)) { free(p); break; }
            chunks.push_back({ p, chunk });
            total += chunk;
        }
    }
    for (auto & [p, sz] : chunks) { unregister_fn(p); free(p); }
    const double gb = total / 1e9;
    if (cap_hit) {
        printf("  Host pin ceiling: >= %.1f GB (stopped at 90%% of available RAM - a floor, not the driver's limit)\n\n", gb);
    } else {
        printf("  Host pin ceiling: %.1f GB\n\n", gb);
    }
    return gb;
}

static void calibrate_pcie(pcie_stress_ctx * pcie) {
    printf("Calibrating standalone PCIe bandwidth...\n");
    bench_timer t;
    t.start();
    const int cal_iterations = 20;
    for (int i = 0; i < cal_iterations; ++i) {
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
    }
    double elapsed = t.stop();
    double bytes_moved = (double)cal_iterations * pcie->transfer_size * 2.0;
    pcie->calibrated_bw_gb_s = bytes_moved / elapsed / 1e9;
    printf("  Standalone PCIe BW: %.1f GB/s\n\n", pcie->calibrated_bw_gb_s);
}

struct bench_result_cpu : bench_result {
    int threads = 0;
    float standalone_gflops = 0.0f;
    float concurrent_gflops = 0.0f;
    float concurrent_efficiency_pct = 0.0f;
    float pcie_standalone_bw_gb_s = 0.0f;
    float pcie_concurrent_gb_s = 0.0f;   // PCIe rate while this op ran under the stress loop

    void print(double pcie_bw_ref = 0.0) const {
        printf("%-20s quant=%-6s threads=%d AI=%.3f FLOP/byte BW=%.2f GB/s Perf=%.2f GFLOP/s",
            op_name.c_str(), quant_type.c_str(), threads,
            arithmetic_intensity, effective_bw_gb_s, effective_gflops);
        if (concurrent_gflops > 0) {
            printf(" | Concur=%.2f (%.1f%%)", concurrent_gflops, concurrent_efficiency_pct);
            if (pcie_bw_ref > 0 && pcie_concurrent_gb_s > 0) {
                printf(" PCIe=%.1f GB/s (%.1f%%)", pcie_concurrent_gb_s, 100.0 * pcie_concurrent_gb_s / pcie_bw_ref);
            }
        }
        print_dims();
        printf("\n");
    }
};

static double benchmark_mul_mat_raw(
        ggml_backend_t be, int N, int K, int batch_size,
        ggml_type quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 4096ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);

    // one weight copy per pass in rotation, each with its own graph
    const int n_copies = bench_n_copies(ggml_row_size(quant, K) * N + (size_t) N * batch_size * sizeof(float));
    std::vector<ggml_tensor *> A(n_copies);
    std::vector<ggml_cgraph *> gf(n_copies);
    ggml_tensor * C = nullptr;
    for (int c = 0; c < n_copies; c++) {
        A[c]  = ggml_new_tensor_2d(ctx, quant, K, N);
        gf[c] = ggml_new_graph_custom(ctx, 8, false);
        C = ggml_mul_mat(ctx, A[c], B_tensor);
        ggml_build_forward_expand(gf[c], C);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d %s (alloc failed)\n", N, K, batch_size, ggml_type_name(quant));
        ggml_free(ctx); return 0.0;
    }
    ggml_backend_buffer_clear(buffer, 0);   // touch every page now, not inside a timed pass

    std::vector<uint8_t> A_data = create_quantized_data(quant, (int64_t)K * N);
    std::vector<float> B_data(K * batch_size, 1.0f);
    for (int c = 0; c < n_copies; c++) {
        ggml_backend_tensor_set(A[c], A_data.data(), 0, ggml_nbytes(A[c]));
    }
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));

    double time_per_iter = time_op(be, [&](int pass) { return gf[pass % n_copies]; });
    double ops_total = 2.0 * N * K * batch_size;
    double bytes_total = (double)(ggml_nbytes(A[0]) + ggml_nbytes(B_tensor) + ggml_nbytes(C));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_mul_mat_id_raw(
        ggml_backend_t be, int N, int K, int n_experts, int n_experts_used,
        int batch_size, ggml_type quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    // a small expert stack gets copies too: the rotation walks the copies and, once around,
    // moves on to the next experts, so no pass rereads bytes a recent pass touched
    const int n_copies = bench_n_copies(ggml_row_size(quant, (int64_t) K * N * n_experts)
                                        + (size_t) N * batch_size * n_experts_used * sizeof(float));
    std::vector<ggml_tensor *> A(n_copies);
    std::vector<ggml_cgraph *> gf(n_copies);
    for (int c = 0; c < n_copies; c++) {
        A[c]  = ggml_new_tensor_3d(ctx, quant, K, N, n_experts);
        gf[c] = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(gf[c], ggml_mul_mat_id(ctx, A[c], B, ids));
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (alloc failed)\n", N, K, batch_size);
        ggml_free(ctx); return 0.0;
    }
    ggml_backend_buffer_clear(buffer, 0);   // touch every page now, not inside a timed pass

    std::vector<uint8_t> A_data = create_quantized_data(quant, (int64_t)K * N * n_experts);
    for (int c = 0; c < n_copies; c++) {
        ggml_backend_tensor_set(A[c], A_data.data(), 0, ggml_nbytes(A[c]));
    }
    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    // every token routes to its own experts until the stack is exhausted: the batch streams
    // min(B * used, E) distinct experts, the independent-token routing the planner prices
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    const int n_distinct = std::min(n_experts_used * batch_size, n_experts);
    auto prepare = [&](int pass) {
        const int copy  = pass % n_copies;
        const int shift = (int) ((((int64_t) pass / n_copies) * n_distinct) % n_experts);
        for (int i = 0; i < n_experts_used * batch_size; i++) {
            ids_data[i] = (i + shift) % n_experts;
        }
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
        return gf[copy];
    };

    double time_per_iter = time_op(be, prepare);
    double ops_total = 2.0 * N * K * batch_size * n_experts_used;
    double bytes_total = (double) ggml_nbytes(A[0]) * n_distinct / n_experts + ggml_nbytes(B) + (double)(N * batch_size * n_experts_used * 4);

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_flash_attn_raw(
        ggml_backend_t be, int n_tokens, int ctx_len,
        int n_q_heads, int n_kv_heads, int head_dim,
        ggml_type kv_quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, 1);

    // one KV block per pass in rotation, each with its own graph
    const int n_copies = bench_n_copies(2 * ggml_row_size(kv_quant, (int64_t) head_dim * ctx_len * n_kv_heads)
                                        + (size_t) head_dim * n_tokens * n_q_heads * sizeof(float));
    std::vector<ggml_tensor *> K(n_copies), V(n_copies);
    std::vector<ggml_cgraph *> gf(n_copies);
    ggml_tensor * out = nullptr;
    for (int c = 0; c < n_copies; c++) {
        K[c]  = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);
        V[c]  = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);
        gf[c] = ggml_new_graph_custom(ctx, 8, false);
        out = ggml_flash_attn_ext(ctx, Q, K[c], V[c], nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
        ggml_build_forward_expand(gf[c], out);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: FLASH_ATTN (alloc failed)\n");
        ggml_free(ctx); return 0.0;
    }
    ggml_backend_buffer_clear(buffer, 0);   // touch every page now, not inside a timed pass

    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    std::vector<uint8_t> KV_data = create_quantized_data(kv_quant, ggml_nelements(K[0]));
    for (int c = 0; c < n_copies; c++) {
        ggml_backend_tensor_set(K[c], KV_data.data(), 0, ggml_nbytes(K[c]));
        ggml_backend_tensor_set(V[c], KV_data.data(), 0, ggml_nbytes(V[c]));
    }

    double time_per_iter = time_op(be, [&](int pass) { return gf[pass % n_copies]; });
    double ops_total = 2.0 * n_tokens * head_dim * ctx_len * n_q_heads * 2;
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K[0]) + ggml_nbytes(V[0]) + ggml_nbytes(out));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static bench_result_cpu run_concurrent(
        std::function<double(double *, double *, double *)> bench_fn,
        const std::string & op_name, const char * quant_name, int threads,
        pcie_stress_ctx * pcie) {

    bench_result_cpu result;
    result.op_name = op_name;
    result.quant_type = quant_name;
    result.threads = threads;
    result.pcie_standalone_bw_gb_s = pcie ? (float)pcie->calibrated_bw_gb_s : 0.0f;

    double standalone = bench_fn(&result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    result.standalone_gflops = result.effective_gflops;

    if (pcie && pcie->gpu_backend) {
        const uint64_t b0 = pcie->stress_bytes.load(), n0 = pcie->stress_ns.load();
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        while (!pcie->active.load(std::memory_order_acquire)) std::this_thread::yield();

        result.concurrent_gflops = (float)bench_fn(nullptr, nullptr, nullptr);

        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
        const uint64_t db = pcie->stress_bytes.load() - b0, dn = pcie->stress_ns.load() - n0;
        result.pcie_concurrent_gb_s = dn > 0 ? (float)((double)db / ((double)dn / 1e9) / 1e9) : 0.0f;
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }

    result.concurrent_efficiency_pct = (standalone > 0.0)
        ? (float)(std::min)(100.0, 100.0 * result.concurrent_gflops / result.standalone_gflops)
        : 100.0f;

    return result;
}

static void run_matmul_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto sizes  = get_matmul_sizes(fast);
    auto quants = get_matmul_quants(fast);

    printf("=== MUL_MAT Operations ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            printf("  [Batch=%d]\n", bs);
            for (const auto & sz : sizes) {
                if (qt == GGML_TYPE_Q2_K && (sz.K % 256 != 0)) continue;

                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_mul_mat_raw(be, sz.N, sz.K, bs, qt, threads, t, o, b);
                    }, "MUL_MAT", ggml_type_name(qt), threads, pcie);
                res.N = sz.N; res.K = sz.K; res.B = bs;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void run_moe_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto configs = get_moe_configs(fast);
    auto quants  = get_matmul_quants(fast);

    printf("=== MUL_MAT_ID Operations (MoE) ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- MoE Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            printf("  [Batch=%d]\n", bs);
            for (const auto & cfg : configs) {
                if (qt == GGML_TYPE_Q2_K && (cfg.K % 256 != 0)) continue;

                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_mul_mat_id_raw(be, cfg.N, cfg.K, cfg.n_experts, cfg.n_experts_used, bs, qt, threads, t, o, b);
                    }, "MUL_MAT_ID", ggml_type_name(qt), threads, pcie);
                res.N = cfg.N; res.K = cfg.K; res.B = bs;
                res.n_tokens = cfg.n_experts_used; res.ctx_len = cfg.n_experts;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void run_attention_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto configs  = get_attn_configs(fast);
    auto ctx_lens = get_attn_ctx_lens(fast);

    printf("=== FLASH_ATTN Operations ===\n\n");
    for (const auto & cfg : configs) {
        printf("--- %s (n_q=%d, n_kv=%d, head_dim=%d) ---\n", cfg.name, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim);
        for (int32_t n_tok : batch_sizes) {
            printf("  [n_tokens=%d]\n", n_tok);
            for (int32_t cl : ctx_lens) {
                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_flash_attn_raw(be, n_tok, cl, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, GGML_TYPE_F16, threads, t, o, b);
                    }, std::string("FLASH_ATTN_") + cfg.name, ggml_type_name(GGML_TYPE_F16), threads, pcie);
                res.n_tokens = n_tok; res.ctx_len = cl; res.n_heads = cfg.n_kv_heads; res.head_dim = cfg.head_dim;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void save_results_cpu(
        const char * path,
        const std::vector<bench_result_cpu> & results,
        const std::vector<int32_t> & batch_sizes,
        int threads, const calib_results & cr) {

    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); return; }

    write_profile_header(f, cr, threads, &batch_sizes, nullptr);
    fprintf(f, "%s\n", CPU_PROFILE_COLUMNS);

    const double dram_bw = cr.dram_bw;
    auto ridges = compute_ridge_points(results, dram_bw);
    std::map<std::string, double> ridge_map;
    for (const auto & rr : ridges) ridge_map[rr.key] = rr.ridge;

    for (const auto & r : results) {
        std::string key = r.op_name + "_" + r.quant_type;
        double ridge = ridge_map.count(key) ? ridge_map[key] : 0.0;
        const double est_pcie = r.pcie_concurrent_gb_s;   // the rate while the op ran under the PCIe stress loop

        fprintf(f, "%s %s %d %.4f %.2f %.2f %.4f %.2f %.2f %d %d %d %d %d %d %d %lld\n",
            r.op_name.c_str(), r.quant_type.c_str(), r.threads,
            r.arithmetic_intensity, r.effective_bw_gb_s, r.effective_gflops, ridge,
            r.concurrent_gflops, est_pcie,
            r.N, r.K, r.B, r.n_tokens, r.ctx_len, r.n_heads, r.head_dim, (long long)r.n_elements);
    }

    fclose(f);
    printf("Results saved to %s (%zu benchmarks)\n", path, results.size());
}

int main(int argc, char ** argv) {
    int32_t fixed_threads = -1;
    bool    fast_mode     = true;
    bool    calibrate_only = false;
    bool    no_pin_ceiling = false;
    const char * splice_path = nullptr;
    const char * output_path = "cpu_profile.txt";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], fixed_threads) || fixed_threads <= 0) {
                fprintf(stderr, "Invalid --threads value: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--fast")) {
            fast_mode = true;
        } else if (!strcmp(argv[i], "--full")) {
            fast_mode = false;
        } else if (!strcmp(argv[i], "--sliced-only") || !strcmp(argv[i], "--calibrate-only")) {
            calibrate_only = true;
        } else if (!strcmp(argv[i], "--splice") && i + 1 < argc) {
            splice_path = argv[++i];
        } else if (!strcmp(argv[i], "--no-pin-ceiling")) {
            no_pin_ceiling = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [options]\n", argv[0]);
            printf("\n");
            printf("options:\n");
            printf("  -h, --help\n");
            printf("  --fast              fast mode with fewer configs (default)\n");
            printf("  --full              full mode with all configs\n");
            printf("  --calibrate-only    run only the machine calibrations and print the\n");
            printf("                      profile header block (no table regeneration)\n");
            printf("  --splice <profile>  run the calibrations and replace that profile's header\n");
            printf("                      block in place, keeping its op tables (old file -> .bak)\n");
            printf("  --no-pin-ceiling    skip the host pin ceiling probe (registers RAM\n");
            printf("                      in 2 GiB chunks until the driver refuses)\n");
            printf("  --threads <n>       number of CPU threads (default: auto)\n");
            printf("  --output <path>     output file (default: cpu_profile.txt)\n");
            return 0;
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    int32_t default_threads = common_cpu_get_num_math();
    int threads = (fixed_threads > 0) ? fixed_threads : default_threads;
    // 4 and 16 are the decode tiers of a multi-sequence server: between the memory-bound
    // single token and the compute-bound prefill batches, priced from an entry in their own regime
    std::vector<int32_t> batch_sizes = { 1, 4, 16, 64, 512 };

    printf("=== CPU Profiler (cold-cache) ===\n");
    printf("Threads: %d%s\n", threads, fixed_threads > 0 ? " (user)" : " (auto)");
    printf("Mode:    %s\n\n", fast_mode ? "FAST" : "FULL");

    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_be) { fprintf(stderr, "Failed to initialize CPU backend\n"); return 1; }

    ggml_quantize_init(GGML_TYPE_Q2_K);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    ggml_quantize_init(GGML_TYPE_Q4_1);
    ggml_quantize_init(GGML_TYPE_Q5_0);
    ggml_quantize_init(GGML_TYPE_Q8_0);
    ggml_quantize_init(GGML_TYPE_MXFP4);

    pcie_stress_ctx pcie;
    pcie.gpu_backend = profiler_gpu_backend_init();
    bool has_gpu = (pcie.gpu_backend != nullptr);

    if (has_gpu) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(pcie.gpu_backend));
        if (host_buft) {
            pcie.host_buf = ggml_backend_buft_alloc_buffer(host_buft, pcie.transfer_size);
            pcie.dev_buf  = ggml_backend_alloc_buffer(pcie.gpu_backend, pcie.transfer_size);
            if (pcie.host_buf == nullptr || pcie.dev_buf == nullptr) {
                fprintf(stderr, "PCIe calibration: could not allocate the %zu MiB %s buffer - GPU lines omitted\n",
                    pcie.transfer_size >> 20, pcie.host_buf == nullptr ? "pinned host" : "device");
                has_gpu = false;
            }
            if (has_gpu && ggml_backend_buffer_get_type(pcie.host_buf) != host_buft) {
                fprintf(stderr, "PCIe calibration: the pinned host allocation fell back to a pageable buffer - the copies below take the staging path\n");
            }
            if (has_gpu) {
                ggml_init_params p = { pcie.transfer_size + 8 * 1024 * 1024, NULL, true };
                pcie.ctx = ggml_init(p);
                pcie.h_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
                pcie.d_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
                ggml_backend_tensor_alloc(pcie.host_buf, pcie.h_tensor, ggml_backend_buffer_get_base(pcie.host_buf));
                ggml_backend_tensor_alloc(pcie.dev_buf,  pcie.d_tensor, ggml_backend_buffer_get_base(pcie.dev_buf));
                std::vector<float> init_data(pcie.transfer_size / 4, 1.0f);
                ggml_backend_tensor_set(pcie.h_tensor, init_data.data(), 0, pcie.transfer_size);
                printf("GPU: %s\n", ggml_backend_name(pcie.gpu_backend));
            }
        } else {
            has_gpu = false;
        }
    }
    if (!has_gpu) printf("No GPU — standalone mode\n\n");

    calib_results cr;
    cr.has_gpu = has_gpu;
    cr.machine = identify_machine(has_gpu ? pcie.gpu_backend : nullptr);
    printf("Machine: gpu=\"%s\" (%zu MiB) cpu=\"%s\" os=%s\n\n",
        cr.machine.gpu.c_str(), cr.machine.vram_mib, cr.machine.cpu.c_str(), cr.machine.os.c_str());

    printf("Measuring DRAM bandwidth...\n");
    cr.dram_bw = benchmark_cpu_dram_bandwidth(threads);
    printf("  DRAM BW: %.1f GB/s\n", cr.dram_bw);

    gpu_procs procs;
    if (has_gpu) {
        calibrate_pcie(&pcie);
        cr.pcie_standalone = pcie.calibrated_bw_gb_s;
        procs = lookup_gpu_procs(pcie.gpu_backend);
        calibrate_pcie_sliced(&pcie, procs, threads, SLICED_DMA, true, cr.sliced_bw);
        if (procs.kernel_copies) {
            calibrate_pcie_sliced(&pcie, procs, threads, SLICED_KERNEL, true, cr.sliced_kernel_bw);
            if (procs.copy_segments) {
                calibrate_pcie_sliced(&pcie, procs, threads, SLICED_SEGS, true,  cr.segs_kernel_bw);
                calibrate_pcie_sliced(&pcie, procs, threads, SLICED_SEGS, false, cr.segs_kernel_idle_bw);
            }
            calibrate_copy_crossover(&pcie, procs, cr);
            calibrate_pool_latencies(&pcie, cpu_be, threads, procs, cr);
        } else {
            printf("Kernel copies not available on this backend - kernel-copy lines omitted\n\n");
        }
        cr.staged_bw = calibrate_staged_upload(&pcie);
        if (calibrate_only || splice_path) {
            cr.pcie_concurrent = calibrate_pcie_concurrent(&pcie, threads);
        }
        if (!no_pin_ceiling) cr.pin_ceiling_gb = calibrate_pin_ceiling(&pcie);
    }

    if (calibrate_only || splice_path) {
        if (splice_path) {
            splice_profile_header(splice_path, cr, threads);
        } else {
            if (has_gpu) {
                printf("CPU_Eff is measured by the op tables only: the header below carries 100%% (no derating)\n");
            }
            printf("Profile header block:\n");
            write_profile_header(stdout, cr, threads, &batch_sizes, nullptr);
        }
        if (has_gpu) {
            if (pcie.ctx) ggml_free(pcie.ctx);
            if (pcie.host_buf) ggml_backend_buffer_free(pcie.host_buf);
            if (pcie.dev_buf) ggml_backend_buffer_free(pcie.dev_buf);
            ggml_backend_free(pcie.gpu_backend);
        }
        ggml_backend_free(cpu_be);
        ggml_quantize_free();
        return 0;
    }

    std::vector<bench_result_cpu> all_results;
    bench_timer overall;
    overall.start();

    run_matmul_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);
    run_moe_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);
    run_attention_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);

    if (has_gpu && !all_results.empty()) {
        double sum_s = 0.0, sum_c = 0.0;
        for (const auto & r : all_results) { sum_s += r.standalone_gflops; sum_c += r.concurrent_gflops; }
        cr.cpu_eff = (sum_s > 0) ? 100.0 * sum_c / sum_s : 100.0;
        // PCIe rate over every concurrent phase: bytes the stress loop moved while the op tables ran
        const uint64_t ns = pcie.stress_ns.load();
        cr.pcie_concurrent = ns > 0 ? (double)pcie.stress_bytes.load() / ((double)ns / 1e9) / 1e9 : 0.0;
    }

    printf("\nTotal time: %.1f s, %zu benchmarks\n", overall.stop(), all_results.size());

    save_results_cpu(output_path, all_results, batch_sizes, threads, cr);

    if (has_gpu) {
        if (pcie.ctx) ggml_free(pcie.ctx);
        if (pcie.host_buf) ggml_backend_buffer_free(pcie.host_buf);
        if (pcie.dev_buf) ggml_backend_buffer_free(pcie.dev_buf);
        ggml_backend_free(pcie.gpu_backend);
    }
    ggml_backend_free(cpu_be);
    ggml_quantize_free();
    return 0;
}
