#include "llama-benchmark.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

static std::string llama_benchmark_cpu_brand() {
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
        if (*p == ' ' && (s.empty() || s.back() == ' ')) {
            continue;
        }
        s += *p;
    }
    while (!s.empty() && s.back() == ' ') {
        s.pop_back();
    }
    return s.empty() ? std::string("unknown") : s;
#else
    return "unknown";
#endif
}

llama_benchmark_stats::machine_t llama_benchmark_stats::machine_current(ggml_backend_dev_t gpu_dev, int n_threads) {
    machine_t m;
    m.cpu     = llama_benchmark_cpu_brand();
    m.threads = n_threads;
    m.schema  = 2;
#if defined(_WIN32)
    m.os = "windows";
#elif defined(__APPLE__)
    m.os = "macos";
#elif defined(__linux__)
    m.os = "linux";
#else
    m.os = "unknown";
#endif
    m.gpu = "none";
    if (gpu_dev != nullptr) {
        const char * d = ggml_backend_dev_description(gpu_dev);
        if (d != nullptr) {
            m.gpu = d;
        }
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(gpu_dev, &fr, &tot);
        m.vram_mib = tot >> 20;
    }
    return m;
}

uint64_t llama_benchmark_stats::machine_hash(const machine_t & m) {
    const std::string key = m.gpu + "|" + m.cpu + "|" + m.os;
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : key) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h == 0 ? 1 : h;   // 0 means "unknown" in the registry
}

std::string llama_benchmark_stats::machine_mismatch(const machine_t & profile, const machine_t & current) {
    std::string why;
    if (profile.gpu != current.gpu) {
        why += "gpu \"" + profile.gpu + "\" vs \"" + current.gpu + "\"; ";
    }
    if (profile.cpu != current.cpu) {
        why += "cpu \"" + profile.cpu + "\" vs \"" + current.cpu + "\"; ";
    }
    if (profile.os != current.os) {
        why += "os " + profile.os + " vs " + current.os + "; ";
    }
    if (profile.threads != current.threads) {
        why += "threads " + std::to_string(profile.threads) + " vs " + std::to_string(current.threads) + "; ";
    }
    if (!why.empty()) {
        why = "profile measured on another machine or configuration: " + why.substr(0, why.size() - 2);
    }
    return why;
}

llama_op_metrics llama_op_metrics_compute(const ggml_tensor * node, double token_scale) {
    llama_op_metrics m = {};

    // activation-sized quantities follow the step's token count, weights do not
    const double ts = (token_scale > 0.0 && token_scale < 1.0) ? token_scale : 1.0;
    auto tokens = [ts](int64_t n) { return std::max<int64_t>(1, (int64_t) std::llround((double) n * ts)); };
    auto act    = [ts](const ggml_tensor * t) { return t ? (double) ggml_nbytes(t) * ts : 0.0; };

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            m.N    = node->ne[0];
            m.M    = tokens(node->ne[1]);
            m.K    = node->src[0]->ne[0];
            m.ops  = 2.0 * m.N * m.K * m.M;
            m.bytes = (double) ggml_nbytes(node->src[0]) + act(node->src[1]) + act(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_MUL_MAT_ID: {
            m.N              = node->ne[0];
            m.n_experts_used = node->ne[1];
            m.M              = tokens(node->ne[2]);
            m.K              = node->src[0]->ne[0];
            const int64_t total_experts = node->src[0]->ne[2];
            m.ops  = 2.0 * m.N * m.K * m.M * m.n_experts_used;
            // a memory-bound step streams the distinct experts the M tokens route to,
            // expected total * (1 - (1 - used/total)^M), not used/total per token
            const double p_miss   = 1.0 - (double) m.n_experts_used / (double) total_experts;
            const double distinct = (double) total_experts * (1.0 - std::pow(p_miss, (double) m.M));
            m.bytes = (double) ggml_nbytes(node->src[0]) * distinct / (double) total_experts
                    + act(node->src[1]) + act(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            m.head_dim  = node->ne[0];
            m.n_q_heads = node->ne[1];
            m.n_tokens  = tokens(node->ne[2]);
            m.ctx_len   = node->src[1]->ne[1];
            m.n_kv_heads = node->src[1]->ne[2];
            m.ops  = 2.0 * m.n_tokens * m.head_dim * m.ctx_len * m.n_q_heads * 2.0;
            m.bytes = act(node->src[0]) + ggml_nbytes(node->src[1])
                    + ggml_nbytes(node->src[2]) + act(node);
            break;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            m.n_elements = tokens(ggml_nelements(node));
            m.ops  = 6.0 * m.n_elements;
            m.bytes = act(node->src[0]) + act(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_RMS_NORM: {
            m.n_elements = tokens(ggml_nelements(node));
            m.ops  = 3.0 * m.n_elements;
            m.bytes = act(node->src[0]) + act(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_GLU: {
            m.n_elements = tokens(ggml_nelements(node));
            m.ops  = 2.0 * m.n_elements;
            m.bytes = act(node->src[0]) + act(node->src[1]) + act(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV: {
            m.n_elements = tokens(ggml_nelements(node));
            m.ops  = m.n_elements;
            m.bytes = act(node->src[0]) + act(node->src[1]) + act(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_GET_ROWS: {
            m.n_elements = tokens(ggml_nelements(node));
            m.bytes = act(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_SET_ROWS: {
            // the node is a view of the whole destination (the cache); the work is the rows written
            m.n_elements = tokens(ggml_nelements(node->src[0]));
            m.bytes = 2.0 * act(node->src[0]);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_CPY: {
            m.n_elements = tokens(ggml_nelements(node));
            m.bytes = act(node->src[0]) + act(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NONE:
            break;
        default: {
            m.n_elements = tokens(ggml_nelements(node));
            m.bytes = act(node) + act(node->src[0]);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
    }
    return m;
}

std::string llama_benchmark_predictor::make_key(
        const std::string & op, const std::string & quant,
        int64_t N, int64_t K, int64_t batch) {
    return op + "|" + quant + "|" + std::to_string(N) + "|"
             + std::to_string(K) + "|" + std::to_string(batch);
}

std::string llama_benchmark_predictor::make_attn_key(
        int64_t ctx_len, int64_t n_kv_heads, int64_t n_tokens) {
    return "FLASH_ATTN|" + std::to_string(ctx_len) + "|"
             + std::to_string(n_kv_heads) + "|" + std::to_string(n_tokens);
}

std::string llama_benchmark_predictor::make_elem_key(
        const std::string & op, int64_t n_elements) {
    return op + "|" + std::to_string(n_elements);
}

static void build_entry_map(
        std::unordered_map<std::string, const llama_benchmark_entry *> & map,
        const std::vector<llama_benchmark_entry> & entries) {
    map.clear();
    map.reserve(entries.size());
    for (const auto & e : entries) {
        std::string key;
        if (e.op_name == "MUL_MAT" || e.op_name == "MUL_MAT_ID") {
            key = llama_benchmark_predictor::make_key(e.op_name, e.quant, e.N, e.K, e.B);
        } else if (e.op_name.compare(0, 10, "FLASH_ATTN") == 0) {
            // n_heads stores n_kv_heads for attention benchmarks (written by profiler)
            key = llama_benchmark_predictor::make_attn_key(e.ctx_len, e.n_heads, e.n_tokens);
        } else {
            key = llama_benchmark_predictor::make_elem_key(e.op_name, e.n_elements);
        }
        map[key] = &e;
    }
}

void llama_benchmark_predictor::build_maps() {
    build_entry_map(cpu_map, cpu_entries);
    build_entry_map(gpu_map, gpu_entries);
}

const char * llama_benchmark_profile_path(bool gpu) {
    const char * env = getenv(gpu ? "PSHARD_GPU_PROFILE" : "PSHARD_CPU_PROFILE");
    return env ? env : (gpu ? "gpu_profile.txt" : "cpu_profile.txt");
}

uint64_t llama_benchmark_profile_hash() {
    uint64_t h = 0xcbf29ce484222325ULL;
    bool any = false;
    for (bool gpu : { false, true }) {
        const char * path = llama_benchmark_profile_path(gpu);
        FILE * f = fopen(path, "rb");
        if (!f) {
            LLAMA_LOG_WARN("%s: profile %s is not readable here - a plan priced from it will not match\n", __func__, path);
            continue;
        }
        any = true;
        unsigned char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            for (size_t i = 0; i < n; i++) {
                h ^= buf[i];
                h *= 0x100000001b3ULL;
            }
        }
        fclose(f);
    }
    return any ? h : 0;
}

bool llama_benchmark_predictor::load_cpu(const char * filepath, int n_threads) {
    cpu_entries.clear();
    cpu_map.clear();

    FILE * f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not open CPU benchmark file: %s\n", __func__, filepath);
        return false;
    }

    double max_dram_bw = 0.0;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            int tc = 0;
            double dram_bw = 0.0, pcie_standalone = 0.0, pcie_concurrent = 0.0, cpu_eff = 0.0;

            if (sscanf(line,
                       "#   Threads=%d: DRAM_BW=%lf GB/s, PCIe_Standalone=%lf GB/s,"
                       " PCIe_Concurrent=%lf GB/s (CPU_Eff=%lf%%)",
                       &tc, &dram_bw, &pcie_standalone, &pcie_concurrent, &cpu_eff) == 5) {
                if (dram_bw > max_dram_bw) {
                    max_dram_bw = dram_bw;
                }
                if (tc == n_threads) {
                    stats.peak_system_bw = dram_bw;
                    stats.peak_pcie_bw   = pcie_standalone;
                    stats.eff_system_bw  = dram_bw * (cpu_eff / 100.0);
                    stats.eff_pcie_bw    = std::min(pcie_concurrent, dram_bw);
                }
            } else if (sscanf(line, "#   Threads=%d: DRAM_BW=%lf GB/s", &tc, &dram_bw) == 2) {
                if (dram_bw > max_dram_bw) {
                    max_dram_bw = dram_bw;
                }
                if (tc == n_threads) {
                    stats.peak_system_bw = dram_bw;
                }
            } else {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                if (sscanf(line, "#   PCIe_Sliced: 0.5MB=%lf 2MB=%lf 8MB=%lf 32MB=%lf GB/s",
                           &s0, &s1, &s2, &s3) == 4) {
                    stats.sliced_bw[0] = s0;
                    stats.sliced_bw[1] = s1;
                    stats.sliced_bw[2] = s2;
                    stats.sliced_bw[3] = s3;
                }
                double pc = 0.0;
                if (sscanf(line, "#   Host_Pin_Ceiling: %lf GB", &pc) == 1) {
                    stats.host_pin_ceiling_gb = pc;
                }
                // schema 2 lines: kernel-copy upload rates, pool latencies and the machine fingerprint
                if (sscanf(line, "#   PCIe_Sliced_Kernel: 0.5MB=%lf 2MB=%lf 8MB=%lf 32MB=%lf GB/s", &s0, &s1, &s2, &s3) == 4) {
                    stats.sliced_kernel_bw[0] = s0; stats.sliced_kernel_bw[1] = s1;
                    stats.sliced_kernel_bw[2] = s2; stats.sliced_kernel_bw[3] = s3;
                }
                if (sscanf(line, "#   PCIe_Segs_Kernel: 0.5MB=%lf 2MB=%lf 8MB=%lf 32MB=%lf GB/s", &s0, &s1, &s2, &s3) == 4) {
                    stats.segs_kernel_bw[0] = s0; stats.segs_kernel_bw[1] = s1;
                    stats.segs_kernel_bw[2] = s2; stats.segs_kernel_bw[3] = s3;
                }
                if (sscanf(line, "#   PCIe_Segs_Kernel_Idle: 0.5MB=%lf 2MB=%lf 8MB=%lf 32MB=%lf GB/s", &s0, &s1, &s2, &s3) == 4) {
                    stats.segs_kernel_idle_bw[0] = s0; stats.segs_kernel_idle_bw[1] = s1;
                    stats.segs_kernel_idle_bw[2] = s2; stats.segs_kernel_idle_bw[3] = s3;
                }
                double v = 0.0;
                if (sscanf(line, "#   PCIe_Staged: %lf GB/s", &v) == 1)   { stats.staged_bw          = v; }
                if (sscanf(line, "#   Kernel_Copy_Cap_MB: %lf", &v) == 1) { stats.kernel_copy_cap_mb = v; }
                if (sscanf(line, "#   Engine_Switch_us: %lf", &v) == 1)   { stats.engine_switch_us   = v; }
                if (sscanf(line, "#   Pool_Serve_us: %lf", &v) == 1)      { stats.pool_serve_us      = v; }
                if (sscanf(line, "#   Pool_Split_us: %lf", &v) == 1)      { stats.pool_split_us      = v; }
                char gpu[160] = { 0 }, cpu[160] = { 0 }, os[32] = { 0 };
                unsigned long long vram = 0;
                int mt = 0, schema = 0;
                if (sscanf(line, "#   Machine: gpu=\"%159[^\"]\" vram_mib=%llu cpu=\"%159[^\"]\" threads=%d os=%31s schema=%d",
                           gpu, &vram, cpu, &mt, os, &schema) == 6) {
                    stats.machine.gpu      = gpu;
                    stats.machine.cpu      = cpu;
                    stats.machine.os       = os;
                    stats.machine.vram_mib = (size_t) vram;
                    stats.machine.threads  = mt;
                    stats.machine.schema   = schema;
                }
            }
            continue;
        }

        if (line[0] == '\n') continue;

        llama_benchmark_entry e;
        char op_name[64], quant[16];

        int parsed = sscanf(line,
                "%63s %15s %d %lf %lf %lf %lf %lf %lf"
                " %lld %lld %lld %lld %lld %lld %lld %lld",
                op_name, quant, &e.threads,
                &e.ai, &e.bw_gb_s, &e.peak_gflops, &e.ridge,
                &e.eff_gflops, &e.eff_pcie_bw,
                &e.N, &e.K, &e.B, &e.n_tokens, &e.ctx_len,
                &e.n_heads, &e.head_dim, &e.n_elements);

        if (parsed >= 9 && e.threads == n_threads) {
            e.op_name = op_name;
            e.quant   = quant;
            cpu_entries.push_back(std::move(e));
        }
    }
    fclose(f);

    // PCIe can't exceed system DRAM BW
    if (max_dram_bw > 0.0) {
        stats.peak_pcie_bw = std::min(stats.peak_pcie_bw, max_dram_bw);
    } else {
        stats.peak_pcie_bw = std::min(stats.peak_pcie_bw, stats.peak_system_bw);
    }

    if (stats.eff_system_bw == 0.0) { stats.eff_system_bw = stats.peak_system_bw; }
    if (stats.eff_pcie_bw   == 0.0) { stats.eff_pcie_bw   = stats.peak_pcie_bw;   }

    // conservative compute floor for quantized matmuls with no benchmark entry:
    // the slowest CPU matmul rate in the profile (any quant, any shape)
    stats.cpu_matmul_floor_gflops = 0.0;
    for (const auto & e : cpu_entries) {
        if (e.op_name != "MUL_MAT" && e.op_name != "MUL_MAT_ID") continue;
        if (e.peak_gflops <= 0.0) continue;
        // only compute-bound entries: a memory-bound one (small batch, or a batch that streams
        // a whole expert stack) reports bytes/time as GFLOPS, an artifact of bandwidth, and a
        // floor taken from it would charge every expert its bandwidth time twice
        if (e.B < 32 || e.ai < e.ridge) continue;
        if (stats.cpu_matmul_floor_gflops == 0.0 || e.peak_gflops < stats.cpu_matmul_floor_gflops) {
            stats.cpu_matmul_floor_gflops = e.peak_gflops;
        }
    }

    build_maps();

    LLAMA_LOG_INFO("%s: loaded %zu CPU entries for %d threads"
                   " (peak: DRAM=%.1f GB/s, PCIe=%.1f GB/s"
                   " | concurrent: DRAM=%.1f GB/s, PCIe=%.1f GB/s)\n",
                   __func__, cpu_entries.size(), n_threads,
                   stats.peak_system_bw, stats.peak_pcie_bw,
                   stats.eff_system_bw, stats.eff_pcie_bw);
    if (stats.machine.schema >= 2) {
        LLAMA_LOG_INFO("%s: profile schema %d measured on gpu=\"%s\" (%zu MiB) cpu=\"%s\" os=%s threads=%d\n",
                       __func__, stats.machine.schema, stats.machine.gpu.c_str(), stats.machine.vram_mib,
                       stats.machine.cpu.c_str(), stats.machine.os.c_str(), stats.machine.threads);
        LLAMA_LOG_INFO("%s: kernel-copy measurements: sliced kernel 2MB=%.1f GB/s, segment kernel 2MB=%.1f GB/s loaded / %.1f idle,"
                       " staged %.1f GB/s, kernel-copy cap %.0f MB, engine switch %.1f us, pool serve %.1f us, split %.1f us\n",
                       __func__, stats.sliced_kernel_bw[1], stats.segs_kernel_bw[1], stats.segs_kernel_idle_bw[1], stats.staged_bw,
                       stats.kernel_copy_cap_mb, stats.engine_switch_us, stats.pool_serve_us, stats.pool_split_us);
    }

    return !cpu_entries.empty();
}

bool llama_benchmark_predictor::load_gpu(const char * filepath) {
    gpu_entries.clear();
    gpu_map.clear();

    FILE * f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not open GPU benchmark file: %s\n", __func__, filepath);
        return false;
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            const char * bw_str = strstr(line, "GPU_Memory_BW=");
            const char * cp_str = strstr(line, "GPU_Peak_Compute=");
            if (bw_str && cp_str) {
                double mem_bw = 0.0, compute = 0.0;
                sscanf(bw_str, "GPU_Memory_BW=%lf", &mem_bw);
                sscanf(cp_str, "GPU_Peak_Compute=%lf", &compute);
                stats.peak_gpu_mem_bw  = mem_bw;
                stats.peak_gpu_compute = compute;
            }
            continue;
        }

        if (line[0] == '\n') continue;

        llama_benchmark_entry e;
        char op_name[64], quant[16];

        int parsed = sscanf(line,
                "%63s %15s %lf %lf %lf %lf"
                " %lld %lld %lld %lld %lld %lld %lld %lld",
                op_name, quant,
                &e.ai, &e.bw_gb_s, &e.peak_gflops, &e.ridge,
                &e.N, &e.K, &e.B, &e.n_tokens, &e.ctx_len,
                &e.n_heads, &e.head_dim, &e.n_elements);

        if (parsed == 14) {
            e.op_name     = op_name;
            e.quant       = quant;
            e.threads     = -1;
            e.eff_gflops  = e.peak_gflops;
            e.eff_pcie_bw = 0.0;
            gpu_entries.push_back(std::move(e));
        }
    }
    fclose(f);

    build_maps();

    LLAMA_LOG_INFO("%s: loaded %zu GPU entries (mem_bw=%.1f GB/s, compute=%.1f GFLOP/s)\n",
                   __func__, gpu_entries.size(),
                   stats.peak_gpu_mem_bw, stats.peak_gpu_compute);

    return !gpu_entries.empty();
}

std::string llama_benchmark_predictor::make_timing_key(
        bool is_gpu, bool async_copy, const char * op_name,
        const llama_op_metrics & m, int32_t batch_size) {
    // mode prefix: GPU vs CPU with/without concurrent PCIe (async_copy uses eff_* bandwidths)
    std::string key = is_gpu ? "GPU|" : (async_copy ? "CPU_ASYNC|" : "CPU_SYNC|");
    key += op_name;
    key += "|";

    if (m.quant_type) {
        key += m.quant_type;
        key += "|";
    }

    if (strstr(op_name, "MUL_MAT_ID")) {
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.M) + "|"
             + std::to_string(m.n_experts_used);
    } else if (strstr(op_name, "MUL_MAT")) {
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.M);
    } else if (strstr(op_name, "FLASH_ATTN")) {
        key += std::to_string(m.head_dim) + "|" + std::to_string(m.n_q_heads) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.n_tokens) + "|"
             + std::to_string(m.ctx_len) + "|" + std::to_string(m.n_kv_heads);
    } else {
        key += std::to_string(m.n_elements);
    }

    return key;
}

llama_split_timing llama_benchmark_predictor::predict_split(
        struct ggml_tensor ** nodes, int n_nodes,
        bool is_gpu, int32_t batch_size, bool async_copy,
        timing_cache_t * timing_cache, double token_scale) const {

    llama_split_timing result = {};
    timing_cache_t local_timing_cache;
    timing_cache_t & cache = timing_cache ? *timing_cache : local_timing_cache;

    const auto & map     = is_gpu ? gpu_map : cpu_map;
    const auto & entries = is_gpu ? gpu_entries : cpu_entries;

    double pcie_sum = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = nodes[i];
        llama_op_metrics m = llama_op_metrics_compute(node, token_scale);

        if (m.ops == 0.0 && m.bytes == 0.0) continue;

        const char * op_name = ggml_op_name(node->op);
        double op_time_ms  = 0.0;
        double pcie_contrib = 0.0;

        // LLAMA_BENCH_PREDICT_NODES=1: per-node pricing dump (bypasses the timing cache
        // so every node shows its matched entry, bytes, and chosen roofline branch)
        static const bool dump_nodes = getenv("LLAMA_BENCH_PREDICT_NODES") != nullptr;

        std::string tkey = make_timing_key(is_gpu, async_copy, op_name, m, batch_size);
        auto cache_it = dump_nodes ? cache.end() : cache.find(tkey);
        if (cache_it != cache.end()) {
            op_time_ms   = cache_it->second.first;
            pcie_contrib = cache_it->second.second;
            result.time_ms += op_time_ms;
            pcie_sum       += pcie_contrib;
            result.n_cache_hit++;
            result.op_count++;
            continue;
        }

        // build hash key for exact lookup
        std::string hkey;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                hkey = make_key("MUL_MAT", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_MUL_MAT_ID:
                hkey = make_key("MUL_MAT_ID", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                hkey = make_attn_key(m.ctx_len, m.n_kv_heads, batch_size);
                break;
            case GGML_OP_ROPE:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GLU:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
                hkey = make_elem_key(op_name, m.n_elements);
                break;
            default:
                break;
        }

        const llama_benchmark_entry * match = nullptr;
        bool exact = false;

        // try exact hash match
        if (!hkey.empty()) {
            auto it = map.find(hkey);
            if (it != map.end()) {
                match = it->second;
                exact = true;
                result.n_exact++;
            }
        }

        // fall back to nearest-neighbor
        if (!match) {
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    match = find_nearest(entries, "MUL_MAT", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                    break;
                case GGML_OP_MUL_MAT_ID:
                    match = find_nearest(entries, "MUL_MAT_ID", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                    break;
                case GGML_OP_FLASH_ATTN_EXT:
                    match = find_nearest(entries, "FLASH_ATTN", nullptr, 0, 0, m.ctx_len, 0, m.n_kv_heads, batch_size);
                    break;
                default:
                    match = find_nearest(entries, op_name, nullptr, 0, 0, 0, m.n_elements, -1, batch_size);
                    break;
            }
            if (match) {
                result.n_nearest++;
            }
        }

        // compute timing from match or fall back to peak BW
        const char * price_branch = "none";
        if (match) {
            // GPU always peak; CPU uses eff (concurrent) when async_copy, peak otherwise
            const double gflops = is_gpu ? match->peak_gflops
                                         : (async_copy ? match->eff_gflops : match->peak_gflops);

            if (exact) {
                if (gflops > 0.0) {
                    op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                    price_branch = "exact-comp";
                }
            } else {
                const double ai = m.bytes > 0.0 ? (m.ops / m.bytes) : 0.0;
                if (ai < match->ridge) {
                    // memory-bound op. the matched entry's observed bytes/s is a bandwidth
                    // only if that entry also ran memory-bound; a compute-bound entry's byte
                    // throughput sits far below the machine's streaming rate and would price
                    // the op far too slow, hiding plans that pin it from the search
                    double mem_bw = match->bw_gb_s;
                    if (match->ai >= match->ridge || mem_bw <= 0.0) {
                        mem_bw = is_gpu ? stats.peak_gpu_mem_bw
                                        : (async_copy ? stats.eff_system_bw : stats.peak_system_bw);
                    }
                    if (mem_bw > 0.0) {
                        op_time_ms = (m.bytes / 1e9) / mem_bw * 1000.0;
                        price_branch = "near-mem";
                    } else if (gflops > 0.0) {
                        op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                        price_branch = "near-comp";
                    }
                } else if (gflops > 0.0) {
                    op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                    price_branch = "near-comp";
                }
            }

            pcie_contrib = op_time_ms * match->eff_pcie_bw;
        } else {
            result.n_fallback++;
            if (m.bytes > 0.0) {
                const double bw = is_gpu ? stats.peak_gpu_mem_bw
                                         : (async_copy ? stats.eff_system_bw : stats.peak_system_bw);
                if (bw > 0.0) {
                    op_time_ms = (m.bytes / 1e9) / bw * 1000.0;
                    price_branch = "fall-mem";
                }
                // quantized CPU matmuls with no benchmark entry (e.g. IQ quants) are
                // dequant-compute-bound at batch and memory bandwidth alone under-charges
                // them; floor with the slowest CPU matmul rate in the profile.
                // batch only: at small row counts the matmul is memory-bound (dequant
                // streams at DRAM speed) and bytes/bw is already the right price
                if (!is_gpu && m.ops > 0.0 && m.M >= 32 && stats.cpu_matmul_floor_gflops > 0.0 &&
                        (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
                        m.quant_type != nullptr &&
                        strcmp(m.quant_type, "f32") != 0 && strcmp(m.quant_type, "f16") != 0 &&
                        strcmp(m.quant_type, "bf16") != 0) {
                    const double comp_ms = (m.ops / 1e9) / stats.cpu_matmul_floor_gflops * 1000.0;
                    if (comp_ms > op_time_ms) {
                        op_time_ms = comp_ms;
                        price_branch = "fall-comp";
                    }
                }
                if (!is_gpu && stats.eff_pcie_bw > 0.0) {
                    pcie_contrib = op_time_ms * stats.eff_pcie_bw;
                }
            }
        }

        if (dump_nodes && op_time_ms > 0.0) {
            LLAMA_LOG_DEBUG("predict_node: [%s] %-16s %-28s q=%-8s NKM=%lld/%lld/%lld ctx=%lld kvh=%lld bytes=%.2fMiB ops=%.1fMF %s ent=[%s q=%s N=%lld K=%lld B=%lld ctx=%lld bw=%.1f pk=%.1f eff=%.1f ridge=%.2f] -> %.4f ms\n",
                is_gpu ? "GPU" : "CPU", op_name, node->name, m.quant_type ? m.quant_type : "-",
                (long long)m.N, (long long)m.K, (long long)m.M, (long long)m.ctx_len, (long long)m.n_kv_heads,
                m.bytes / (1024.0*1024.0), m.ops / 1e6, price_branch,
                match ? match->op_name.c_str() : "-", match ? match->quant.c_str() : "-",
                match ? (long long)match->N : 0, match ? (long long)match->K : 0,
                match ? (long long)match->B : 0, match ? (long long)match->ctx_len : 0,
                match ? match->bw_gb_s : 0.0, match ? match->peak_gflops : 0.0,
                match ? match->eff_gflops : 0.0, match ? match->ridge : 0.0,
                op_time_ms);
        }

        result.time_ms += op_time_ms;
        pcie_sum       += pcie_contrib;
        result.op_count++;

        cache[tkey] = {op_time_ms, pcie_contrib};
    }

    // convert accumulated pcie_sum to time-weighted average BW
    if (!is_gpu) {
        result.eff_pcie_bw = (result.time_ms > 0.0) ? (pcie_sum / result.time_ms)
                                                      : stats.eff_pcie_bw;
    }

    return result;
}

static double get_bits_per_weight(const std::string & quant) {
    static const std::unordered_map<std::string, double> table = {
        {"q2_K",  2.6}, {"q3_K",  3.4}, {"q4_0",  4.5}, {"q4_1",  5.0}, {"q4_K",  4.5},
        {"q5_0",  5.5}, {"q5_1",  6.0}, {"q5_K",  5.5}, {"q6_K",  6.6},
        {"q8_0",  8.5}, {"q8_1",  9.0}, {"f16",  16.0}, {"f32",  32.0},
    };
    auto it = table.find(quant);
    return it != table.end() ? it->second : -1.0;
}

const llama_benchmark_entry * llama_benchmark_predictor::find_nearest(
        const std::vector<llama_benchmark_entry> & entries,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t ctx_len, int64_t n_elements,
        int64_t n_kv_heads, int64_t target_batch) {

    if (!op_name || entries.empty()) {
        return nullptr;
    }

    const llama_benchmark_entry * best = nullptr;
    double best_score = 1e20;

    const double target_bpw = quant ? get_bits_per_weight(quant) : -1.0;

    // FLASH_ATTN entries have suffixed names (e.g. FLASH_ATTN_MHA, FLASH_ATTN_GQA-8)
    const bool attn_query = (strncmp(op_name, "FLASH_ATTN", 10) == 0);
    const bool matmul_query = (strcmp(op_name, "MUL_MAT") == 0
                            || strcmp(op_name, "MUL_MAT_ID") == 0);

    for (const auto & b : entries) {
        if (attn_query) {
            if (b.op_name.compare(0, 10, "FLASH_ATTN") != 0) continue;
        } else {
            if (b.op_name != op_name) continue;
        }

        double dim_score   = 0.0;
        double batch_score = 0.0;
        double quant_score = 0.0;

        if (matmul_query) {
            const double n_diff = std::abs((double)b.N - N) / std::max(N, (int64_t)1);
            const double k_diff = std::abs((double)b.K - K) / std::max(K, (int64_t)1);
            dim_score = n_diff + k_diff;
        } else if (attn_query) {
            const double ctx_diff = std::abs((double)b.ctx_len - ctx_len) / std::max(ctx_len, (int64_t)1);
            double kv_diff = 0.0;
            if (n_kv_heads > 0 && b.n_heads > 0) {
                kv_diff = std::abs((double)b.n_heads - n_kv_heads) / std::max(n_kv_heads, (int64_t)1);
            }
            dim_score = ctx_diff + kv_diff * 0.5;
        } else {
            const double elem_diff = std::abs((double)b.n_elements - n_elements) / std::max(n_elements, (int64_t)1);
            dim_score = elem_diff;
        }

        // attention entries carry their token count in n_tokens, not B: without it a
        // prefill attention op matches a decode-shaped entry and its bandwidth-bound rate
        const int64_t entry_batch = attn_query ? b.n_tokens : b.B;
        if (target_batch > 0 && entry_batch > 0) {
            batch_score = std::abs((double)entry_batch - target_batch) / std::max(target_batch, (int64_t)1);
        }

        if (target_bpw > 0) {
            const double bench_bpw = get_bits_per_weight(b.quant);
            if (bench_bpw > 0) {
                quant_score = std::abs(bench_bpw - target_bpw) / target_bpw;
            } else {
                quant_score = 1.0;
            }
        } else if (quant && b.quant != quant) {
            continue;
        }

        // batch most important (memory vs compute regime), then dims, then quant
        const double score = batch_score * 1.0 + dim_score * 0.5 + quant_score * 0.3;

        if (score < best_score) {
            best_score = score;
            best       = &b;
        }
    }

    return best;
}

double llama_benchmark_predictor::predict_tps(
        ggml_backend_sched_t sched,
        int cpu_backend_id,
        uint32_t kv_size,
        int32_t batch_size,
        int32_t n_tokens_graph,
        uint32_t n_outputs,
        bool has_rs,
        breakdown * bd) const {

    const int n_splits = ggml_backend_sched_get_n_splits(sched);
    if (n_splits <= 0) return 0.0;

    // the graph may have been reserved for more tokens than the step carries
    const double token_scale = (n_tokens_graph > 0 && batch_size > 0 && batch_size < n_tokens_graph)
        ? (double) batch_size / (double) n_tokens_graph : 1.0;

    LLAMA_LOG_DEBUG("%s: n_splits=%d, bs=%d, n_tokens_graph=%d, kv_size=%u, n_outputs=%u, has_rs=%d\n",
        __func__, n_splits, batch_size, n_tokens_graph, kv_size, n_outputs, (int)has_rs);

    const double pcie_bw = stats.peak_pcie_bw;
    // weight uploads may be repriced by the planner's per-mapping page-lock model
    // (mappings past the driver's pin ceiling stage through the pinned ring at a
    // host-DRAM-bound rate); KV/RS writebacks and activations always move through
    // pinned pools at the full rate.
    const double weight_bw = stats.upload_bw > 0.0 ? stats.upload_bw : pcie_bw;
    // per-class writeback ratios, matching what the runtime actually moves:
    //   attention KV: write-cells delta sync -> batch_size/kv_size in both directions
    //   recurrent state: FULL mode, entire tensor every eval -> 1.0
    // one ratio for both would charge a hybrid model's attention KV at full-cache cost
    // per decode step and under-predict plans that pin attention
    GGML_UNUSED(has_rs);
    const double kv_ratio = (kv_size > 0) ? std::min(1.0, (double)batch_size / kv_size) : 1.0;
    double total_ms = 0.0;
    bool copy_prefetched = false;
    timing_cache_t timing_cache;

    for (int i = 0; i < n_splits; i++) {
        struct ggml_backend_sched_split_info si = {};
        if (!ggml_backend_sched_get_split_info(sched, i, &si)) continue;

        const bool is_gpu = (si.backend_id != cpu_backend_id);

        double input_copy_ms = 0.0;
        double input_copy_weight_ms = 0.0;
        double input_copy_bytes = 0.0;
        if (is_gpu && pcie_bw > 0.0) {
            // a prefetched split still pays its sliced-by-used-ids expert copies at
            // consume time (the prefetch pass skips those tensors on purpose).
            // sliced expert uploads are many small gathered transfers and run far
            // below peak PCIe, so pricing them at peak over-predicts sliced strategies;
            // price the sliced share at the profiled chunk-size rate, the contiguous rest at peak.
            const double sliced_bytes = (double)si.input_weight_sliced_bytes;
            const double rest_weight_bytes = copy_prefetched
                ? 0.0
                : std::max(0.0, (double)si.input_weight_copy_bytes - sliced_bytes);
            const double rest_wb_bytes = copy_prefetched
                ? 0.0
                : (double)si.writeback_kv_bytes * kv_ratio
                    + (double)si.writeback_rs_bytes;
            input_copy_bytes = sliced_bytes + rest_weight_bytes + rest_wb_bytes;
            // sliced expert gathers source from the same mappings: a staged mapping
            // gates them at the blended rate even when the chunk curve is faster
            const double sliced_bw = sliced_bytes > 0.0
                ? (stats.upload_bw > 0.0
                    ? std::min(stats.slice_bw((double)si.input_weight_sliced_chunk_bytes), weight_bw)
                    : stats.slice_bw((double)si.input_weight_sliced_chunk_bytes))
                : pcie_bw;
            input_copy_weight_ms = (rest_weight_bytes / 1e9 / weight_bw) * 1000.0
                                 + (sliced_bw > 0.0 ? (sliced_bytes / 1e9 / sliced_bw) * 1000.0 : 0.0);
            input_copy_ms = input_copy_weight_ms + (rest_wb_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // peek at next split to determine if async prefetch will overlap with this split
        double prefetch_bytes = 0.0;
        double prefetch_weight_bytes = 0.0;
        double prefetch_wb_bytes = 0.0;
        bool next_copy_prefetched = false;
        if (i + 1 < n_splits) {
            struct ggml_backend_sched_split_info next_si = {};
            if (ggml_backend_sched_get_split_info(sched, i + 1, &next_si) &&
                next_si.can_prefetch_weights) {
                // the prefetch pass moves non-sliceable weights + writebacks only
                prefetch_weight_bytes = (double)next_si.input_weight_prefetch_bytes;
                prefetch_wb_bytes     = (double)next_si.writeback_kv_bytes * kv_ratio
                                      + (double)next_si.writeback_rs_bytes;
                prefetch_bytes = prefetch_weight_bytes + prefetch_wb_bytes;
                next_copy_prefetched = prefetch_bytes > 0.0;
            }
        }

        const bool async_copy = (prefetch_bytes > 0.0);

        // compute cost (CPU splits use eff_gflops when async_copy due to PCIe contention)
        struct ggml_tensor ** nodes = ggml_graph_nodes(si.graph);
        int n_nodes = ggml_graph_n_nodes(si.graph);
        llama_split_timing t = predict_split(nodes, n_nodes, is_gpu, batch_size, async_copy, &timing_cache, token_scale);

        // prefetch cost: use concurrent PCIe BW for CPU splits (bus shared with DRAM),
        // peak PCIe BW for GPU splits (GPU compute doesn't contend with PCIe DMA).
        // the per-op aggregate eff_pcie_bw can sit far below the machine's concurrent
        // PCIe rate when the matched entries are unrepresentative (as in the memory-bound
        // branch) - floor it with the profiler's own concurrent value
        double prefetch_ms = 0.0;
        if (prefetch_bytes > 0.0) {
            double eff_bw = pcie_bw;
            if (!is_gpu) {
                eff_bw = std::max(t.eff_pcie_bw, stats.eff_pcie_bw);
                if (eff_bw <= 0.0) {
                    eff_bw = pcie_bw;
                }
            }
            const double pf_weight_bw = stats.upload_bw > 0.0 ? std::min(eff_bw, weight_bw) : eff_bw;
            prefetch_ms = (prefetch_weight_bytes / 1e9 / pf_weight_bw) * 1000.0
                        + (prefetch_wb_bytes / 1e9 / eff_bw) * 1000.0;
        }

        // output scaling: use the output rows in the reserved graph, then scale to
        // the runtime number of logits. This keeps the memory-probe graph as the
        // source of truth and avoids double-scaling when it was already reduced.
        // The output rows are not rounded with the tokens, so the head is re-priced
        // at its real rows whenever a token scale was applied
        if (i == n_splits - 1 && n_outputs > 0) {
            for (int j = 0; j < n_nodes; j++) {
                if (nodes[j]->name && strstr(nodes[j]->name, "result_output")) {
                    const llama_op_metrics out_m = llama_op_metrics_compute(nodes[j]);
                    const int32_t graph_outputs = (int32_t) std::max<int64_t>(1, out_m.M);
                    if (token_scale < 1.0 || graph_outputs != batch_size || n_outputs < (uint32_t) graph_outputs) {
                        llama_split_timing out_included = predict_split(&nodes[j], 1, is_gpu, batch_size, async_copy, &timing_cache, token_scale);
                        llama_split_timing out_graph    = predict_split(&nodes[j], 1, is_gpu, graph_outputs, async_copy, &timing_cache, /*token_scale=*/1.0);
                        if (out_included.time_ms > 0.0 || out_graph.time_ms > 0.0) {
                            const double scale = n_outputs < (uint32_t) graph_outputs
                                               ? (double)n_outputs / (double)graph_outputs
                                               : 1.0;
                            const double scaled_ms = out_graph.time_ms * scale;
                            t.time_ms -= out_included.time_ms;
                            t.time_ms += scaled_ms;
                            LLAMA_LOG_DEBUG("%s: out_t: included=%f graph=%f scaled=%f rows=%d->%u\n",
                                    __func__, out_included.time_ms, out_graph.time_ms, scaled_ms, graph_outputs, n_outputs);
                        }
                    }
                    break;
                }
            }
        }

        // KV/RS download cost (GPU splits, after compute, on compute stream)
        double kv_dl_ms = 0.0;
        if (is_gpu && si.writeback_bytes > 0 && pcie_bw > 0.0) {
            double dl_bytes = (double)si.writeback_kv_bytes * kv_ratio
                            + (double)si.writeback_rs_bytes;
            kv_dl_ms = (dl_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // activation copy cost (synchronous, between splits on different backends)
        double activ_copy_ms = 0.0;
        if (si.input_activ_bytes > 0 && pcie_bw > 0.0) {
            activ_copy_ms = ((double)si.input_activ_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // Prefetch is enqueued after current inputs/precompute and before graph compute,
        // so whatever remains can overlap the current split compute. CPU splits use the
        // effective PCIe BW above because DMA contends with CPU memory traffic.
        const double comp_ms = t.time_ms + kv_dl_ms;
        double split_ms = input_copy_ms + activ_copy_ms;
        if (stats.upload_staged_frac > 0.0 && stats.upload_staged_bw > 0.0 && prefetch_weight_bytes > 0.0) {
            // mixture over mapping classes (see llama_benchmark_stats::upload_staged_frac)
            double eff_bw2 = pcie_bw;
            if (!is_gpu) {
                eff_bw2 = std::max(t.eff_pcie_bw, stats.eff_pcie_bw);
                if (eff_bw2 <= 0.0) {
                    eff_bw2 = pcie_bw;
                }
            }
            const double wb_ms     = (prefetch_wb_bytes / 1e9 / eff_bw2) * 1000.0;
            const double pf_pinned = (prefetch_weight_bytes / 1e9 / std::min(eff_bw2, pcie_bw)) * 1000.0 + wb_ms;
            const double pf_staged = (prefetch_weight_bytes / 1e9 / std::min(eff_bw2, stats.upload_staged_bw)) * 1000.0 + wb_ms;
            const double f = std::min(1.0, stats.upload_staged_frac);
            split_ms += f * std::max(comp_ms, pf_staged) + (1.0 - f) * std::max(comp_ms, pf_pinned);
        } else {
            split_ms += std::max(comp_ms, prefetch_ms);
        }
        if (bd != nullptr) {
            // the max() term minus the compute it hid = exposed prefetch (weights)
            const double overlap_ms = split_ms - input_copy_ms - activ_copy_ms;
            bd->compute_ms       += t.time_ms;
            bd->weight_upload_ms += input_copy_weight_ms + std::max(0.0, overlap_ms - comp_ms);
            bd->other_ms         += (input_copy_ms - input_copy_weight_ms) + activ_copy_ms + kv_dl_ms;
        }
        total_ms += split_ms;
        copy_prefetched = next_copy_prefetched;

        double dl_bytes = is_gpu ? (double)si.writeback_kv_bytes * kv_ratio + (double)si.writeback_rs_bytes : 0.0;
        LLAMA_LOG_DEBUG("%s:   split %d/%d [%s] input_copy=%.3f (%.2f MiB) compute=%.3f kv_dl=%.3f (%.2f MiB) prefetch=%.3f (%.2f MiB) activ=%.3f (%.2f MiB) -> %.3f ms"
            " (exact=%d near=%d fall=%d cache=%d)\n",
            __func__, i, n_splits, is_gpu ? "GPU" : "CPU",
            input_copy_ms, input_copy_bytes / (1024.0 * 1024.0),
            t.time_ms,
            kv_dl_ms, dl_bytes / (1024.0 * 1024.0),
            prefetch_ms, prefetch_bytes / (1024.0 * 1024.0),
            activ_copy_ms, (double)si.input_activ_bytes / (1024.0 * 1024.0),
            split_ms,
            t.n_exact, t.n_nearest, t.n_fallback, t.n_cache_hit);
    }

    double tps = (total_ms > 0.0) ? (batch_size * 1000.0 / total_ms) : 0.0;
    LLAMA_LOG_DEBUG("%s: total=%.3f ms, kv_ratio=%.4f, pcie_bw=%.1f GB/s, weight_bw=%.1f GB/s -> %.1f tps\n",
        __func__, total_ms, kv_ratio, pcie_bw, weight_bw, tps);
    return tps;
}
