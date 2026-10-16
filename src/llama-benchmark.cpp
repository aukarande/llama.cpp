#include "llama-benchmark.h"
#include "llama-impl.h"
#include "llama-pshard-workload.h"

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

double llama_expert_distinct_share(const llama_pshard_workload * wl, double rows, double n_used, double n_expert) {
    // the workload's table holds the model's routing: an op with another expert count or top-k is not in it
    const bool same = wl != nullptr && (double) wl->share_n_expert == n_expert && (double) wl->share_n_used == n_used;
    const double m  = same ? wl->distinct_share(rows) : -1.0;
    if (m >= 0.0) {
        return m;
    }
    return 1.0 - std::pow(1.0 - std::min(1.0, n_used / n_expert), std::max(1.0, rows));
}

llama_op_metrics llama_op_metrics_compute(const ggml_tensor * node, double token_scale, const llama_pshard_workload * wl) {
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
            // a compact sliced copy holds fewer slots than the model has experts
            const int64_t total_experts = ggml_backend_sched_weight_n_expert(node->src[0]);
            m.ops  = 2.0 * m.N * m.K * m.M * m.n_experts_used;
            // a memory-bound step streams the distinct experts the M tokens route to, not used/total per token
            const double share = llama_expert_distinct_share(wl, (double) m.M, (double) m.n_experts_used, (double) total_experts);
            m.bytes = (double) node->src[0]->nb[2] * (double) total_experts * share
                    + act(node->src[1]) + act(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            const ggml_tensor * k = node->src[1];
            const ggml_tensor * v = node->src[2];
            m.head_dim  = k->ne[0];   // the K head size: profile rows are matched on it
            m.n_q_heads = node->ne[1];
            m.n_tokens  = tokens(node->ne[2]);
            m.ctx_len   = k->ne[1];
            m.n_kv_heads = k->ne[2];
            // QK^T over the K head size, then the V product over the V head size (the output's)
            m.ops  = 2.0 * m.n_tokens * m.ctx_len * m.n_q_heads * (double) (k->ne[0] + node->ne[0]);
            // a V that reads K's own rows (MLA: K's first columns; V = K) adds no bytes
            const ggml_tensor * kb = k->view_src ? k->view_src : k;
            const ggml_tensor * vb = v->view_src ? v->view_src : v;
            const bool v_in_k = kb == vb && (k->view_src ? k->view_offs : 0) == (v->view_src ? v->view_offs : 0) && k->nb[1] == v->nb[1];
            m.bytes = act(node->src[0]) + ggml_nbytes(k) + (v_in_k ? 0.0 : (double) ggml_nbytes(v)) + act(node);
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
            // no exact key: the KV head count it would carry does not fix the shape (find_nearest matches the
            // queries per KV head)
            continue;
        } else {
            key = llama_benchmark_predictor::make_elem_key(e.op_name, e.n_elements);
        }
        map[key] = &e;
    }
}

void llama_benchmark_predictor::build_maps() {
    build_entry_map(cpu_map, cpu_entries);
    build_entry_map(gpu_map, gpu_entries);
    build_entry_map(gpu_host_map, gpu_host_entries);
}

const char * llama_benchmark_profile_path(bool gpu) {
    const char * env = getenv(gpu ? "PSHARD_GPU_PROFILE" : "PSHARD_CPU_PROFILE");
    return env ? env : (gpu ? "gpu_profile.txt" : "cpu_profile.txt");
}

static const char LLAMA_BENCHMARK_SSD_PREFIX[] = "#   SSD_Read:";

static bool llama_benchmark_read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return true;
}

uint64_t llama_benchmark_profile_hash() {
    uint64_t h = 0xcbf29ce484222325ULL;
    bool any = false;
    for (bool gpu : { false, true }) {
        const char * path = llama_benchmark_profile_path(gpu);
        std::string data;
        if (!llama_benchmark_read_file(path, data)) {
            LLAMA_LOG_WARN("%s: profile %s is not readable here - a plan priced from it will not match\n", __func__, path);
            continue;
        }
        any = true;
        // every byte but the SSD_Read line: measuring it re-plans only the models the RAM expert store serves
        for (size_t b = 0; b < data.size(); ) {
            const size_t nl = data.find('\n', b);
            const size_t e  = nl == std::string::npos ? data.size() : nl + 1;
            if (data.compare(b, sizeof(LLAMA_BENCHMARK_SSD_PREFIX) - 1, LLAMA_BENCHMARK_SSD_PREFIX) != 0) {
                for (size_t i = b; i < e; i++) {
                    h ^= (unsigned char) data[i];
                    h *= 0x100000001b3ULL;
                }
            }
            b = e;
        }
    }
    return any ? h : 0;
}

std::string llama_benchmark_ssd_line(const std::string & unit) {
    std::string data;
    if (unit.empty() || !llama_benchmark_read_file(llama_benchmark_profile_path(false), data)) {
        return "";
    }
    const size_t b = data.find(std::string("\n") + LLAMA_BENCHMARK_SSD_PREFIX + " unit=" + unit + " ");
    return b == std::string::npos ? "" : data.substr(b + 1, data.find_first_of("\r\n", b + 1) - b - 1);
}

bool llama_benchmark_store::parse_ssd(const std::string & line) {
    // a rate of 0 would price the reads as free
    return sscanf(line.c_str(), "#   SSD_Read: unit=%*s sat=%lf GB/s qd1=%lf GB/s lat=%lf ms", &sat_gbps, &qd1_gbps, &lat_ms) == 3 &&
        sat_gbps > 0.0 && qd1_gbps > 0.0 && lat_ms >= 0.0;
}

double llama_benchmark_host_pin_ceiling_gb() {
    FILE * f = fopen(llama_benchmark_profile_path(false), "r");
    if (!f) {
        return 0.0;
    }
    double pc = 0.0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') {
            break;   // the header ends at the first entry
        }
        if (sscanf(line, "#   Host_Pin_Ceiling: %lf GB", &pc) == 1) {
            break;
        }
    }
    fclose(f);
    return pc;
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
                if (sscanf(line, "#   Copy_Setup_us: %lf", &v) == 1)      { stats.copy_setup_us      = v; }
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
                       " staged %.1f GB/s, kernel-copy cap %.0f MB, engine switch %.1f us, pool serve %.1f us, split %.1f us,"
                       " copy setup %.1f us\n",
                       __func__, stats.sliced_kernel_bw[1], stats.segs_kernel_bw[1], stats.segs_kernel_idle_bw[1], stats.staged_bw,
                       stats.kernel_copy_cap_mb, stats.engine_switch_us, stats.pool_serve_us, stats.pool_split_us,
                       stats.copy_setup_us);
    }

    return !cpu_entries.empty();
}

bool llama_benchmark_predictor::load_gpu(const char * filepath) {
    gpu_entries.clear();
    gpu_map.clear();
    gpu_host_entries.clear();
    gpu_host_map.clear();
    stats.host_read_bw = 0.0;

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
            // an in-place row keeps its op name without the suffix, so the lookups match it like a device row
            char * host = strstr(op_name, "@host");
            if (host != nullptr) {
                *host = '\0';
            }
            e.op_name     = op_name;
            e.quant       = quant;
            e.threads     = -1;
            e.eff_gflops  = e.peak_gflops;
            e.eff_pcie_bw = 0.0;
            if (host != nullptr) {
                if (e.ai < e.ridge) {
                    stats.host_read_bw = std::max(stats.host_read_bw, e.bw_gb_s);
                }
                gpu_host_entries.push_back(std::move(e));
            } else {
                gpu_entries.push_back(std::move(e));
            }
        }
    }
    fclose(f);

    build_maps();

    LLAMA_LOG_INFO("%s: loaded %zu GPU entries (mem_bw=%.1f GB/s, compute=%.1f GFLOP/s), %zu in place (read %.1f GB/s)\n",
                   __func__, gpu_entries.size(),
                   stats.peak_gpu_mem_bw, stats.peak_gpu_compute, gpu_host_entries.size(), stats.host_read_bw);

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

// a GPU small-op entry (elementwise, norm, routing, recurrence): timed by element count, no matmul or attention shape
static bool llama_benchmark_small_op(const llama_benchmark_entry & e) {
    return e.N == 0 && e.K == 0 && e.B == 0 && e.n_tokens == 0 && e.n_elements > 0 &&
           e.op_name.compare(0, 10, "FLASH_ATTN") != 0;
}

enum llama_benchmark_series_kind { LLAMA_SERIES_EXPERTS, LLAMA_SERIES_ATTN, LLAMA_SERIES_SIZE };

// where an entry sits in its series: rows per expert of a MUL_MAT_ID entry (B tokens routed to n_used of n_expert
// experts), tokens of an attention entry, elements of a small-op entry
static double llama_benchmark_series_x(const llama_benchmark_entry & e, llama_benchmark_series_kind kind) {
    switch (kind) {
        case LLAMA_SERIES_ATTN: return (double) e.n_tokens;
        case LLAMA_SERIES_SIZE: return (double) e.n_elements;
        default: return e.ctx_len > 0 ? (double) e.B * (double) e.n_tokens / (double) e.ctx_len : (double) e.B;
    }
}

// the rate a series is measured by: GFLOP/s, bandwidth for small ops
static double llama_benchmark_series_rate(const llama_benchmark_entry & e, llama_benchmark_series_kind kind) {
    return kind == LLAMA_SERIES_SIZE ? e.bw_gb_s : e.peak_gflops;
}

// measured rate of the matched entry's series (its op, quant and shape at every profiled batch or size) at x,
// log-linear in x between the two entries around it; 0 when x lies outside the series. x_min: the series'
// smallest x
static double llama_benchmark_series_gflops(const std::vector<llama_benchmark_entry> & entries,
        const llama_benchmark_entry & match, double x, llama_benchmark_series_kind kind, double & x_min) {
    const llama_benchmark_entry * lo = nullptr;
    const llama_benchmark_entry * hi = nullptr;
    for (const auto & e : entries) {
        bool same = e.op_name == match.op_name && e.quant == match.quant;
        switch (kind) {
            case LLAMA_SERIES_ATTN:
                same = same && e.ctx_len == match.ctx_len && e.n_heads == match.n_heads && e.head_dim == match.head_dim;
                break;
            case LLAMA_SERIES_SIZE:
                same = same && llama_benchmark_small_op(e);
                break;
            default:
                same = same && e.N == match.N && e.K == match.K && e.n_tokens == match.n_tokens && e.ctx_len == match.ctx_len;
                break;
        }
        if (!same || llama_benchmark_series_rate(e, kind) <= 0.0) {
            continue;
        }
        const double ex = llama_benchmark_series_x(e, kind);
        if (ex <= 0.0) {
            continue;
        }
        x_min = x_min > 0.0 ? std::min(x_min, ex) : ex;
        if (ex <= x && (lo == nullptr || ex > llama_benchmark_series_x(*lo, kind))) {
            lo = &e;
        }
        if (ex >= x && (hi == nullptr || ex < llama_benchmark_series_x(*hi, kind))) {
            hi = &e;
        }
    }
    if (lo == nullptr || hi == nullptr) {
        return 0.0;
    }
    const double x0 = llama_benchmark_series_x(*lo, kind);
    const double x1 = llama_benchmark_series_x(*hi, kind);
    const double r0 = llama_benchmark_series_rate(*lo, kind);
    const double r1 = llama_benchmark_series_rate(*hi, kind);
    if (x1 <= x0) {
        return r0;
    }
    const double t = std::log(x / x0) / std::log(x1 / x0);
    return std::exp(std::log(r0) + t * (std::log(r1) - std::log(r0)));
}

// bits per weight of a type (0: a removed type)
static double llama_benchmark_type_bpw(ggml_type t) {
    return ggml_blck_size(t) > 0 ? 8.0 * (double) ggml_type_size(t) / (double) ggml_blck_size(t) : 0.0;
}

// bits per weight of the type a profile row names (0: unknown)
static double llama_benchmark_quant_bpw(const std::string & quant) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        if (ggml_blck_size((ggml_type) i) > 0 && quant == ggml_type_name((ggml_type) i)) {
            return llama_benchmark_type_bpw((ggml_type) i);
        }
    }
    return 0.0;
}

// the quant of the rows nearest in bits per weight to type t (nullptr: no matmul rows)
static const char * llama_benchmark_nearest_quant(const std::vector<llama_benchmark_entry> & entries, ggml_type t) {
    const double bpw = llama_benchmark_type_bpw(t);
    const char * best   = nullptr;
    double       best_d = 1e20;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type ti = (ggml_type) i;
        if (ggml_blck_size(ti) <= 0) {
            continue;
        }
        const char * name = ggml_type_name(ti);
        const bool profiled = std::any_of(entries.begin(), entries.end(),
            [&](const llama_benchmark_entry & b) { return b.quant == name; });
        const double d = std::abs(llama_benchmark_type_bpw(ti) - bpw);
        if (profiled && d < best_d) {
            best_d = d;
            best   = name;
        }
    }
    return best;
}

llama_split_timing llama_benchmark_predictor::predict_split(
        struct ggml_tensor ** nodes, int n_nodes,
        bool is_gpu, int32_t batch_size, bool async_copy,
        timing_cache_t * timing_cache, double token_scale) const {

    llama_split_timing result = {};
    timing_cache_t local_timing_cache;
    timing_cache_t & cache = timing_cache ? *timing_cache : local_timing_cache;

    double pcie_sum = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = nodes[i];
        llama_op_metrics m = llama_op_metrics_compute(node, token_scale, workload);

        if (m.ops == 0.0 && m.bytes == 0.0) continue;

        // a GPU matmul whose weight stays in pinned host memory (the scheduler left it uncopied: it reads it in
        // place) is priced from the profile's in-place rows
        const ggml_backend_buffer_t wbuf = (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]
            ? (node->src[0]->view_src ? node->src[0]->view_src->buffer : node->src[0]->buffer) : nullptr;
        const bool in_place = is_gpu && has_host_rows() && wbuf != nullptr && ggml_backend_buffer_is_host(wbuf) &&
            ggml_backend_buffer_get_usage(wbuf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const auto & map     = in_place ? gpu_host_map : is_gpu ? gpu_map : cpu_map;
        const auto & entries = in_place ? gpu_host_entries : is_gpu ? gpu_entries : cpu_entries;
        const double gpu_bw  = in_place ? stats.host_read_bw : stats.peak_gpu_mem_bw;

        const char * op_name = ggml_op_name(node->op);
        double op_time_ms  = 0.0;
        double pcie_contrib = 0.0;

        // LLAMA_BENCH_PREDICT_NODES=1: per-node pricing dump (bypasses the timing cache
        // so every node shows its matched entry, bytes, and chosen roofline branch)
        static const bool dump_nodes = getenv("LLAMA_BENCH_PREDICT_NODES") != nullptr;

        std::string tkey = (in_place ? "H|" : "") + make_timing_key(is_gpu, async_copy, op_name, m, batch_size);
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
                    match = find_nearest(entries, "MUL_MAT", m.quant_type, m.N, m.K, 0, 0, 0.0, batch_size);
                    break;
                case GGML_OP_MUL_MAT_ID:
                    match = find_nearest(entries, "MUL_MAT_ID", m.quant_type, m.N, m.K, 0, 0, 0.0, batch_size);
                    break;
                case GGML_OP_FLASH_ATTN_EXT:
                    match = find_nearest(entries, "FLASH_ATTN", nullptr, 0, m.head_dim, m.ctx_len, 0,
                        m.bytes > 0.0 ? m.ops / m.bytes : 0.0, batch_size);
                    break;
                default:
                    match = find_nearest(entries, op_name, nullptr, 0, 0, 0, m.n_elements, 0.0, batch_size);
                    break;
            }
            if (!match && in_place && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
                // a quant the in-place rows lack (IQ): the rows of the profiled quant nearest in bits per weight
                const char * q = llama_benchmark_nearest_quant(entries, node->src[0]->type);
                if (q != nullptr) {
                    match = find_nearest(entries, op_name, q, m.N, m.K, 0, 0, 0.0, batch_size);
                }
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

            if (is_gpu && llama_benchmark_small_op(*match)) {
                // a small op moves its bytes at the rate its op measured near its size, launch share included;
                // past the profiled sizes, the nearest one's
                double x_min = 0.0;
                double bw = llama_benchmark_series_gflops(entries, *match, (double) m.n_elements, LLAMA_SERIES_SIZE, x_min);
                if (bw <= 0.0) {
                    bw = match->bw_gb_s;
                }
                if (bw > 0.0) {
                    op_time_ms = (m.bytes / 1e9) / bw * 1000.0;
                    price_branch = "small-op";
                }
            } else if (exact) {
                if (gflops > 0.0) {
                    op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                    price_branch = "exact-comp";
                }
            } else if (in_place && node->op == GGML_OP_MUL_MAT_ID) {
                // an in-place expert matmul runs at its series' measured rate at its rows per expert: the rows count
                // one token's experts as their bytes, so their bandwidth column holds at one token only
                const double n_expert = (double) ggml_backend_sched_weight_n_expert(node->src[0]);
                const double x = n_expert > 0.0 ? (double) m.M * (double) m.n_experts_used / n_expert : 0.0;
                double x_min = 0.0;
                double rate = x > 0.0 ? llama_benchmark_series_gflops(entries, *match, x, LLAMA_SERIES_EXPERTS, x_min) : 0.0;
                if (rate <= 0.0) {
                    rate = gflops;
                }
                if (rate > 0.0) {
                    op_time_ms = (m.ops / 1e9) / rate * 1000.0;
                    // rows of another quant read other bytes: the read scales with the weight's bits
                    const double bpw_w = llama_benchmark_type_bpw(node->src[0]->type);
                    const double bpw_m = llama_benchmark_quant_bpw(match->quant);
                    if (bpw_w > 0.0 && bpw_m > 0.0) {
                        op_time_ms *= bpw_w / bpw_m;
                    }
                    price_branch = "in-place";
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
                        mem_bw = is_gpu ? gpu_bw
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

                // between two profiled batches a GPU expert matmul or attention runs at a rate between theirs,
                // not at the nearer one's: the nearer row prices a small prompt's expert matmuls at the
                // bandwidth of a far larger batch, and its attention at the decode rate. Expert matmuls go by
                // rows per expert and only rise to this rate (below it they stay memory-bound); attention
                // takes it inside the series, the decode row itself keeps its price
                const bool attn = node->op == GGML_OP_FLASH_ATTN_EXT;
                if (is_gpu && m.ops > 0.0 && (attn || node->op == GGML_OP_MUL_MAT_ID)) {
                    const double n_expert = attn ? 0.0 : (double) ggml_backend_sched_weight_n_expert(node->src[0]);
                    const double x = attn ? (double) m.n_tokens
                                          : (n_expert > 0.0 ? (double) m.M * (double) m.n_experts_used / n_expert : 0.0);
                    double x_min = 0.0;
                    const double rate = x > 0.0 ? llama_benchmark_series_gflops(entries, *match, x,
                        attn ? LLAMA_SERIES_ATTN : LLAMA_SERIES_EXPERTS, x_min) : 0.0;
                    if (rate > 0.0) {
                        const double series_ms = (m.ops / 1e9) / rate * 1000.0;
                        if (attn ? x > x_min : series_ms > op_time_ms) {
                            op_time_ms   = series_ms;
                            price_branch = "series";
                        }
                    }
                }
            }

            pcie_contrib = op_time_ms * match->eff_pcie_bw;
        } else {
            result.n_fallback++;
            if (m.bytes > 0.0) {
                const double bw = is_gpu ? gpu_bw
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
        double ai, int64_t target_batch) {

    if (!op_name || entries.empty()) {
        return nullptr;
    }

    const llama_benchmark_entry * best = nullptr;
    double best_score = 1e20;
    double best_shape = 1e20;

    const double target_bpw = quant ? get_bits_per_weight(quant) : -1.0;

    // FLASH_ATTN entries have suffixed names (e.g. FLASH_ATTN_MHA, FLASH_ATTN_GQA-8)
    const bool attn_query = (strncmp(op_name, "FLASH_ATTN", 10) == 0);
    const bool matmul_query = (strcmp(op_name, "MUL_MAT") == 0
                            || strcmp(op_name, "MUL_MAT_ID") == 0);

    // attention: K is the head size, which picks the kernel; only rows of the nearest profiled head size compete
    int64_t attn_head_dim = 0;
    if (attn_query && K > 0) {
        double best_hd = 1e20;
        for (const auto & b : entries) {
            if (b.op_name.compare(0, 10, "FLASH_ATTN") != 0 || b.head_dim <= 0) continue;
            const double d = std::abs((double) b.head_dim - (double) K) / (double) K;
            if (d < best_hd) {
                best_hd = d;
                attn_head_dim = b.head_dim;
            }
        }
    }

    for (const auto & b : entries) {
        if (attn_query) {
            if (b.op_name.compare(0, 10, "FLASH_ATTN") != 0) continue;
            if (attn_head_dim > 0 && b.head_dim != attn_head_dim) continue;
        } else {
            if (b.op_name != op_name) continue;
        }

        double dim_score   = 0.0;
        double batch_score = 0.0;
        double quant_score = 0.0;
        double shape_score = 0.0;

        if (matmul_query) {
            // a matmul's rate follows the bytes it streams: rows are matched on matrix size in doublings (a row
            // 100x too small is not as close as one 2x too small), the shape only breaks ties
            const double n_e = (double) std::max<int64_t>(b.N, 1), k_e = (double) std::max<int64_t>(b.K, 1);
            const double n_q = (double) std::max<int64_t>(N, 1),   k_q = (double) std::max<int64_t>(K, 1);
            dim_score   = std::abs(std::log2((n_e * k_e) / (n_q * k_q)));
            shape_score = std::abs(std::log2((n_e / k_e) / (n_q / k_q)));
        } else if (attn_query) {
            const double ctx_diff = std::abs((double)b.ctx_len - ctx_len) / std::max(ctx_len, (int64_t)1);
            // queries per KV head, as arithmetic intensity: 16 queries over 2 KV heads run like 32 over 4, not like
            // 32 over 1
            double ai_diff = 0.0;
            if (ai > 0.0 && b.ai > 0.0) {
                ai_diff = std::abs(std::log2(b.ai / ai));
            }
            dim_score = ctx_diff + ai_diff * 0.5;
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

        if (score < best_score || (score == best_score && shape_score < best_shape)) {
            best_score = score;
            best_shape = shape_score;
            best       = &b;
        }
    }

    return best;
}

// a matmul of a layer's FFN proper: the dense up/gate/down or the routed experts (not the router, not a shared
// expert). Weight copies keep the weight's name inside their own
static bool llama_benchmark_node_is_ffn(const ggml_tensor * node) {
    if ((node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) || node->src[0] == nullptr) {
        return false;
    }
    const char * name = node->src[0]->name;
    return strstr(name, "exps") != nullptr || strstr(name, ".ffn_up.") != nullptr ||
           strstr(name, ".ffn_gate.") != nullptr || strstr(name, ".ffn_down.") != nullptr;
}

// the consume path copies the runs of consecutive used experts as one segment-batch launch when the device
// offers it, else run by run on the copy engine; staged mappings cannot take the kernel path. The copy waits for
// ids its own layer computed, so no CPU expert chain runs beside it: the CPU-idle kernel curve prices it
double llama_benchmark_predictor::sliced_expert_copy_ms(double bytes, double share, double n_expert, double expert_size,
        bool segments) const {
    const double e  = n_expert;
    const double es = expert_size;
    // runs of consecutive used experts are 1/(1 - share) experts long on average
    const double run = share < 1.0 ? std::min(share * e, 1.0 / (1.0 - share)) : e;
    const double chunk_run = es * std::max(1.0, run);
    double chunk  = chunk_run;
    double f_long = 0.0;   // share of the bytes in runs above the kernel-copy cap
    double bw     = 0.0;
    // a cap below one expert refuses every segment batch: all runs then go by DMA
    if (segments && (stats.kernel_copy_cap_mb < 0.0 || stats.kernel_copy_cap_mb * 1024.0 * 1024.0 >= es)) {
        // the device refuses a segment batch holding a run above the kernel-copy cap: the runs then go one by one,
        // the long ones by DMA. With experts used independently at rate p, the used experts in runs longer than
        // c experts are p^c (1 + c (1 - p)) of them
        double cap = INFINITY;
        if (stats.kernel_copy_cap_mb >= 0.0) {
            cap = stats.kernel_copy_cap_mb * 1024.0 * 1024.0;
            const double c = std::floor(cap / es);
            if (c < e) {
                f_long = share < 1.0 ? std::pow(share, c) * (1.0 + c * (1.0 - share)) : 1.0;
                chunk  = es * std::max(1.0, std::min(run, c));
            }
        }
        // curve points above the cap are never reached: the last one below it holds
        double top = 0.0;
        for (double mb : llama_benchmark_stats::sliced_bw_chunk_mb) {
            if (mb * 1024.0 * 1024.0 <= cap) {
                top = mb * 1024.0 * 1024.0;
            }
        }
        if (top > 0.0) {
            chunk = std::min(chunk, top);
        }
        bw = stats.slice_bw_kernel(chunk, /*cpu_loaded =*/ false);
    }
    if (bw <= 0.0) {
        // no segment path: every run by DMA
        f_long = 0.0;
        bw     = stats.slice_bw(chunk_run);
    }
    // runs above the cap are single DMAs of a full-copy size
    double ms = bytes / 1e6 * ((1.0 - f_long) / bw + (f_long > 0.0 ? f_long / stats.peak_pcie_bw : 0.0));
    if (stats.upload_staged_frac > 0.0 && stats.upload_staged_bw > 0.0) {
        // mixture over mapping classes (see llama_benchmark_stats::upload_staged_frac)
        const double f = std::min(1.0, stats.upload_staged_frac);
        ms = f * bytes / 1e6 / stats.upload_staged_bw + (1.0 - f) * ms;
    }
    return ms;
}

double llama_benchmark_predictor::store_misses(double rows, double n_used) const {
    if (store == nullptr || store->n_ram >= store->n_expert) {
        return 0.0;
    }
    const double e = (double) store->n_expert;
    if (workload != nullptr && workload->share_n_expert == store->n_expert) {
        // one row: the routes outside the resident experts at the fitted (split-half) Zipf exponent; the rows model may
        // have set the histogram aside for uniform routing
        if (rows <= 1.0 && workload->zipf_alpha >= 0.0) {
            return n_used * (1.0 - llama_pshard_workload::zipf_h(workload->zipf_alpha, store->n_ram, store->n_expert));
        }
        const double m = workload->pool_misses(rows, store->n_ram);
        if (m >= 0.0) {
            return m;
        }
    }
    return e * llama_expert_distinct_share(nullptr, rows, n_used, e) * (e - store->n_ram) / e;
}

double llama_benchmark_predictor::store_read_ms(double m, double e_bytes, int n_waits, int n_reads) const {
    if (store == nullptr || m <= 0.0 || e_bytes <= 0.0) {
        return 0.0;
    }
    // e_bytes: one expert of the tensors read, one read per tensor
    n_reads = n_reads > 0 ? n_reads : std::max(1, n_waits);
    if (!store->overlap) {
        // reads one after another: each its latency and its bytes at the rate of one at a time
        return m * (e_bytes / 1e6 / store->qd1_gbps + n_reads * store->lat_ms);
    }
    // the bytes at the rate with many reads in flight, and per wait what a lone read takes beyond that rate (a wait
    // with m expected reads waits for at least one with probability 1 - e^-m): its latency and its tensor's bytes at the
    // rate of one read at a time, less the same bytes at the rate with many in flight
    const double lone_ms = store->lat_ms + e_bytes / n_reads / 1e6 * std::max(0.0, 1.0 / store->qd1_gbps - 1.0 / store->sat_gbps);
    return m * e_bytes / 1e6 / store->sat_gbps + n_waits * (1.0 - std::exp(-m)) * lone_ms;
}

double llama_benchmark_predictor::predict_tps(
        ggml_backend_sched_t sched,
        int cpu_backend_id,
        uint32_t kv_size,
        int32_t batch_size,
        int32_t n_tokens_graph,
        uint32_t n_outputs,
        bool has_rs,
        breakdown * bd,
        int32_t expert_slice_tokens,
        uint32_t n_kv_stream) const {

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
    // writebacks as the runtime moves them:
    //   by cells (writeback_kv): every cell in use goes up before the split (the graph is priced at its reserved
    //   length, so the whole cache), the cells the step wrote come back -> batch_size/kv_size
    //   in full (writeback_rs: recurrent state, transposed V, caches without per-token cells): the whole tensor
    //   every eval, both ways
    GGML_UNUSED(has_rs);
    const double kv_ratio = (kv_size > 0) ? std::min(1.0, (double)batch_size / kv_size) : 1.0;
    double total_ms = 0.0;
    timing_cache_t timing_cache;

    std::vector<ggml_backend_sched_split_info> infos(n_splits);
    std::vector<bool> have_info(n_splits);
    for (int i = 0; i < n_splits; i++) {
        have_info[i] = ggml_backend_sched_get_split_info(sched, i, &infos[i]);
    }

    // classify each split's expert tensors with the crossover at this step's rows, as the runtime scheduler will
    if (expert_slice_tokens >= 0) {
        for (int i = 0; i < n_splits; i++) {
            auto & si = infos[i];
            const size_t governed = si.input_weight_sliced_bytes + si.input_expert_bytes;
            if (!have_info[i] || governed == 0) {
                continue;
            }
            const double rows  = std::max(1.0, std::round((double) si.input_expert_n_rows * token_scale));
            const bool   slice = rows < (double) expert_slice_tokens;
            const int32_t governed_n = si.input_weight_sliced_n + si.input_expert_n;
            si.input_weight_prefetch_bytes = si.input_weight_prefetch_bytes - si.input_expert_bytes + (slice ? 0 : governed);
            si.input_weight_sliced_bytes   = slice ? governed : 0;
            si.input_expert_bytes          = slice ? 0 : governed;
            si.input_weight_prefetch_n     = si.input_weight_prefetch_n - si.input_expert_n + (slice ? 0 : governed_n);
            si.input_weight_sliced_n       = slice ? governed_n : 0;
            si.input_expert_n              = slice ? 0 : governed_n;
        }
    }

    // a cache with a stream per sequence moves only the streams the step touches, at most one per token: their cells
    // in use go up, their written cells come back, a copy per touched stream and tensor
    const double kv_stream_f = n_kv_stream > 1 ? std::min(1.0, (double) batch_size / n_kv_stream) : 1.0;
    if (kv_stream_f < 1.0) {
        for (int i = 0; i < n_splits; i++) {
            if (have_info[i] && infos[i].writeback_kv_n > 0) {
                infos[i].writeback_kv_n = std::max<int32_t>(1, (int32_t) std::lround(infos[i].writeback_kv_n * kv_stream_f));
            }
        }
    }

    // a sliced copy waits for its layer's ids on the host: readback, sync, decision, launch, and the
    // copy-engine transition of the readback
    const double ids_round_trip_ms = (std::max(0.0, stats.pool_serve_us) + std::max(0.0, stats.engine_switch_us)) / 1000.0;
    const bool   prefetch_window   = ggml_backend_sched_get_prefetch_window(sched);
    // weights and writebacks move tensor by tensor: each copy adds its setup time to its bytes
    const double copy_setup_ms     = std::max(0.0, stats.copy_setup_us) / 1000.0;

    // the scheduler's prefetch state machine (ggml_backend_sched_next_prefetch_split): one prefetch outstanding; a
    // trigger split issues the next split within the lookahead whose weights the prefetch pass moves (window mode:
    // only a split that streams nothing triggers). Expert-pool scheds decide their prefetch in a callback: the next
    // split stands in for it
    const bool pool_sched = ggml_backend_sched_has_copy_overrides(sched);
    const int  lookahead  = pool_sched ? 1 : ggml_backend_sched_get_prefetch_lookahead();
    // the RAM expert store serves the experts (an expert pool uploads its own)
    const llama_benchmark_store * st = pool_sched ? nullptr : store;
    std::vector<int>  target_of(n_splits, -1);
    std::vector<char> is_target(n_splits, 0);
    {
        auto streams = [&](int j) { return have_info[j] && infos[j].can_prefetch_weights; };
        auto movable = [&](int j) {
            const double wb = (double) infos[j].writeback_kv_bytes + (double) infos[j].writeback_rs_bytes;
            return streams(j) && (infos[j].input_weight_prefetch_bytes > 0 || (pool_sched && wb > 0.0));
        };
        int outstanding = -1;
        for (int i = 0; i < n_splits; i++) {
            if (outstanding == i) {
                outstanding = -1;
            }
            if (outstanding >= 0 || (prefetch_window && streams(i))) {
                continue;
            }
            for (int j = i + 1; j < std::min(n_splits, i + 1 + lookahead); j++) {
                if (movable(j)) {
                    target_of[i] = j;
                    is_target[j] = 1;
                    outstanding  = j;
                    break;
                }
            }
        }
    }

    // a split's weights come from one mapping class (see llama_benchmark_stats::upload_staged_frac): the prefetch is
    // timed in a pinned world and a staged world, the exposed time weighted by the staged share
    const double f_staged = stats.upload_staged_frac > 0.0 && stats.upload_staged_bw > 0.0
        ? std::min(1.0, stats.upload_staged_frac) : 0.0;
    // rates of a prefetch under a split's compute: a GPU split leaves the link to it, a CPU split shares host DRAM
    auto prefetch_rates = [&](bool gpu, double split_eff_pcie, double w_bw[2], double & wb_bw) {
        double eff = pcie_bw;
        if (!gpu) {
            eff = std::max(split_eff_pcie, stats.eff_pcie_bw);
            if (eff <= 0.0) {
                eff = pcie_bw;
            }
        }
        wb_bw = eff;
        if (f_staged > 0.0) {
            w_bw[0] = std::min(eff, pcie_bw);
            w_bw[1] = std::min(eff, stats.upload_staged_bw);
        } else {
            w_bw[0] = w_bw[1] = stats.upload_bw > 0.0 ? std::min(eff, weight_bw) : eff;
        }
    };
    double pending_w[2]  = { 0.0, 0.0 };   // bytes of the outstanding prefetch still to move: pinned, staged world
    double pending_wb[2] = { 0.0, 0.0 };
    double pending_su[2] = { 0.0, 0.0 };   // ms of its copies' setup still to run
    int    pending_to    = -1;

    for (int i = 0; i < n_splits; i++) {
        if (!have_info[i]) continue;
        const ggml_backend_sched_split_info & si = infos[i];
        const bool copy_prefetched = is_target[i] != 0;

        const bool is_gpu = (si.backend_id != cpu_backend_id);

        double input_copy_ms = 0.0;
        double input_copy_weight_ms = 0.0;
        double input_copy_bytes = 0.0;
        if (is_gpu && pcie_bw > 0.0) {
            // expert tensors the consume path slices by used ids: those the prefetch pass skips (paid even when the
            // split was prefetched) and, in a split that was not prefetched, those it would have moved in full.
            // Only the distinct experts the step's rows route to move
            const double sliced_bytes = (double) si.input_weight_sliced_bytes;
            const double expert_bytes = copy_prefetched ? 0.0 : (double) si.input_expert_bytes;
            const double rows  = std::max(1.0, std::round((double) si.input_expert_n_rows * token_scale));
            const double share = sliced_bytes + expert_bytes > 0.0
                ? llama_expert_distinct_share(workload, rows, (double) si.input_expert_n_used, (double) si.input_expert_n_expert)
                : 0.0;
            const double slice_copy_bytes = (sliced_bytes + expert_bytes) * share;
            const double rest_weight_bytes = copy_prefetched
                ? 0.0
                : std::max(0.0, (double) si.input_weight_bytes - sliced_bytes - expert_bytes);
            const double rest_wb_bytes = copy_prefetched
                ? 0.0
                : (double)si.writeback_kv_bytes * kv_stream_f + (double)si.writeback_rs_bytes;
            const int rest_weight_n = copy_prefetched ? 0 : si.input_weight_prefetch_n - si.input_expert_n;
            const int rest_wb_n     = copy_prefetched ? 0 : si.writeback_kv_n + si.writeback_rs_n;
            input_copy_bytes = slice_copy_bytes + rest_weight_bytes + rest_wb_bytes;
            input_copy_weight_ms = (rest_weight_bytes / 1e9 / weight_bw) * 1000.0 + rest_weight_n * copy_setup_ms
                                 + (slice_copy_bytes > 0.0
                                     ? sliced_expert_copy_ms(slice_copy_bytes, share, (double) si.input_expert_n_expert,
                                           (double) si.input_expert_size, si.input_expert_segments) + ids_round_trip_ms
                                     : 0.0);
            input_copy_ms = input_copy_weight_ms + (rest_wb_bytes / 1e9 / pcie_bw) * 1000.0 + rest_wb_n * copy_setup_ms;

            if (bd != nullptr) {
                // this split's copy work, prefetched or not; a dense FFN's streamed weights are FFN copies like an
                // expert stack's, not the layer's attention (weight copies carry the weight's name after a '#')
                double dense_ffn   = 0.0;
                int    dense_ffn_n = 0;
                {
                    std::vector<const ggml_tensor *> seen;
                    struct ggml_tensor ** gnodes = ggml_graph_nodes(si.graph);
                    const int n_gnodes = ggml_graph_n_nodes(si.graph);
                    for (int j = 0; j < n_gnodes; j++) {
                        const ggml_tensor * w = gnodes[j]->src[0];
                        if (gnodes[j]->op != GGML_OP_MUL_MAT || !llama_benchmark_node_is_ffn(gnodes[j]) ||
                                strchr(w->name, '#') == nullptr ||
                                std::find(seen.begin(), seen.end(), w) != seen.end()) {
                            continue;
                        }
                        seen.push_back(w);
                        dense_ffn += (double) ggml_nbytes(w);
                        dense_ffn_n++;
                    }
                }
                const double sliced = (double) si.input_weight_sliced_bytes;
                const double whole  = std::max(0.0, (double) si.input_moe_bytes - sliced) + dense_ffn;
                const double s_all  = sliced > 0.0
                    ? llama_expert_distinct_share(workload, rows, (double) si.input_expert_n_used, (double) si.input_expert_n_expert)
                    : 0.0;
                bd->expert_copy_ms += (sliced > 0.0
                        ? sliced_expert_copy_ms(sliced * s_all, s_all, (double) si.input_expert_n_expert,
                              (double) si.input_expert_size, si.input_expert_segments) + ids_round_trip_ms
                        : 0.0)
                    + (whole / 1e9 / weight_bw) * 1000.0 + (si.input_moe_n - si.input_weight_sliced_n + dense_ffn_n) * copy_setup_ms;
                // a streamed attention layer moves its weights and its cache upload together
                bd->stream_copy_ms += (std::max(0.0, (double) si.input_weight_bytes - sliced - whole) / 1e9 / weight_bw) * 1000.0
                    + std::max(0, si.input_weight_n - si.input_moe_n - dense_ffn_n) * copy_setup_ms
                    + (((double) si.writeback_kv_bytes * kv_stream_f + (double) si.writeback_rs_bytes) / 1e9 / pcie_bw) * 1000.0
                    + (si.writeback_kv_n + si.writeback_rs_n) * copy_setup_ms;
            }
        }

        // the store's SSD reads for the split's expert uploads, waited for tensor by tensor at consume time (the store
        // refuses a prefetch with a miss): a stack whose router runs in this split reads every expert not in RAM, one
        // whose ids an earlier split computed is copied sliced and reads the experts its rows route to. The copies of
        // the experts in RAM this split pays for run while the reads are in flight
        double ssd_ms = 0.0;
        if (st != nullptr && is_gpu && si.input_moe_bytes > 0 &&
                (si.input_expert_n_expert == 0 || si.input_expert_n_expert == (int64_t) st->n_expert)) {
            const double e       = (double) st->n_expert;
            const double sliced  = (double) (si.input_weight_sliced_bytes + si.input_expert_bytes);
            const double whole   = std::max(0.0, (double) si.input_moe_bytes - sliced);
            const int    n_sl    = si.input_weight_sliced_n + si.input_expert_n;
            const int    n_wh    = std::max(0, si.input_moe_n - n_sl);
            const double rows    = std::max(1.0, std::round((double) si.input_expert_n_rows * token_scale));
            const double n_used  = (double) si.input_expert_n_used;
            const double m       = store_misses(rows, n_used);
            const double read_ms = store_read_ms(e - st->n_ram, whole / e, n_wh) + store_read_ms(m, sliced / e, n_sl);
            const double hits    = std::max(0.0, e * llama_expert_distinct_share(workload, rows, n_used, e) - m);
            const double hit_b   = (double) si.input_weight_sliced_bytes / e * hits +
                (copy_prefetched ? 0.0 : (double) si.input_expert_bytes / e * hits + whole / e * st->n_ram);
            ssd_ms = std::max(0.0, read_ms - (weight_bw > 0.0 ? hit_b / 1e9 / weight_bw * 1000.0 : 0.0));
        }

        // the outstanding prefetch arrives here: what the compute of the splits since its trigger did not hide is
        // exposed at this split
        double exposed_ms = 0.0;
        if (pending_to == i) {
            double w_bw[2], wb_bw;
            prefetch_rates(is_gpu, 0.0, w_bw, wb_bw);
            for (int c = 0; c < 2; c++) {
                const double share = c == 0 ? 1.0 - f_staged : f_staged;
                exposed_ms += share * ((pending_w[c] / 1e9 / w_bw[c] + pending_wb[c] / 1e9 / wb_bw) * 1000.0 + pending_su[c]);
                pending_w[c]  = 0.0;
                pending_wb[c] = 0.0;
                pending_su[c] = 0.0;
            }
            pending_to = -1;
        }
        // this split issues a prefetch: enqueued after its inputs, before its compute (the prefetch pass moves the
        // target's non-sliced weights and its writebacks)
        double staged_serial_ms = 0.0;
        if (target_of[i] >= 0) {
            const ggml_backend_sched_split_info & tsi = infos[target_of[i]];
            for (int c = 0; c < 2; c++) {
                pending_w[c]  = (double) tsi.input_weight_prefetch_bytes;
                pending_wb[c] = (double) tsi.writeback_kv_bytes * kv_stream_f + (double) tsi.writeback_rs_bytes;
                pending_su[c] = (tsi.input_weight_prefetch_n + tsi.writeback_kv_n + tsi.writeback_rs_n) * copy_setup_ms;
            }
            pending_to = target_of[i];
            if (!is_gpu && f_staged > 0.0) {
                // a staged copy issued from a CPU split's host thread completes before that split computes
                staged_serial_ms = f_staged * (pending_w[1] / 1e9 / stats.upload_staged_bw * 1000.0 + pending_su[1]);
                pending_w[1]  = 0.0;
                pending_su[1] = 0.0;
            }
        }

        // a CPU split contends with the prefetch only while its bytes move
        const bool async_copy = pending_to >= 0 &&
            (pending_w[0] + pending_wb[0] + pending_w[1] + pending_wb[1] + pending_su[0] + pending_su[1]) > 0.0;

        // compute cost (CPU splits use eff_gflops when async_copy due to PCIe contention)
        struct ggml_tensor ** nodes = ggml_graph_nodes(si.graph);
        int n_nodes = ggml_graph_n_nodes(si.graph);
        llama_split_timing t = predict_split(nodes, n_nodes, is_gpu, batch_size, async_copy, &timing_cache, token_scale);

        // the store's SSD reads for the split's CPU expert matmuls: a layer's tensors share its routing, and the layer
        // waits once for the experts its rows use that are not in RAM
        if (st != nullptr && !is_gpu) {
            struct ids_reads { const ggml_tensor * ids; double bytes; int n; };   // one expert's bytes over n tensors
            std::vector<ids_reads> lay;
            for (int j = 0; j < n_nodes; j++) {
                const ggml_tensor * w   = nodes[j]->src[0];
                const ggml_tensor * ids = nodes[j]->src[2];
                if (nodes[j]->op != GGML_OP_MUL_MAT_ID || w == nullptr || ids == nullptr ||
                        w->ne[2] != (int64_t) st->n_expert || strstr(w->name, "_exps.weight") == nullptr) {
                    continue;
                }
                auto it = std::find_if(lay.begin(), lay.end(), [&](const auto & l) { return l.ids == ids; });
                if (it == lay.end()) {
                    lay.push_back({ ids, 0.0, 0 });
                    it = lay.end() - 1;
                }
                it->bytes += (double) w->nb[2];
                it->n++;
            }
            for (const auto & l : lay) {
                const double rows = std::max(1.0, std::round((double) l.ids->ne[1] * token_scale));
                ssd_ms += store_read_ms(store_misses(rows, (double) l.ids->ne[0]), l.bytes, 1, l.n);
            }
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
            kv_dl_ms = (dl_bytes / 1e9 / pcie_bw) * 1000.0 + (si.writeback_kv_n + si.writeback_rs_n) * copy_setup_ms;
        }

        // activation copy cost (synchronous, between splits on different backends)
        double activ_copy_ms = 0.0;
        if (si.input_activ_bytes > 0 && pcie_bw > 0.0) {
            activ_copy_ms = ((double)si.input_activ_bytes / 1e9 / pcie_bw) * 1000.0;
        }
        // a CPU split's handoff: its inputs come down and its result goes up as copies ordered against the GPU's
        // kernels, each a copy-engine transition (the expert pool prices its CPU routes' handoff itself)
        double handoff_ms = 0.0;
        if (!is_gpu && !pool_sched && stats.engine_switch_us > 0.0) {
            const bool down = si.input_activ_bytes > 0;
            const bool up   = i + 1 < n_splits && have_info[i + 1] && infos[i + 1].backend_id != cpu_backend_id;
            handoff_ms = ((down ? 1 : 0) + (up ? 1 : 0)) * stats.engine_switch_us / 1000.0;
            activ_copy_ms += handoff_ms;
        }

        // the outstanding prefetch moves under this split's compute (the link is the split's own after its input
        // copies)
        const double comp_ms = t.time_ms + kv_dl_ms;
        if (pending_to >= 0 && comp_ms > 0.0) {
            double w_bw[2], wb_bw;
            prefetch_rates(is_gpu, t.eff_pcie_bw, w_bw, wb_bw);
            for (int c = 0; c < 2; c++) {
                double cap_ms = comp_ms;
                const double su_use = std::min(cap_ms, pending_su[c]);
                pending_su[c] -= su_use;
                cap_ms -= su_use;
                const double w_use = std::min(cap_ms, pending_w[c] / 1e9 / w_bw[c] * 1000.0);
                pending_w[c] = std::max(0.0, pending_w[c] - w_use / 1000.0 * 1e9 * w_bw[c]);
                cap_ms -= w_use;
                const double wb_use = std::min(cap_ms, pending_wb[c] / 1e9 / wb_bw * 1000.0);
                pending_wb[c] = std::max(0.0, pending_wb[c] - wb_use / 1000.0 * 1e9 * wb_bw);
            }
        }
        const double split_ms = input_copy_ms + activ_copy_ms + exposed_ms + staged_serial_ms + comp_ms + ssd_ms;
        if (bd != nullptr) {
            (is_gpu ? bd->ssd_up_ms : bd->ssd_cpu_ms) += ssd_ms;
            bd->compute_ms       += t.time_ms;
            bd->cpu_ms           += is_gpu ? 0.0 : t.time_ms;
            bd->weight_upload_ms += input_copy_weight_ms + exposed_ms + staged_serial_ms;
            bd->other_ms         += (input_copy_ms - input_copy_weight_ms) + activ_copy_ms + kv_dl_ms;
            if (is_gpu) {
                for (int j = 0; j < n_nodes; j++) {
                    if (llama_benchmark_node_is_ffn(nodes[j])) {
                        bd->gpu_ffn_ms += predict_split(&nodes[j], 1, true, batch_size, async_copy, &timing_cache, token_scale).time_ms;
                    }
                }
            } else {
                for (int j = 0; j < n_nodes; j++) {
                    if (llama_benchmark_node_is_ffn(nodes[j])) {
                        bd->cpu_ffn_ms += t.time_ms + handoff_ms;
                        break;
                    }
                }
            }
        }
        total_ms += split_ms;

        double dl_bytes = is_gpu ? (double)si.writeback_kv_bytes * kv_ratio + (double)si.writeback_rs_bytes : 0.0;
        LLAMA_LOG_DEBUG("%s:   split %d/%d [%s] input_copy=%.3f (%.2f MiB) compute=%.3f kv_dl=%.3f (%.2f MiB) prefetch exposed=%.3f -> split %d activ=%.3f (%.2f MiB) ssd=%.3f -> %.3f ms"
            " (exact=%d near=%d fall=%d cache=%d)\n",
            __func__, i, n_splits, is_gpu ? "GPU" : "CPU",
            input_copy_ms, input_copy_bytes / (1024.0 * 1024.0),
            t.time_ms,
            kv_dl_ms, dl_bytes / (1024.0 * 1024.0),
            exposed_ms, target_of[i],
            activ_copy_ms, (double)si.input_activ_bytes / (1024.0 * 1024.0),
            ssd_ms, split_ms,
            t.n_exact, t.n_nearest, t.n_fallback, t.n_cache_hit);
    }

    double tps = (total_ms > 0.0) ? (batch_size * 1000.0 / total_ms) : 0.0;
    LLAMA_LOG_DEBUG("%s: total=%.3f ms, kv_ratio=%.4f, pcie_bw=%.1f GB/s, weight_bw=%.1f GB/s -> %.1f tps\n",
        __func__, total_ms, kv_ratio, pcie_bw, weight_bw, tps);
    return tps;
}
