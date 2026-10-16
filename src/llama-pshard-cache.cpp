#include "llama-kv-cache-dsv4.h"
#include "llama-pshard-plan.h"
#include "llama-benchmark.h"
#include "llama-expert-store.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-ssd-unit.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

const char * llama_get_overflow_pattern(size_t il, llama_layer_fraction lf) {
    constexpr size_t n_strings = 1000;
    GGML_ASSERT(il < n_strings);
    switch (lf) {
        case LLAMA_LAYER_FRACTION_ATTN: {
            thread_local std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_UP: {
            thread_local std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_GATE: {
            thread_local std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_MOE: {
            thread_local std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate)_(ch|)exps"; }
            return p[il].c_str();
        }
        default:
            return nullptr;
    }
}

void llama_pshard_generate_overrides(
        uint32_t n_pinned,
        uint32_t n_layers,
        ggml_backend_buffer_type_t gpu_buft,
        ggml_backend_buffer_type_t host_buft,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        llama_layer_fraction overflow_type,
        llama_pshard_strategy strategy,
        const pshard_dev_layout & layout,
        bool pin_from_back,
        bool output_on_gpu,
        uint32_t n_attn_pinned,
        bool overlap,
        bool ids_cross,
        uint32_t n_ffn_gpu,
        uint8_t in_place) {
    GGML_UNUSED(gpu_buft);

    thread_local std::array<std::string, 1000> patterns_layer;
    thread_local std::array<std::string, 1000> patterns_layer_router;
    thread_local std::array<std::string, 1000> patterns_layer_attn;
    thread_local std::array<std::string, 1000> patterns_layer_ffn;
    thread_local std::array<std::string, 1000> patterns_layer_moe_exps;
    thread_local std::string pat_output = "^output";

    const uint32_t il_pin_start = pin_from_back ? (n_layers - n_pinned) : 0;
    GGML_ASSERT(n_layers <= 1000);
    const uint32_t il_pin_end   = pin_from_back ? n_layers : n_pinned;
    const uint32_t il_boundary_raw = pin_from_back ? (il_pin_start > 0 ? il_pin_start - 1 : UINT32_MAX) : il_pin_end;
    const uint32_t il_boundary = (overflow_type != LLAMA_LAYER_FRACTION_NONE && il_boundary_raw < n_layers) ? il_boundary_raw : UINT32_MAX;
    const bool output_on_cpu = !output_on_gpu;

    size_t itbo = 0;
    // HYBRID_ATTNPRIO_FFNBALANCE: the unpinned layers (the MTP head aside) and those placed so far
    const uint32_t n_ffn = n_layers > n_pinned + g_pshard_n_layers_mtp ? n_layers - n_pinned - g_pshard_n_layers_mtp : 0;
    uint32_t n_unpinned = 0;
    uint32_t n_ffn_streamed = 0;   // HYBRID_ATTNPRIO_FFNBALANCE: FFNs streamed so far

    auto emit = [&](const char * pat, ggml_backend_buffer_type_t buft, int32_t bid) {
        tensor_buft_overrides[itbo] = { pat, buft, bid };
        itbo++;
    };

    {
        thread_local std::string pat_tok_embd = "^token_embd";
        const int32_t out_bid = output_on_cpu ? layout.cpu : layout.compute;
        emit(pat_output.c_str(), host_buft, out_bid);
        emit(pat_tok_embd.c_str(), host_buft, layout.cpu);
    }

    for (uint32_t il = 0; il < n_layers; il++) {
        if (patterns_layer[il].empty())      { patterns_layer[il]      = "blk\\." + std::to_string(il) + "\\..*"; }
        if (patterns_layer_attn[il].empty()) { patterns_layer_attn[il] = "blk\\." + std::to_string(il) + "\\.attn_(q|k|v|output|q_norm|k_norm).*"; }
        if (patterns_layer_ffn[il].empty())  { patterns_layer_ffn[il]  = "blk\\." + std::to_string(il) + "\\.ffn_((up|gate|down)\\.|(up|down|gate|gate_up)_(ch|)exps).*"; }
        if (patterns_layer_router[il].empty()) { patterns_layer_router[il] = "blk\\." + std::to_string(il) + "\\.(ffn_gate_inp|ffn_exp_probs_b).*"; }
        if (patterns_layer_moe_exps[il].empty()) { patterns_layer_moe_exps[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate|gate_up)_exps\\..*"; }

        if (strategy == LLAMA_PSHARD_EXPERT_POOL) {
            // MTP head layers: pinned whole, experts included, the placement the planner's copy prices
            // (the draft context is a stock sched with no pool and reads them every draft step). homing
            // the head's experts on the shard bid instead makes the post-fit MTP probe overshoot the plan
            // and the refit lands on a budget with no saved variant, which disables pshard
            if (g_pshard_n_layers_mtp > 0 && il >= n_layers - g_pshard_n_layers_mtp) {
                emit(patterns_layer[il].c_str(), host_buft, g_pshard_mtp_head_cpu ? layout.cpu : layout.compute);
                continue;
            }
            // fetch corner: routed experts live in host RAM (the pool's home);
            // everything else pins - see the planner copy for the rationale
            emit(patterns_layer_moe_exps[il].c_str(), host_buft, overlap ? layout.shard(il) : layout.shard_a);
            emit(patterns_layer[il].c_str(), host_buft, layout.compute);
            continue;
        }

        if (il == il_boundary) {
            const char * overflow_pat = llama_get_overflow_pattern(il, overflow_type);
            if (overflow_pat) {
                emit(overflow_pat, host_buft, layout.shard(il));
            }
            emit(patterns_layer[il].c_str(), host_buft, layout.compute);
        } else if (il >= il_pin_start && il < il_pin_end) {
            emit(patterns_layer[il].c_str(), host_buft, layout.compute);
        } else {
            // MTP head layers: read every draft step by the stock-sched draft context, so never
            // slot-streamed (concurrent reader). whenever the plan pins anything at all the head goes
            // to the compute GPU first (the probes price it, so viability shrinks the trunk pins);
            // pinning needs the draft ctx on stock, backed KV (per-context gate)
            if (g_pshard_n_layers_mtp > 0 && il >= n_layers - g_pshard_n_layers_mtp) {
                const bool pin_head = !g_pshard_mtp_head_cpu && (n_pinned > 0 || n_attn_pinned > 0);
                emit(patterns_layer[il].c_str(), host_buft, pin_head ? layout.compute : layout.cpu);
                continue;
            }
            // overlap=1: alternating shard slots (double-buffering lets the copy of layer i+1
            // overlap compute that still reads layer i's slot). overlap=0: one slot, no
            // prefetch - cheaper plan for budgets that cannot fund the second slot
            const int32_t shard_bid = overlap ? layout.shard(il) : layout.shard_a;
            switch (strategy) {
                case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM:
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;
                case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS: {
                    // the unpinned part runs on the CPU, or on the GPU reading its weights in place (in_place)
                    const int32_t rest = in_place ? shard_bid : layout.cpu;
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, rest);
                        emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, rest);
                    }
                    break;
                }
                case LLAMA_PSHARD_HYBRID_ATTNPRIO_FFNBALANCE: {
                    // n_ffn_gpu of the unpinned FFNs stream to the GPU, spread evenly, the rest compute on the CPU;
                    // attention is pinned for the first n_attn_pinned layers and streamed for the rest. Consecutive
                    // streamed FFNs alternate lanes, so the expert copies split evenly over both lanes, and a layer
                    // streaming only attention takes the lane of the next streamed FFN (the first lane when none
                    // streams): its shared expert then copies with the next layer's attention
                    const bool    ffn_gpu  = llama_pshard_hybrid_ffn_on_gpu(n_unpinned++, n_ffn_gpu, n_ffn);
                    const bool    attn_gpu = n_attn_pinned > 0 && il < n_attn_pinned;
                    const int32_t lane     = overlap ? layout.shard(n_ffn_streamed) : layout.shard_a;
                    n_ffn_streamed += ffn_gpu ? 1 : 0;
                    if (ffn_gpu) {
                        if (ids_cross) {
                            // router pinned on the compute GPU: ids land in an earlier split
                            // than the streamed experts -> sliced-by-used-ids uploads
                            emit(patterns_layer_router[il].c_str(), host_buft, layout.compute);
                        }
                        emit(patterns_layer_ffn[il].c_str(), host_buft, lane);
                    } else {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    }
                    if (attn_gpu) {
                        emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, lane);
                    }
                    break;
                }
                default: break;
            }
        }
    }
    tensor_buft_overrides[itbo] = { nullptr, nullptr, -1 };
}

static int pshard_overflow_from_name(const char * name) {
    if (!name) return 0;
    if (strcmp(name, "ATTN") == 0) return 1;
    if (strcmp(name, "UP")   == 0) return 2;
    if (strcmp(name, "GATE") == 0) return 3;
    if (strcmp(name, "MOE")  == 0) return 4;
    return 0;
}

static const size_t PSHARD_MIB = 1024ULL * 1024ULL;

static uint32_t pshard_bytes_to_mib_ceil(size_t bytes) {
    return (uint32_t)((bytes + PSHARD_MIB - 1) / PSHARD_MIB);
}

static size_t pshard_mib_to_bytes(uint32_t mib) {
    return (size_t)mib * PSHARD_MIB;
}

static size_t pshard_mib_to_bytes(double mib) {
    return (size_t)(mib * (double)PSHARD_MIB + 0.5);
}

static bool pshard_parse_variant_header(const std::string & line, uint32_t & budget_mib, uint32_t & cache_ubatch) {
    cache_ubatch = 0;
    if (sscanf(line.c_str(), "[variant budget=%u cache_ubatch=%u]", &budget_mib, &cache_ubatch) == 2) {
        return true;
    }
    return sscanf(line.c_str(), "[variant budget=%u]", &budget_mib) == 1;
}

// the loader's split of the weights (llama_model::load_tensors): the routed experts (expert_count experts per tensor,
// row_b the largest expert of one tensor) and every other tensor it loads; the nextn layers load only with load_mtp
static bool pshard_store_weight_bytes(const char * path_model, bool load_mtp, size_t & exps_b, size_t & other_b, size_t & row_b) {
    uint32_t n_layer_load = UINT32_MAX;   // blk.N loads for N below it
    uint32_t n_expert     = 0;
    bool     first        = true;
    for (const auto & path : llama_expert_store_files(path_model)) {
        struct gguf_init_params gip = { /*.no_alloc =*/ true, /*.ctx =*/ NULL };
        struct gguf_context * g = gguf_init_from_file(path.c_str(), gip);
        if (g == nullptr) {
            return false;
        }
        if (first) {
            first = false;
            const int64_t ka = gguf_find_key(g, "general.architecture");
            const std::string arch = ka >= 0 && gguf_get_kv_type(g, ka) == GGUF_TYPE_STRING ? gguf_get_val_str(g, ka) : "";
            auto get_u32 = [&](const char * suffix) -> uint32_t {
                const int64_t k = gguf_find_key(g, (arch + suffix).c_str());
                return k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(g, k) : 0;
            };
            n_expert = get_u32(".expert_count");
            const uint32_t n_block = get_u32(".block_count");
            const uint32_t n_nextn = get_u32(".nextn_predict_layers");
            if (!load_mtp && n_nextn > 0 && n_nextn < n_block) {
                n_layer_load = n_block - n_nextn;
            }
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char *    name = gguf_get_tensor_name(g, i);
            const int64_t * ne   = gguf_get_tensor_ne(g, i);
            const size_t    b    = gguf_get_tensor_size(g, i);
            int il = -1;
            const bool blk = sscanf(name, "blk.%d.", &il) == 1 && il >= 0;
            if (blk && (uint32_t) il >= n_layer_load) {
                continue;
            }
            const bool exps = blk && n_expert > 0 && ne[2] == (int64_t) n_expert && ne[3] == 1 && strstr(name, "_exps.weight") != nullptr;
            (exps ? exps_b : other_b) += b;
            row_b = exps ? std::max(row_b, b / n_expert) : row_b;
        }
        gguf_free(g);
    }
    return true;
}

llama_pshard_store_info llama_pshard_store_info_for(const char * path_model, const struct llama_model_params * mparams) {
    llama_pshard_store_info info;
    // the store reads the experts a pass misses from the mapping's files
    const auto mode = mparams->load_mode;
    const bool mmap = mode == LLAMA_LOAD_MODE_AUTO || mode == LLAMA_LOAD_MODE_MMAP || mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (path_model == nullptr || !mmap || mparams->max_ram_alloc == 0) {
        return info;
    }
    if (auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        size_t ram_free = 0;
        ggml_backend_dev_memory(cpu_dev, &ram_free, &info.ram_total);
    }
    const size_t reserve = (size_t) (LLAMA_PSHARD_HOST_PIN_RESERVE_GB * 1e9);
    uint64_t files_b = 0;
    const std::string largest = llama_expert_store_largest_file(path_model, &files_b);
    size_t exps_b  = 0;
    size_t other_b = 0;
    size_t row_b   = 0;
    // the tensors are less than the files: a model whose files fit in RAM is decided without reading its tables
    if (files_b + reserve <= info.ram_total || !pshard_store_weight_bytes(path_model, mparams->load_mtp, exps_b, other_b, row_b)) {
        return info;
    }
    const auto dec = llama_expert_store_decide(exps_b, other_b, info.ram_total, reserve, mparams->max_ram_alloc);
    info.on     = dec.on;
    info.budget = dec.budget;
    if (info.on) {
        // the store reads the largest file most: the line of the disk it is on
        info.unit = llama_ssd_unit_of(largest);
        info.line = llama_benchmark_ssd_line(info.unit);
    }
    return info;
}

static std::string pshard_store_slots_path(const std::string & path_model) {
    return path_model + ".pshard_store_slots";
}

static bool pshard_store_slots_parse(const char * line, uint64_t fingerprint, uint32_t budget_mib, uint32_t cache_ubatch, uint32_t & n) {
    unsigned long long fp = 0;
    uint32_t b = 0, u = 0;
    return sscanf(line, "fingerprint=%llx budget=%u cache_ubatch=%u slots=%u", &fp, &b, &u, &n) == 4 &&
        fp == fingerprint && b == budget_mib && u == cache_ubatch;
}

uint32_t llama_pshard_store_slots_seen(const std::string & path_model, uint64_t fingerprint, uint32_t budget_mib, uint32_t cache_ubatch) {
    FILE * f = ggml_fopen(pshard_store_slots_path(path_model).c_str(), "r");
    if (f == nullptr) {
        return 0;
    }
    uint32_t seen = 0;
    char line[256];
    while (fgets(line, sizeof(line), f) != nullptr) {
        uint32_t n = 0;
        if (pshard_store_slots_parse(line, fingerprint, budget_mib, cache_ubatch, n)) {
            seen = n;
        }
    }
    fclose(f);
    return seen;
}

void llama_pshard_store_slots_record(const std::string & path_model, uint64_t fingerprint, uint32_t budget_mib, uint32_t cache_ubatch, uint32_t n_slots) {
    const std::string path = pshard_store_slots_path(path_model);
    std::vector<std::string> keep;
    if (FILE * f = ggml_fopen(path.c_str(), "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f) != nullptr) {
            uint32_t n = 0;
            if (!pshard_store_slots_parse(line, fingerprint, budget_mib, cache_ubatch, n)) {
                keep.push_back(line);
            }
        }
        fclose(f);
    }
    FILE * f = ggml_fopen(path.c_str(), "w");
    if (f == nullptr) {
        LLAMA_LOG_WARN("%s: cannot write %s\n", __func__, path.c_str());
        return;
    }
    for (const auto & l : keep) {
        fputs(l.c_str(), f);
    }
    fprintf(f, "fingerprint=%016llx budget=%u cache_ubatch=%u slots=%u\n", (unsigned long long) fingerprint, budget_mib, cache_ubatch, n_slots);
    fclose(f);
}

size_t llama_pshard_ssd_profile(char * buf, size_t buf_size) {
    // the store reads each expert of a tensor on its own: the rate with many such reads in flight (a pass with many
    // misses), and a lone read's latency and rate, fitted to a small and a large read (priced at every read size). The
    // scratch file is larger than a drive's own cache (and fits FAT32); a slow drive writes for a few seconds only
    constexpr uint64_t max_b   = (4ull << 30) - (16u << 20);
    constexpr uint64_t min_b   = 256ull << 20;
    constexpr double   write_s = 4.0;
    constexpr size_t   small_b = 1u << 20;
    constexpr size_t   large_b = 16u << 20;
    std::string out;
    for (const auto & u : llama_ssd_units()) {
        std::unique_ptr<llama_ssd_scratch> scratch;
        const char * why = "no directory on it takes a scratch file";
        for (const auto & d : u.dirs) {
            scratch = std::make_unique<llama_ssd_scratch>();
            const bool ok = scratch->create(d, max_b, min_b, write_s);
            // a directory that is a link elsewhere would measure another disk
            if (ok && llama_ssd_unit_of(scratch->path) == u.id) {
                break;
            }
            const bool slow = !ok && !scratch->path.empty();
            scratch.reset();
            if (slow) {
                why = "it writes less than 256 MiB in 4 s";
                break;
            }
        }
        if (!scratch) {
            LLAMA_LOG_WARN("%s: %s (%s): %s - not measured\n", __func__, u.id.c_str(), u.desc.c_str(), why);
            continue;
        }
        const double sat   = llama_expert_store_read_gbps(scratch->path, large_b, 16, 1.5);
        const double qd1_s = llama_expert_store_read_gbps(scratch->path, small_b, 1, 1.5);
        const double qd1_l = llama_expert_store_read_gbps(scratch->path, large_b, 1, 1.5);
        if (sat <= 0.0 || qd1_s <= 0.0 || qd1_l <= 0.0) {
            LLAMA_LOG_WARN("%s: %s (%s): the scratch file did not read back - not measured\n", __func__, u.id.c_str(), u.desc.c_str());
            continue;
        }
        // a lone read takes lat + bytes / qd1 (ms, with MB and GB/s): the line through both sizes
        const double t_s = small_b / 1e6 / qd1_s;
        const double t_l = large_b / 1e6 / qd1_l;
        const double qd1 = t_l > t_s ? (large_b - small_b) / 1e6 / (t_l - t_s) : qd1_l;
        const double lat = std::max(0.0, t_s - small_b / 1e6 / qd1);
        LLAMA_LOG_INFO("%s: %s (%s): %.2f GB/s with 16 reads in flight; one at a time %.2f GB/s at 1 MiB, %.2f GB/s at 16 MiB (%.1f GiB file)\n",
            __func__, u.id.c_str(), u.desc.c_str(), sat, qd1_s, qd1_l, scratch->bytes / 1073741824.0);
        char nums[128];
        snprintf(nums, sizeof(nums), " sat=%.4g GB/s qd1=%.4g GB/s lat=%.4g ms (", sat, qd1, lat);
        out += "#   SSD_Read: unit=" + u.id + nums + u.desc + ")\n";
    }
    if (buf != nullptr && buf_size > 0) {
        const size_t n = std::min(out.size(), buf_size - 1);
        memcpy(buf, out.data(), n);
        buf[n] = '\0';
    }
    return out.size();
}

uint64_t pshard_registry_fingerprint(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        int64_t model_file_size,
        const llama_pshard_store_info & store) {

    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };

    // an MTP-loaded model has one extra placeable layer (the nextn head), so its
    // plans are not interchangeable with the trunk-only variant
    mix((uint64_t)(mparams->load_mtp ? 1 : 0));
    // verify-tier shape: spec runs advertise n_draft+1 outputs per sequence
    mix((uint64_t)cparams->n_outputs_max_per_seq);
    // the logits scratch every tier reserves: min(bs, n_outputs_max) rows (0 = n_batch)
    mix((uint64_t)cparams->n_outputs_max);

    mix(cparams->n_ctx);
    mix(cparams->n_seq_max);
    // the attention scratch follows the KV stream size: one unified stream of n_ctx cells
    // needs an n_ctx x n_tokens mask, n_seq_max streams need n_seq_max smaller ones
    mix((uint64_t)(cparams->kv_unified ? 1 : 0));
    mix(cparams->n_threads);
    mix((uint64_t)cparams->flash_attn_type);
    mix((uint64_t)cparams->type_k);
    mix((uint64_t)cparams->type_v);
    mix((uint64_t)model_file_size);
    // the pricing model and the profile files the plans were priced with: a plan priced by an
    // older predictor or from another profile is re-planned, not reused
    mix((uint64_t)LLAMA_BENCHMARK_PREDICTOR_VERSION);
    mix(llama_benchmark_profile_hash());
    mix((uint64_t)pshard_strategy_from_env());
    // mapped weights are not in the device's pinned host buffers: only a GPU that reads ordinary host memory reads
    // them in place (mixed only when one does, so other machines keep their fingerprints)
    mix((uint64_t)(llama_pshard_load_maps(mparams->load_mode) ? 1 : 0));
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && props.caps.host_pageable) {
            mix(0x70616765ull);
            break;
        }
    }
    // forced miss policy (QA override) plans differently; mixed only when set so
    // existing registries keep their fingerprints
    const int mp_env = pshard_miss_policy_from_env();
    if (mp_env >= 0) {
        mix((uint64_t)(mp_env + 1));
    }
    // a model the RAM expert store serves is priced from the store's RAM and the SSD_Read line, which the profile hash
    // leaves out ("" = no line: priced without SSD reads); mixed only then, so the other models keep their fingerprints
    if (store.on) {
        mix((uint64_t) LLAMA_PSHARD_STORE_PRICING_VERSION);
        mix((uint64_t) mparams->max_ram_alloc);
        mix((uint64_t) (store.ram_total / (1024 * 1024)));
        uint64_t hl = 0xcbf29ce484222325ULL;
        for (const unsigned char c : store.line) {
            hl ^= c;
            hl *= 0x100000001b3ULL;
        }
        mix(hl);
    }

    return h;
}


llama_pshard_plan_registry * llama_pshard_registry_create(uint32_t n_tier_max, uint32_t n_seq_max, uint32_t n_draft) {
    auto * registry = new llama_pshard_plan_registry();
    registry->init(n_tier_max, n_seq_max, n_draft);
    return registry;
}

void llama_pshard_registry_free(llama_pshard_plan_registry * registry) {
    delete registry;
}

struct llama_pshard_cache_probe {
    std::vector<llama_device> devs;
    uint32_t n_layers    = 0;
    uint32_t n_ctx_train = 0;
    uint32_t n_expert    = 0;
    size_t   vram_free   = 0;
    size_t   vram_budget = 0;
    size_t   vram_total  = 0;
    ggml_backend_buffer_type_t host_buft = nullptr;
};

size_t llama_pshard_extra_device_bytes(const llama_model & model, uint32_t n_seq_max, uint32_t n_rs_seq) {
    if (model.arch == LLM_ARCH_DEEPSEEK4) {
        return llama_kv_cache_dsv4_comp_state_bytes(model, n_seq_max, n_rs_seq);
    }
    return 0;
}

static bool llama_pshard_probe_model_only(
        const char * path_model,
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        size_t max_vram_mb,
        size_t fit_target_mb,
        llama_pshard_cache_probe & probe) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= GGML_LOG_LEVEL_ERROR ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_probe = *mparams;
    mparams_probe.no_alloc  = true;
    mparams_probe.pshard    = false;
    mparams_probe.load_mode = LLAMA_LOAD_MODE_NONE;

    // hparams and devices only: a vocabulary of the right size, no tokens
    llama_model * model = llama_model_load_for_probe(path_model, mparams_probe, nullptr);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    if (!model) {
        return false;
    }

    probe.devs        = model->devices;
    probe.n_layers    = model->hparams.n_layer();
    probe.n_ctx_train = model->hparams.n_ctx_train;
    probe.n_expert    = model->hparams.n_expert;
    g_pshard_unsupported_reason = llama_pshard_arch_unsupported(*model);
    g_pshard_extra_device_bytes = llama_pshard_extra_device_bytes(*model, cparams->n_seq_max, cparams->n_rs_seq);

    // MTP: the nextn head is a full extra layer (attn + experts) within block_count. when it will be
    // loaded, plan it like any other layer; otherwise blk.<nextn> falls to the loader's dev_layer
    // default (wholesale on GPU, outside the budget) and the warmup reserves see plan-blind view sizes
    if (mparams->load_mtp) {
        probe.n_layers += model->hparams.n_layer_nextn;
    }
    g_pshard_n_layers_mtp = mparams->load_mtp ? model->hparams.n_layer_nextn : 0;

    if (!probe.devs.empty()) {
        ggml_backend_dev_t dev = probe.devs[0].dev;
        ggml_backend_dev_memory(dev, &probe.vram_free, &probe.vram_total);

        const size_t mib = 1024ULL * 1024ULL;
        const size_t fit_target_bytes = fit_target_mb * mib;
        probe.vram_budget = max_vram_mb > 0
            ? max_vram_mb * mib
            : (probe.vram_free > fit_target_bytes ? probe.vram_free - fit_target_bytes : 0);

        probe.host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!probe.host_buft) {
            probe.host_buft = ggml_backend_cpu_buffer_type();
        }
    }

    llama_model_free(model);
    return true;
}

static bool llama_pshard_params_supported(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams) {
    const llama_model_params default_mparams = llama_model_default_params();

    auto disable = [](const char * reason) {
        LLAMA_LOG_WARN("%s: %s, disabling pshard\n", "llama_params_fit_pshard", reason);
        return false;
    };

    if (!cparams->offload_kqv) {
        return disable("offload_kqv=false is not supported");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        return disable("SPLIT_MODE_TENSOR is not supported");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
        return disable("SPLIT_MODE_ROW is not supported");
    }
    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        return disable("n_gpu_layers is already set by the user");
    }
    if (mparams->tensor_split) {
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (mparams->tensor_split[i] != 0.0f) {
                return disable("tensor_split is already set by the user");
            }
        }
    }
    if (mparams->tensor_buft_overrides &&
        (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        return disable("tensor_buft_overrides are already set by the user");
    }

    return true;
}

void llama_params_fit_pshard(
        const char * path_model,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t max_vram_mb,
        size_t fit_target_mb) {
    const std::string cache_path = std::string(path_model) + ".tensor_overrides.pshard_registry";

    if (!llama_pshard_params_supported(mparams, cparams)) {
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    llama_pshard_cache_probe probe;
    if (!llama_pshard_probe_model_only(path_model, mparams, cparams, max_vram_mb, fit_target_mb, probe)) {
        LLAMA_LOG_WARN("%s: failed to probe model metadata, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    if (probe.devs.empty()) {
        LLAMA_LOG_WARN("%s: no GPU devices found, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }
    if (g_pshard_unsupported_reason) {
        LLAMA_LOG_WARN("%s: %s, disabling pshard\n", __func__, g_pshard_unsupported_reason);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    const auto &   devs      = probe.devs;
    const uint32_t n_layers  = probe.n_layers;
    size_t         vram_free = probe.vram_budget;
    if (g_pshard_extra_device_bytes > 0 && vram_free > g_pshard_extra_device_bytes) {
        LLAMA_LOG_INFO("%s: reserving %.2f MiB for device memory kept outside the pshard arena (compressor state): budget %.1f -> %.1f MiB\n",
            __func__, g_pshard_extra_device_bytes / (1024.0 * 1024.0),
            vram_free / (1024.0 * 1024.0), (vram_free - g_pshard_extra_device_bytes) / (1024.0 * 1024.0));
        vram_free -= g_pshard_extra_device_bytes;
    }

    if (vram_free > 0) {
        mparams->max_vram_alloc = std::max<size_t>(1, pshard_bytes_to_mib_ceil(vram_free));
    }

    ggml_backend_buffer_type_t host_buft = probe.host_buft;

    LLAMA_LOG_INFO("%s: probe: %u layers, %.1f MiB VRAM free, %.1f MiB budget%s\n",
        __func__, n_layers,
        probe.vram_free / (1024.0 * 1024.0),
        vram_free / (1024.0 * 1024.0),
        max_vram_mb > 0 ? " (-mva)" : " (free - fit target)");

    const uint32_t n_ctx_plan = cparams->n_ctx > 0 ? cparams->n_ctx : probe.n_ctx_train;

    auto * registry = mparams->pshard_registry;
    if (!registry) {
        LLAMA_LOG_ERROR("%s: pshard_registry is null\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }
    const uint32_t requested_tier_max = registry->cache_ubatch;
    const uint32_t tier_max_auto = std::min(std::max(cparams->n_batch, (uint32_t) 16384), n_ctx_plan);
    const uint32_t tier_max = std::min(requested_tier_max > 0 ? requested_tier_max : tier_max_auto, n_ctx_plan);

    // speculative verify batches: the target context advertises n_draft+1
    // outputs per sequence; give that batch size its own priced tier
    const uint32_t n_draft_tier = cparams->n_outputs_max_per_seq > 1 ? cparams->n_outputs_max_per_seq - 1 : 0;
    if (registry->cache_ubatch != tier_max) {
        registry->init(tier_max, cparams->n_seq_max, n_draft_tier);
    }

    registry->budget_mib = pshard_bytes_to_mib_ceil(vram_free);
    registry->probe_budget_mib = pshard_bytes_to_mib_ceil(probe.vram_budget);
    const uint32_t runtime_n_batch = std::min(n_ctx_plan, cparams->n_batch);
    const uint32_t runtime_cache_ubatch = std::min(runtime_n_batch, cparams->n_ubatch == 0 ? runtime_n_batch : cparams->n_ubatch);
    registry->cache_ubatch = registry->cache_ubatch ? std::min(n_ctx_plan, registry->cache_ubatch) : runtime_cache_ubatch;

    int64_t model_file_size = 0;
    {
        FILE * mf = fopen(path_model, "rb");
        if (mf) {
#ifdef _WIN32
            _fseeki64(mf, 0, SEEK_END);
            model_file_size = _ftelli64(mf);
#else
            fseeko(mf, 0, SEEK_END);
            model_file_size = ftello(mf);
#endif
            fclose(mf);
        }
    }

    // the planner resolves n_ctx 0 to the training length; fingerprint the same number, but leave
    // the caller's 0 in place on every stock-fallback return (the stock fit treats 0 as auto)
    llama_context_params cparams_fp = *cparams;
    if (cparams_fp.n_ctx == 0) {
        cparams_fp.n_ctx = n_ctx_plan;
    }
    const uint64_t fp = pshard_registry_fingerprint(
        mparams, &cparams_fp, model_file_size, llama_pshard_store_info_for(path_model, mparams));

    if (!pshard_registry_load(registry, fp, cache_path.c_str(), host_buft, vram_free, false)) {
        LLAMA_LOG_WARN("%s: no matching plan cache at %s (fingerprint=0x%016" PRIx64 "): pshard is off unless this configuration is planned\n",
            __func__, cache_path.c_str(), fp);
        LLAMA_LOG_WARN("%s: fingerprint inputs (n_ctx, n_seq_max, kv_unified, n_threads, flash_attn, type_k/v, model file size, PSHARD_STRATEGY env, predictor version, the profile files' bytes; for a model larger than RAM also -mra, the RAM and the profile's SSD_Read line) must match the planner invocation exactly\n", __func__);
        registry->cache_missed = true;
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    if (!registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from cache (variant budget=%u MiB cache_ubatch=%u)\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch);
    }
    registry->fingerprint = fp;
    // the last run of these plans got fewer store slots than they were priced at: planning again prices them at those
    // (the plans run until then). Only plans this configuration's planner writes: it plans up to its own tier max
    const uint32_t plan_tier_max = std::min(requested_tier_max > 0 ? std::min(requested_tier_max, tier_max_auto) : tier_max_auto, n_ctx_plan);
    if (const uint32_t seen = llama_pshard_store_slots_seen(path_model, fp, registry->budget_mib, registry->cache_ubatch);
            seen > 0 && registry->store_slots > seen && registry->cache_ubatch <= plan_tier_max) {
        LLAMA_LOG_WARN("%s: the last run of these plans got %u store slots per layer, they were priced at %u: they need planning again\n",
            __func__, seen, registry->store_slots);
        registry->store_stale = true;
    }

    // cached baseline fit for this budget
    // use the normal load path
    if (registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: cache says baseline %.1f MiB fits this budget (variant budget=%u MiB cache_ubatch=%u), using baseline loading\n",
            __func__, registry->baseline_vram_req / (1024.0 * 1024.0), registry->budget_mib, registry->cache_ubatch);
        mparams->pshard = false;
        cparams->pshard = false;
        mparams->n_gpu_layers = n_layers + 1;
        tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
        mparams->tensor_buft_overrides = nullptr;
        return;
    }

    if (registry->cache_ubatch > 0) {
        const uint32_t pshard_ubatch = std::min(n_ctx_plan, registry->cache_ubatch);
        cparams->n_batch  = pshard_ubatch;
        cparams->n_ubatch = pshard_ubatch;
    }

    // pick the highest viable tier
    size_t default_tier = registry->tier_sizes.size();
    llama_pshard_plan * best = nullptr;
    for (size_t t = registry->tier_sizes.size(); t-- > 0; ) {
        llama_pshard_plan * candidate = registry->get_best(t);
        if (candidate && candidate->is_viable) {
            default_tier = t;
            best = candidate;
            break;
        }
    }
    if (!best) {
        LLAMA_LOG_WARN("%s: no viable plan in cache, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }
    if (default_tier < registry->tier_sizes.size() - 1) {
        LLAMA_LOG_INFO("%s: highest tier (bs=%u) not viable, falling back to bs=%u\n",
            __func__, registry->tier_sizes.back(), registry->tier_sizes[default_tier]);
        const uint32_t tier_bs = registry->tier_sizes[default_tier];
        cparams->n_batch  = std::min(cparams->n_batch,  tier_bs);
        cparams->n_ubatch = std::min(cparams->n_ubatch, tier_bs);
        LLAMA_LOG_INFO("%s: clamped n_batch/n_ubatch to %u\n", __func__, tier_bs);
    }

    if (best->n_pinned > n_layers) {
        LLAMA_LOG_WARN("%s: cache stale: n_pinned=%u > n_layers=%u, regenerate cache\n",
            __func__, best->n_pinned, n_layers);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    // the plan is accepted: the context runs at the length the plan was made for
    if (cparams->n_ctx == 0) {
        cparams->n_ctx = n_ctx_plan;
    }

    // mirror the plan path (llama_params_fit_pshard_plan step 8): the load-time
    // delegate flag must match the active plan's strategy, or the initial context
    // and scheduler reserves are built un-delegated - a different scheduling
    // regime than the one the planner's probes priced (first plan apply would
    // correct the model flag, but the early reserve graphs are already shaped)
    mparams->pshard_delegate_compute = llama_pshard_strategy_delegates_compute(best->strategy, best->in_place);

    // the variant records whether the union enforcer demoted the MTP head to CPU;
    // regenerate the load overrides under the placement the plans were priced with
    g_pshard_mtp_head_cpu = registry->mtp_head_cpu;

    const int32_t cpu_bid = pshard_dev_layout::compute_cpu_backend_id(devs.size());
    const pshard_dev_layout layout = pshard_dev_layout::for_device(0, cpu_bid);
    llama_pshard_generate_overrides(
        best->n_pinned, n_layers, host_buft, host_buft,
        tensor_buft_overrides,
        (llama_layer_fraction)best->overflow,
        best->strategy, layout,
        best->pin_from_back, best->output_on_gpu, best->n_attn_pinned,
        best->overlap, best->ids_cross, best->n_ffn_gpu, best->in_place);

    for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
        if (tensor_buft_overrides[i].backend_id == layout.compute) {
            tensor_buft_overrides[i].buft = host_buft;
        }
    }

    mparams->tensor_buft_overrides = tensor_buft_overrides;
    mparams->n_gpu_layers = n_layers + 1;
    // the switch-cost model prices structural attention pins from the layer count, which
    // the registry file does not carry: publish it in-memory exactly as the planner does
    mparams->pshard_registry->n_layers = n_layers;

    LLAMA_LOG_INFO("%s: plan: %s, n_pinned=%u/%u, vram=%zu MiB, n_gpu_layers=%d\n",
        __func__, llama_pshard_strategy_name(best->strategy),
        best->n_pinned, n_layers, mparams->max_vram_alloc, mparams->n_gpu_layers);
}
