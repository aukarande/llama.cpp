#include "llama-pshard-plan.h"
#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

const char * llama_get_overflow_pattern(size_t il, llama_layer_fraction lf) {
    constexpr size_t n_strings = 1000;
    GGML_ASSERT(il < n_strings);
    switch (lf) {
        case LLAMA_LAYER_FRACTION_ATTN: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_UP: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_GATE: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_MOE: {
            static std::array<std::string, n_strings> p;
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
        bool ids_cross) {
    GGML_UNUSED(gpu_buft);

    thread_local std::array<std::string, 1000> patterns_layer;
    thread_local std::array<std::string, 1000> patterns_layer_router;
    thread_local std::array<std::string, 1000> patterns_layer_attn;
    thread_local std::array<std::string, 1000> patterns_layer_ffn;
    thread_local std::string pat_output = "^output";

    const uint32_t il_pin_start = pin_from_back ? (n_layers - n_pinned) : 0;
    GGML_ASSERT(n_layers <= 1000);
    const uint32_t il_pin_end   = pin_from_back ? n_layers : n_pinned;
    const uint32_t il_boundary_raw = pin_from_back ? (il_pin_start > 0 ? il_pin_start - 1 : UINT32_MAX) : il_pin_end;
    const uint32_t il_boundary = (overflow_type != LLAMA_LAYER_FRACTION_NONE && il_boundary_raw < n_layers) ? il_boundary_raw : UINT32_MAX;
    const bool output_on_cpu = !output_on_gpu;

    size_t itbo = 0;

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

        // MTP head layers: ALWAYS CPU-resident. The stock-sched draft context reads
        // them concurrently (never slot-streamed), and its layer-40 KV cache picks
        // CPU mirrors only when is_cpu_only(il) - a PINNED nextn layer would select
        // pipe-shard GPU KV tensors that nothing packs for the draft ctx (unbacked ->
        // per-graph scratch -> garbage history -> n_accept=0).
        if (g_pshard_n_layers_mtp > 0 && il >= n_layers - g_pshard_n_layers_mtp) {
            emit(patterns_layer[il].c_str(), host_buft, layout.cpu);
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
            // overlap=1: alternating shard slots (double-buffering lets the copy of layer i+1
            // overlap compute that still reads layer i's slot). overlap=0: one slot, no
            // prefetch - cheaper plan for budgets that cannot fund the second slot
            const int32_t shard_bid = overlap ? layout.shard(il) : layout.shard_a;
            switch (strategy) {
                case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM:
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;
                case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, shard_bid);
                    emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    break;
                case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;
                case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                        emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, layout.cpu);
                    }
                    break;
                case LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE:
                    // even unpinned FFNs compute on CPU, odd ones stream to alternating shard
                    // slots so the copy overlaps CPU-FFN + attn compute; attn is pinned for the
                    // first n_attn_pinned layers and streamed for the rest (budget knob)
                    if (il % 2 == 0) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    } else {
                        if (ids_cross) {
                            // router pinned on the compute GPU: ids land in an earlier split
                            // than the streamed experts -> sliced-by-used-ids uploads
                            emit(patterns_layer_router[il].c_str(), host_buft, layout.compute);
                        }
                        emit(patterns_layer_ffn[il].c_str(), host_buft, overlap ? layout.shard(il / 2) : layout.shard_a);
                    }
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, overlap ? layout.shard(il / 2) : layout.shard_a);
                    }
                    break;
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

static bool pshard_plan_is_better(const llama_pshard_plan & candidate, const llama_pshard_plan & current) {
    if (!current.is_viable) return true;
    const bool candidate_has_tps = candidate.tps > 0.0f;
    const bool current_has_tps   = current.tps   > 0.0f;
    if (candidate_has_tps || current_has_tps) {
        if (candidate_has_tps != current_has_tps) return candidate_has_tps;
        if (candidate.tps != current.tps) return candidate.tps > current.tps;
    }
    if (candidate.n_pinned != current.n_pinned) return candidate.n_pinned > current.n_pinned;
    if (candidate.n_attn_pinned != current.n_attn_pinned) return candidate.n_attn_pinned > current.n_attn_pinned;
    if (candidate.overflow != current.overflow) return candidate.overflow > current.overflow;
    return candidate.total_vram_req < current.total_vram_req;
}

uint64_t pshard_registry_fingerprint(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        int64_t model_file_size) {

    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };

    // an MTP-loaded model has one extra placeable layer (the nextn head), so its
    // plans are not interchangeable with the trunk-only variant
    mix((uint64_t)(mparams->load_mtp ? 1 : 0));
    // verify-tier shape: spec runs advertise n_draft+1 outputs per sequence
    mix((uint64_t)cparams->n_outputs_max_per_seq);

    mix(cparams->n_ctx);
    mix(cparams->n_seq_max);
    mix(cparams->n_threads);
    mix((uint64_t)cparams->flash_attn_type);
    mix((uint64_t)cparams->type_k);
    mix((uint64_t)cparams->type_v);
    mix((uint64_t)model_file_size);
    mix((uint64_t)pshard_strategy_from_env());

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

static bool llama_pshard_probe_model_only(
        const char * path_model,
        const struct llama_model_params * mparams,
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

    llama_model * model = llama_model_load_from_file(path_model, mparams_probe);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    if (!model) {
        return false;
    }

    probe.devs        = model->devices;
    probe.n_layers    = model->hparams.n_layer();
    probe.n_ctx_train = model->hparams.n_ctx_train;
    probe.n_expert    = model->hparams.n_expert;

    // MTP: the nextn head is a full extra layer (attn + experts) WITHIN block_count.
    // When it will actually be loaded, plan it like any other layer - leaving it out
    // let blk.<nextn> fall to the loader's dev_layer default (wholesale on GPU,
    // outside the budget) and crashed warmup reserves with plan-blind view sizes.
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
    if (!llama_pshard_probe_model_only(path_model, mparams, max_vram_mb, fit_target_mb, probe)) {
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

    const auto &   devs      = probe.devs;
    const uint32_t n_layers  = probe.n_layers;
    const size_t   vram_free = probe.vram_budget;

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

    const uint64_t fp = pshard_registry_fingerprint(
        mparams, cparams, model_file_size);

    if (!pshard_registry_load(registry, fp, cache_path.c_str(), host_buft, vram_free, false)) {
        LLAMA_LOG_WARN("%s: no matching plan cache at %s (fingerprint=0x%016" PRIx64 "), disabling pshard\n",
            __func__, cache_path.c_str(), fp);
        LLAMA_LOG_WARN("%s: >>> pshard DISABLED: this run uses the STOCK path - benchmark numbers will not be pshard numbers <<<\n", __func__);
        LLAMA_LOG_WARN("%s: fingerprint inputs (n_ctx, n_seq_max, n_threads, flash_attn, type_k/v, model file size, PSHARD_STRATEGY env) must match the planner invocation exactly\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    if (!registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from cache (variant budget=%u MiB cache_ubatch=%u)\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch);
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

    // mirror the plan path (llama_params_fit_pshard_plan step 8): the load-time
    // delegate flag must match the active plan's strategy, or the initial context
    // and scheduler reserves are built un-delegated - a different scheduling
    // regime than the one the planner's probes priced (first plan apply would
    // correct the model flag, but the early reserve graphs are already shaped)
    mparams->pshard_delegate_compute = llama_pshard_strategy_delegates_compute(best->strategy);

    const int32_t cpu_bid = pshard_dev_layout::compute_cpu_backend_id(devs.size());
    const pshard_dev_layout layout = pshard_dev_layout::for_device(0, cpu_bid);
    llama_pshard_generate_overrides(
        best->n_pinned, n_layers, host_buft, host_buft,
        tensor_buft_overrides,
        (llama_layer_fraction)best->overflow,
        best->strategy, layout,
        best->pin_from_back, best->output_on_gpu, best->n_attn_pinned,
        best->overlap, best->ids_cross);

    for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
        if (tensor_buft_overrides[i].backend_id == layout.compute) {
            tensor_buft_overrides[i].buft = host_buft;
        }
    }

    mparams->tensor_buft_overrides = tensor_buft_overrides;
    mparams->n_gpu_layers = n_layers + 1;

    LLAMA_LOG_INFO("%s: plan: %s, n_pinned=%u/%u, vram=%zu MiB, n_gpu_layers=%d\n",
        __func__, llama_pshard_strategy_name(best->strategy),
        best->n_pinned, n_layers, mparams->max_vram_alloc, mparams->n_gpu_layers);
}
