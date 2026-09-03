#include "llama-pshard-plan.h"

#include "llama-benchmark.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <atomic>
#include <mutex>
#include <thread>

static std::mutex g_probe_mutex;

static std::vector<llama_device_memory_data> llama_get_device_memory_data_safe(
        const char * path_model, const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<llama_device> & devs, uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert, uint32_t & hp_n_embd_r,
        ggml_log_level log_level,
        llama_probe_hook_t probe_hook = nullptr,
        void * probe_hook_data = nullptr,
        uint32_t probe_n_tokens = 0,
        uint32_t probe_n_outputs = 0) {
    std::lock_guard<std::mutex> lock(g_probe_mutex);
    return llama_get_device_memory_data(path_model, mparams, cparams, devs,
        hp_ngl, hp_n_ctx_train, hp_n_expert, hp_n_embd_r, log_level, probe_hook, probe_hook_data,
        probe_n_tokens, probe_n_outputs);
}


static void llama_pshard_generate_overrides(
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
    thread_local std::array<std::string, 1000> patterns_layer;
    thread_local std::array<std::string, 1000> patterns_layer_attn;
    thread_local std::array<std::string, 1000> patterns_layer_ffn;
    thread_local std::array<std::string, 1000> patterns_layer_router;
    thread_local std::string pat_output = "^output";

    const uint32_t il_pin_start = pin_from_back ? (n_layers - n_pinned) : 0;
    GGML_ASSERT(n_layers <= 1000 && "pshard: n_layers exceeds thread_local pattern array capacity");
    const uint32_t il_pin_end   = pin_from_back ? n_layers : n_pinned;
    const uint32_t il_boundary_raw = pin_from_back ? (il_pin_start > 0 ? il_pin_start - 1 : UINT32_MAX) : il_pin_end;
    const uint32_t il_boundary = (overflow_type != LLAMA_LAYER_FRACTION_NONE && il_boundary_raw < n_layers) ? il_boundary_raw : UINT32_MAX;
    const bool output_on_cpu = !output_on_gpu;

    static constexpr size_t OVERRIDE_CAP = 4096;
    size_t itbo = 0;

    auto emit = [&](const char * pat, ggml_backend_buffer_type_t buft, int32_t bid) {
        GGML_ASSERT(itbo + 1 < OVERRIDE_CAP && "override array overflow");
        tensor_buft_overrides[itbo] = { pat, buft, bid };
        itbo++;
    };

    {
        thread_local std::string pat_tok_embd = "^token_embd";
        const int32_t out_bid = output_on_cpu ? layout.cpu : layout.compute;
        emit(pat_output.c_str(), output_on_cpu ? host_buft : gpu_buft, out_bid);
        emit(pat_tok_embd.c_str(), host_buft, layout.cpu);
    }

    // one pattern cache per thread
    for (uint32_t il = 0; il < n_layers; il++) {
        if (patterns_layer[il].empty())      { patterns_layer[il]      = "blk\\." + std::to_string(il) + "\\..*"; }
        if (patterns_layer_attn[il].empty()) { patterns_layer_attn[il] = "blk\\." + std::to_string(il) + "\\.attn_(q|k|v|output|q_norm|k_norm).*"; }
        if (patterns_layer_ffn[il].empty())  { patterns_layer_ffn[il]  = "blk\\." + std::to_string(il) + "\\.ffn_((up|gate|down)\\.|(up|down|gate|gate_up)_(ch|)exps).*"; }
        if (patterns_layer_router[il].empty()) { patterns_layer_router[il] = "blk\\." + std::to_string(il) + "\\.(ffn_gate_inp|ffn_exp_probs_b).*"; }

        if (il == il_boundary) {
            const char * overflow_pat = llama_get_overflow_pattern(il, overflow_type);
            if (overflow_pat) {
                emit(overflow_pat, host_buft, layout.shard(il));
            }
            emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
        } else if (il >= il_pin_start && il < il_pin_end) {
            emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
        } else {
            // MTP head layers: read every draft step by the stock-sched draft context.
            // Never slot-streamed (concurrent reader); PIN-PRIORITY: whenever the plan
            // pins anything at all, the head goes to the compute GPU first (the probes
            // price it, so viability shrinks the trunk pins accordingly). Pinned is
            // sound now that the draft ctx gets stock, backed KV (per-context gate).
            if (g_pshard_n_layers_mtp > 0 && il >= n_layers - g_pshard_n_layers_mtp) {
                const bool pin_head = !g_pshard_mtp_head_cpu && (n_pinned > 0 || n_attn_pinned > 0);
                emit(patterns_layer[il].c_str(), host_buft, pin_head ? layout.compute : layout.cpu);
                continue;
            }
            // overlap=1: alternating shard slots (double-buffering). overlap=0: one slot,
            // no prefetch - cheaper plan for budgets that cannot fund the second slot
            const int32_t shard_bid = overlap ? layout.shard(il) : layout.shard_a;

            switch (strategy) {
                case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM:
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;

                case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, shard_bid);
                    emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
                    break;

                case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;

                case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                        emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
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
                            // pin the tiny router on the compute GPU: its ids then land in an
                            // earlier split than the streamed experts, enabling the runtime's
                            // sliced-by-used-ids uploads (predictor decides per tier whether
                            // that beats full-upload prefetch on this machine)
                            emit(patterns_layer_router[il].c_str(), gpu_buft, layout.compute);
                        }
                        emit(patterns_layer_ffn[il].c_str(), host_buft, overlap ? layout.shard(il / 2) : layout.shard_a);
                    }
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, overlap ? layout.shard(il / 2) : layout.shard_a);
                    }
                    break;

                default: break;
            }
        }
    }
    tensor_buft_overrides[itbo] = { nullptr, nullptr, -1 };
    LLAMA_LOG_DEBUG("%s: %zu overrides emitted\n", __func__, itbo);
}

// overflow names by enum value
static const char * const PSHARD_FRAC_NAMES[] = { "NONE", "ATTN", "UP", "GATE", "MOE" };

struct llama_pshard_search_ctx {
    const char                               * path_model;
    const struct llama_model_params           * mparams;
    const struct llama_context_params         * cparams;
    struct llama_model_tensor_buft_override   * overrides;
    uint32_t                                   n_layers;
    size_t                                     vram_free;
    ggml_backend_buffer_type_t                 gpu_buft;
    ggml_backend_buffer_type_t                 host_buft;
    pshard_dev_layout                          layout;
    bool                                       is_moe;
    bool                                       has_rs = false;

    // optional TPS predictor
    const llama_benchmark_predictor          * predictor = nullptr;
    uint32_t                                   kv_size   = 0;
    uint32_t                                   cache_ubatch = 0;

    // MoE geometry + model size for the ids-cross decision (0 = unknown)
    uint32_t                                   n_expert      = 0;
    uint32_t                                   n_expert_used = 0;
    int64_t                                    model_size    = 0;
};

// decide whether ALTERNATE's streamed layers should pin their routers on the compute GPU
// (ids-crossing): sliced-by-used-ids uploads beat full-upload prefetch iff the sliced copy
// costs less than the un-hidable part of the full copy. On a fast interconnect (or with a
// slow CPU providing lots of cover) the full overlapped upload wins and this returns false.
// Pure analytic decision - no probing.
static bool pshard_alternate_ids_cross_wins(const struct llama_pshard_search_ctx & ctx);

static bool pshard_alternate_ids_cross_wins(const struct llama_pshard_search_ctx & ctx) {
    if (!ctx.is_moe || ctx.n_expert == 0 || ctx.n_expert_used == 0 ||
            ctx.model_size <= 0 || ctx.n_layers == 0) {
        return false;
    }
    const uint32_t bs = ctx.cparams->n_batch;
    // mirror the runtime gate: slicing only engages when few expert-token pairs are gathered
    if ((uint64_t) bs * ctx.n_expert_used * 2 >= ctx.n_expert) {
        return false;
    }
    double pcie = (ctx.predictor && ctx.predictor->stats.eff_pcie_bw > 0.0)
        ? ctx.predictor->stats.eff_pcie_bw : 25.0;
    if (ctx.predictor && ctx.predictor->stats.upload_bw > 0.0) {
        // expert streams source from the mmap mappings; past the pin ceiling they
        // move at the staged (host-DRAM-bound) rate, not the pinned-concurrent one
        pcie = std::min(pcie, ctx.predictor->stats.upload_bw);
    }
    const double dram = (ctx.predictor && ctx.predictor->stats.peak_system_bw > 0.0)
        ? ctx.predictor->stats.peak_system_bw : 40.0;
    const double b_full  = 0.85 * (double) ctx.model_size / ctx.n_layers;  // full expert set per layer
    const double frac    = std::min(1.0, (double) bs * ctx.n_expert_used / ctx.n_expert);
    const double b_slice = b_full * frac;
    const double t_full_ms  = b_full  / 1e9 / pcie * 1000.0;
    const double t_slice_ms = b_slice / 1e9 / pcie * 1000.0 + 0.3;  // + ids sync latency
    // cover the full upload could hide behind: the paired CPU-FFN (DRAM-bound expert
    // reads) plus the attention compute of the streamed layer
    const double cover_ms   = b_slice / 1e9 / dram * 1000.0 + 0.1;
    return t_slice_ms < std::max(0.0, t_full_ms - cover_ms);
}

struct llama_pshard_tps_hook_data {
    const llama_benchmark_predictor * predictor;
    int      cpu_backend_id;
    uint32_t kv_size;
    int32_t  batch_size;
    uint32_t n_outputs;
    bool     has_rs;
    float  * out_tps;
};

static void pshard_tps_probe_hook(llama_context * ctx, void * user_data) {
    auto * d = (llama_pshard_tps_hook_data *) user_data;
    if (!d || !d->predictor || !ctx) return;

    double tps = d->predictor->predict_tps(ctx->get_sched(), d->cpu_backend_id, d->kv_size, d->batch_size, d->n_outputs, d->has_rs);
    if (d->out_tps) {
        *d->out_tps = (float)tps;
    }
}

static std::vector<llama_device_memory_data> llama_pshard_probe_memory(
        const llama_pshard_search_ctx & ctx,
        const llama_model_params      & mparams,
        const llama_context_params    & cparams,
        ggml_log_level                  log_level,
        llama_probe_hook_t              probe_hook = nullptr,
        void                          * probe_hook_data = nullptr,
        bool                            overlap = true) {
    std::vector<llama_device> devs;
    uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0, hp_n_embd_r = 0;

    // probes must never take the canonical-preload path: once best_plans is
    // partially populated (fallbacks, demotion re-plans), a registry-carrying
    // probe load packs pinned KV into the external preload buffer and the
    // measurement no longer attributes it to mb.context (measured cache = 0)
    llama_model_params mparams_probe_clean = mparams;
    mparams_probe_clean.pshard_registry = nullptr;

    const uint32_t probe_n_tokens  = std::max<uint32_t>(1, cparams.n_batch ? cparams.n_batch : cparams.n_ubatch);
    const uint32_t probe_n_outputs = probe_n_tokens;

    llama_context_params cparams_probe = cparams;
    cparams_probe.pshard_overlap = overlap;
    if (ctx.cache_ubatch != 0) {
        cparams_probe.n_batch  = std::max(cparams_probe.n_batch, ctx.cache_ubatch);
        cparams_probe.n_ubatch = ctx.cache_ubatch;
    }

    return llama_get_device_memory_data_safe(
        ctx.path_model, &mparams_probe_clean, &cparams_probe, devs,
        hp_ngl, hp_n_ctx_train, hp_n_expert, hp_n_embd_r,
        log_level, probe_hook, probe_hook_data,
        probe_n_tokens, probe_n_outputs);
}

struct llama_pshard_tier_prune {
    uint32_t hi_pinned[LLAMA_PSHARD_COUNT];
    uint32_t hi_attn;
    bool     skip[LLAMA_PSHARD_COUNT];

    void init(uint32_t n_layers) {
        for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
            hi_pinned[s] = n_layers;
            skip[s] = false;
        }
        hi_attn = n_layers;
    }

    void update(int s, const llama_pshard_plan & plan) {
        if (!plan.is_viable) {
            return;
        }
        if (s == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
            if (plan.n_attn_pinned > 0) {
                hi_attn = plan.n_attn_pinned;
                hi_pinned[s] = plan.n_pinned;
            }
        } else {
            if (plan.n_pinned > 0) {
                hi_pinned[s] = plan.n_pinned;
            }
        }
    }

    // an attn-pin bound proven at a larger batch is INVALID at bs=1: activation
    // scratch shrinks ~350x between bs=8192 and bs=1, so far more attention fits.
    // (q35-16k-mva2000: inherited hi_attn=11 hid the attn=40 STATIC winner, 12.1
    // vs 29.6 predicted tps.) Search the decode tier with a fresh bound.
    uint32_t attn_hint(uint32_t n_batch) const {
        return n_batch <= 1 ? UINT32_MAX : hi_attn;
    }
};

static llama_pshard_plan llama_pshard_search_strategy(
        const llama_pshard_search_ctx & ctx,
        llama_pshard_strategy strategy,
        uint32_t hi_hint = UINT32_MAX,
        uint32_t lo_hint = 0,
        bool overlap = true) {

    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan plan;
    plan.strategy   = strategy;
    plan.batch_size = cparams->n_batch;
    plan.overlap    = overlap;
    plan.ids_cross  = (strategy == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) &&
                      pshard_alternate_ids_cross_wins(ctx);
    const bool ids_cross = plan.ids_cross;

    const bool delegate_compute = llama_pshard_strategy_delegates_compute(strategy);

    const uint32_t hi_default = n_layers - 1;
    uint32_t lo = lo_hint, hi = (hi_hint < hi_default) ? hi_hint : hi_default;
    uint32_t best_n_pinned = lo_hint;
    int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

    // try the upper bound first
    {
        llama_pshard_generate_overrides(hi, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy, layout, false, false, 0, overlap, ids_cross);
        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = delegate_compute;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;
        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, nullptr, nullptr, overlap);
            const int64_t gpu_used = d[0].mb.total();
            LLAMA_LOG_INFO("%s: [%s] n_pinned=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, llama_pshard_strategy_name(strategy), hi,
                gpu_used / (1024.0 * 1024.0),
                d[0].mb.model / (1024.0 * 1024.0), d[0].mb.context / (1024.0 * 1024.0), d[0].mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                best_n_pinned = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] hi-first probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), hi);
        }
    }

    while (lo <= hi) {
        uint32_t mid;
        if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
            int64_t target = (int64_t)vram_free;
            mid = lo + (uint32_t)((double)(target - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
            if (mid <= lo) mid = lo + 1;
            if (mid > hi)  mid = hi;
        } else {
            mid = (lo + hi) / 2;
        }

        llama_pshard_generate_overrides(mid, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy, layout, false, false, 0, overlap, ids_cross);

        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = delegate_compute;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, nullptr, nullptr, overlap);
            const int64_t gpu_used = d[0].mb.total();

            LLAMA_LOG_INFO("%s: [%s] n_pinned=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, llama_pshard_strategy_name(strategy), mid,
                gpu_used / (1024.0 * 1024.0),
                d[0].mb.model   / (1024.0 * 1024.0),
                d[0].mb.context / (1024.0 * 1024.0),
                d[0].mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

            if (gpu_used <= (int64_t)vram_free) {
                best_n_pinned = mid;
                lo = mid + 1;
                mem_lo = gpu_used;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), mid);
            if (mid == 0) break;
            hi = mid - 1;
        }
    }

    llama_layer_fraction best_overflow = LLAMA_LAYER_FRACTION_NONE;
    const uint32_t            fallback_n_pinned = best_n_pinned;                    // known fit
    const llama_layer_fraction fallback_overflow = LLAMA_LAYER_FRACTION_NONE;
    if (best_n_pinned < n_layers - 1) {
        const uint32_t frac_n_pinned = best_n_pinned + 1; // pin one more layer, partially
        auto try_frac = [&](llama_layer_fraction frac) -> bool {
            llama_pshard_generate_overrides(frac_n_pinned, n_layers, gpu_buft, host_buft,
                tensor_buft_overrides, frac, strategy, layout, false, false, 0, overlap, ids_cross);
            llama_model_params mp = *mparams;
            mp.pshard = true;
            mp.pshard_delegate_compute = delegate_compute;
            mp.n_gpu_layers = n_layers + 1;
            mp.tensor_buft_overrides = tensor_buft_overrides;
            try {
                const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, nullptr, nullptr, overlap);
                return d[0].mb.total() <= (int64_t)vram_free;
            } catch (...) {
                LLAMA_LOG_WARN("%s: [%s] overflow probe failed (frac=%d)\n", __func__, llama_pshard_strategy_name(strategy), (int)frac);
                return false;
            }
        };
        // try one partial boundary layer
        if (try_frac(LLAMA_LAYER_FRACTION_ATTN)) {
            best_n_pinned = frac_n_pinned;
            best_overflow = LLAMA_LAYER_FRACTION_ATTN;
            if (try_frac(LLAMA_LAYER_FRACTION_UP))   { best_overflow = LLAMA_LAYER_FRACTION_UP;
            if (try_frac(LLAMA_LAYER_FRACTION_GATE)) { best_overflow = LLAMA_LAYER_FRACTION_GATE;
            if (try_frac(LLAMA_LAYER_FRACTION_MOE))  { best_overflow = LLAMA_LAYER_FRACTION_MOE; }}}
        }
    }

    plan.n_pinned      = best_n_pinned;
    plan.overflow      = best_overflow;
    plan.output_on_gpu = false;

    llama_pshard_generate_overrides(best_n_pinned, n_layers, gpu_buft, host_buft,
        tensor_buft_overrides, best_overflow, strategy, layout, false, plan.output_on_gpu, 0, overlap, ids_cross);
    {
        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = delegate_compute;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata, overlap);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] final measurement probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), best_n_pinned);
            plan.is_viable = false;
        }
    }

    // drop the partial boundary layer if the final probe exceeds budget
    if (!plan.is_viable && best_overflow != LLAMA_LAYER_FRACTION_NONE) {
        best_n_pinned = fallback_n_pinned;
        best_overflow = fallback_overflow;
        plan.n_pinned = best_n_pinned;
        plan.overflow = best_overflow;
        llama_pshard_generate_overrides(best_n_pinned, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, best_overflow, strategy, layout, false, plan.output_on_gpu, 0, overlap, ids_cross);
        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = delegate_compute;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;
        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata, overlap);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            plan.is_viable = false;
        }
    }

    for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
        plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
    }

    return plan;
}

static llama_pshard_plan llama_pshard_search_attn_pin(
        const llama_pshard_search_ctx & ctx,
        llama_pshard_strategy strategy,
        uint32_t hi_attn_hint = UINT32_MAX,
        uint32_t hi_full_hint = UINT32_MAX,
        uint32_t lo_full_hint = 0,
        bool overlap = true) {

    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan plan;
    plan.strategy  = strategy;
    plan.overlap   = overlap;
    plan.ids_cross = (strategy == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) &&
                     pshard_alternate_ids_cross_wins(ctx);
    const bool ids_cross = plan.ids_cross;

    auto measure_vram = [&](uint32_t n_full, uint32_t n_attn, bool out_gpu) -> llama_memory_breakdown_data {
        llama_pshard_generate_overrides(n_full, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy,
            layout, false, out_gpu, n_attn, overlap, ids_cross);

        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = llama_pshard_strategy_delegates_compute(strategy);
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;

        const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, nullptr, nullptr, overlap);
        return d[0].mb;
    };

    // phase 1: maximize attention layers on GPU
    uint32_t n_attn = 0;
    {
        uint32_t lo = 0, hi = (hi_attn_hint < n_layers) ? hi_attn_hint : n_layers;
        int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

        try {
            auto mb = measure_vram(0, hi, false);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1] n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, hi, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                n_attn = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1] hi-first probe failed (n_attn=%u)\n", __func__, hi);
        }

        while (lo <= hi) {
            uint32_t mid;
            if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
                mid = lo + (uint32_t)((double)((int64_t)vram_free - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
                if (mid <= lo) mid = lo + 1;
                if (mid > hi)  mid = hi;
            } else {
                mid = (lo + hi) / 2;
            }

            try {
                auto mb = measure_vram(0, mid, false);
                int64_t gpu_used = mb.total();
                LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1] n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                    __func__, mid, gpu_used / (1024.0 * 1024.0),
                    mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                    vram_free / (1024.0 * 1024.0),
                    gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

                if (gpu_used <= (int64_t)vram_free) {
                    n_attn = mid;
                    lo = mid + 1;
                    mem_lo = gpu_used;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                    mem_hi = gpu_used;
                }
            } catch (...) {
                LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1] probe failed (n_attn=%u)\n", __func__, mid);
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
    }

    // moe tries output on gpu before ffn pinning
    // dense tries output on gpu after ffn pinning
    bool output_on_gpu = false;

    if (is_moe && n_attn >= n_layers) {
        try {
            auto mb = measure_vram(0, n_attn, true);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1b] output_on_gpu probe -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                output_on_gpu = true;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1b] output_on_gpu probe failed\n", __func__);
        }
    }

    // phase 2: maximize fully pinned layers
    uint32_t n_full = 0;
    {
        uint32_t hi_full_max = (hi_full_hint < n_attn) ? hi_full_hint : n_attn;
        uint32_t lo = lo_full_hint, hi = hi_full_max;
        int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

        try {
            auto mb = measure_vram(hi, n_attn, output_on_gpu);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p2] n_full=%u n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, hi, n_attn, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                n_full = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p2] hi-first probe failed (n_full=%u, n_attn=%u)\n", __func__, hi, n_attn);
        }

        while (lo <= hi) {
            uint32_t mid;
            if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
                mid = lo + (uint32_t)((double)((int64_t)vram_free - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
                if (mid <= lo) mid = lo + 1;
                if (mid > hi)  mid = hi;
            } else {
                mid = (lo + hi) / 2;
            }

            try {
                auto mb = measure_vram(mid, n_attn, output_on_gpu);
                int64_t gpu_used = mb.total();
                LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p2] n_full=%u n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                    __func__, mid, n_attn, gpu_used / (1024.0 * 1024.0),
                    mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                    vram_free / (1024.0 * 1024.0),
                    gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

                if (gpu_used <= (int64_t)vram_free) {
                    n_full = mid;
                    lo = mid + 1;
                    mem_lo = gpu_used;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                    mem_hi = gpu_used;
                }
            } catch (...) {
                LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p2] probe failed (n_full=%u, n_attn=%u)\n", __func__, mid, n_attn);
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
    }

    // dense tries output on gpu only after all layers fit
    if (!is_moe && !output_on_gpu && n_full >= n_layers) {
        try {
            auto mb = measure_vram(n_full, n_attn, true);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p3] output_on_gpu probe (n_full=%u) -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, n_full, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                output_on_gpu = true;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p3] output_on_gpu probe failed\n", __func__);
        }
    }

    plan.n_pinned      = n_full;
    plan.n_attn_pinned = n_attn;
    plan.output_on_gpu = output_on_gpu;

    llama_pshard_generate_overrides(n_full, n_layers, gpu_buft, host_buft,
        tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy,
        layout, false, plan.output_on_gpu, n_attn, overlap, ids_cross);
    {
        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.pshard_delegate_compute = llama_pshard_strategy_delegates_compute(strategy);
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata, overlap);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS] final measurement probe failed (n_full=%u, n_attn=%u)\n", __func__, n_full, n_attn);
            plan.is_viable = false;
        }
    }

    for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
        plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
    }

    return plan;
}



// plan cache serialization


static std::string pshard_plan_to_ot(const llama_pshard_plan & plan, ggml_backend_buffer_type_t host_buft) {
    std::string ot;
    const char * buft_name = ggml_backend_buft_name(host_buft);
    for (size_t i = 0; i < plan.overrides.size(); i++) {
        if (i > 0) ot += ',';
        ot += plan.overrides[i].pattern;
        ot += '=';
        ot += buft_name;
        ot += ':';
        ot += std::to_string(plan.overrides[i].backend_id);
    }
    return ot;
}

static const char * pshard_overflow_name(int overflow) {
    static const char * const names[] = { "NONE", "ATTN", "UP", "GATE", "MOE" };
    return (overflow >= 0 && overflow < 5) ? names[overflow] : "NONE";
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
static const int PSHARD_CACHE_MAX_SECTIONS = 32;
static const int PSHARD_CACHE_MAX_VARIANTS = 16;

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

static bool pshard_plan_is_better(const llama_pshard_plan & candidate, const llama_pshard_plan & current);

// estimate, for every tier plan, the one-way cost of switching into it from the decode
// (tier 0) plan: the pinned-residency delta uploaded over PCIe. Byte counts are coarse
// (file-size based - the registry has no per-layer tensor sizes), which is fine: the term
// exists to separate "same residency, free switch" from "multi-GB pin swap around every
// prompt", not to rank close calls.
static void pshard_compute_switch_costs(
        llama_pshard_plan_registry * registry,
        int64_t model_file_size, uint32_t n_layers, bool is_moe, double pcie_gb_s) {
    if (!registry || registry->best_plans.empty() || n_layers == 0 ||
            model_file_size <= 0 || pcie_gb_s <= 0.0) {
        return;
    }
    const llama_pshard_plan & base = registry->best_plans[0];
    if (!base.is_viable) {
        return;
    }
    const double layer_bytes = 0.92 * (double)model_file_size / n_layers;
    const double attn_frac   = is_moe ? 0.12 : 0.35;  // attention share of a layer's bytes
    const double head_bytes  = 0.04 * (double)model_file_size;

    // publish the estimate constants: switches are pairwise (any plan to any plan), so
    // the runtime evaluates switch_cost_ms(from, to) on demand from these
    registry->switch_layer_mb  = (float)(layer_bytes / 1e6);
    registry->switch_attn_frac = (float)attn_frac;
    registry->switch_head_mb   = (float)(head_bytes / 1e6);
    registry->switch_pcie_gb_s = (float)pcie_gb_s;
    registry->n_layers         = n_layers;

    const double base_attn_extra = (double)(registry->attn_resident(base) - base.n_pinned);
    for (auto & plan : registry->best_plans) {
        if (&plan == &registry->best_plans[0] || !plan.is_viable) {
            plan.switch_ms = 0.0f;
            continue;
        }
        double bytes = 0.0;
        // fully pinned layers: nested sets when pinned from the same end, disjoint otherwise
        if (plan.pin_from_back == base.pin_from_back) {
            bytes += std::abs((double)plan.n_pinned - (double)base.n_pinned) * layer_bytes;
        } else {
            bytes += ((double)plan.n_pinned + (double)base.n_pinned) * layer_bytes;
        }
        // attention-only pins: resident attention beyond the fully pinned layers
        // (structural pins included, see llama_pshard_plan_registry::attn_resident)
        const double attn_extra = (double)(registry->attn_resident(plan) - plan.n_pinned);
        bytes += std::abs(attn_extra - base_attn_extra) * layer_bytes * attn_frac;
        if (plan.output_on_gpu != base.output_on_gpu) {
            bytes += head_bytes;
        }
        plan.switch_ms = (float)(bytes / 1e9 / pcie_gb_s * 1000.0);
    }
}

size_t llama_pshard_registry_arena_bytes(const struct llama_pshard_plan_registry * registry, size_t budget_bytes) {
    return registry ? registry->arena_bytes(budget_bytes) : budget_bytes;
}

bool pshard_registry_save(
        const llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        const llama_context_params * cparams) {
    if (!registry || !cache_path) return false;

    struct cache_section {
        std::string header;
        std::vector<std::string> lines;
    };
    std::vector<cache_section> sections;

    FILE * existing = fopen(cache_path, "r");
    if (existing) {
        char line[8192];
        cache_section * cur = nullptr;
        while (fgets(line, sizeof(line), existing)) {
            std::string s = line;
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s.compare(0, 13, "[fingerprint=") == 0) {
                sections.push_back({s, {}});
                cur = &sections.back();
            } else if (cur) {
                cur->lines.push_back(s);
            }
        }
        fclose(existing);
    }

    std::vector<cache_section> preserved_sections;
    std::vector<std::vector<std::string>> preserved_variants;

    size_t inferred_budget = registry->pshard_disabled ? registry->baseline_vram_req : 0;
    if (!registry->pshard_disabled) {
        for (const auto & plan : registry->best_plans) {
            if (plan.is_viable) {
                inferred_budget = std::max(inferred_budget, plan.total_vram_req);
            }
        }
    }
    const uint32_t budget_mib = registry->budget_mib
        ? registry->budget_mib
        : pshard_bytes_to_mib_ceil(inferred_budget);
    const uint32_t cache_ubatch = registry->cache_ubatch
        ? registry->cache_ubatch
        : (registry->tier_sizes.empty() ? 0 : registry->tier_sizes.back());

    for (const auto & sec : sections) {
        uint64_t fp = 0;
        sscanf(sec.header.c_str(), "[fingerprint=0x%" SCNx64, &fp);
        if (fp != fingerprint) {
            preserved_sections.push_back(sec);
            continue;
        }

        std::vector<std::string> cur_variant;
        uint32_t cur_budget = 0;
        uint32_t cur_cache_ubatch = 0;
        bool in_variant = false;
        auto flush_variant = [&]() {
            if (in_variant && (cur_budget != budget_mib || cur_cache_ubatch != cache_ubatch)) {
                preserved_variants.push_back(cur_variant);
            }
            cur_variant.clear();
            cur_budget = 0;
            cur_cache_ubatch = 0;
            in_variant = false;
        };

        for (const auto & ln : sec.lines) {
            uint32_t parsed_budget = 0;
            uint32_t parsed_cache_ubatch = 0;
            if (pshard_parse_variant_header(ln, parsed_budget, parsed_cache_ubatch)) {
                flush_variant();
                in_variant = true;
                cur_budget = parsed_budget;
                cur_cache_ubatch = parsed_cache_ubatch;
                cur_variant.push_back(ln);
            } else if (in_variant) {
                cur_variant.push_back(ln);
            }
        }
        flush_variant();
    }

    while ((int)preserved_sections.size() >= PSHARD_CACHE_MAX_SECTIONS) {
        preserved_sections.erase(preserved_sections.begin());
    }
    while ((int)preserved_variants.size() >= PSHARD_CACHE_MAX_VARIANTS) {
        preserved_variants.erase(preserved_variants.begin());
    }

    FILE * f = fopen(cache_path, "w");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not write plan cache: %s\n", __func__, cache_path);
        return false;
    }

    fprintf(f, "# Generated file. Edit at your own risk.\n");

    for (const auto & sec : preserved_sections) {
        fprintf(f, "\n%s\n", sec.header.c_str());
        for (const auto & ln : sec.lines) {
            fprintf(f, "%s\n", ln.c_str());
        }
    }

    fprintf(f, "\n[fingerprint=0x%016" PRIx64 "]\n", fingerprint);
    if (cparams) {
        const char * fa_str = "unknown";
        switch (cparams->flash_attn_type) {
            case LLAMA_FLASH_ATTN_TYPE_DISABLED: fa_str = "off";  break;
            case LLAMA_FLASH_ATTN_TYPE_ENABLED:  fa_str = "on";   break;
            case LLAMA_FLASH_ATTN_TYPE_AUTO:     fa_str = "auto"; break;
        }
        const int forced_strategy = pshard_strategy_from_env();
        fprintf(f, "# n_ctx=%u n_seq_max=%u n_threads=%d fa=%s type_k=%d type_v=%d strategy=%s\n",
            cparams->n_ctx, cparams->n_seq_max, cparams->n_threads,
            fa_str, (int)cparams->type_k, (int)cparams->type_v,
            forced_strategy >= 0 ? llama_pshard_strategy_name((llama_pshard_strategy)forced_strategy) : "auto");
    }

    for (const auto & variant : preserved_variants) {
        fprintf(f, "\n");
        for (const auto & ln : variant) {
            fprintf(f, "%s\n", ln.c_str());
        }
    }

    // trailing fields after cache_ubatch are ignored by older parsers (sscanf assigns
    // the two %u before the literal ']' mismatch and still returns 2)
    fprintf(f, "\n[variant budget=%u cache_ubatch=%u switch_mb=%.1f attn_frac=%.2f head_mb=%.1f pcie=%.1f mtp_head_cpu=%d union_mb=%zu]\n",
        budget_mib, cache_ubatch,
        registry->switch_layer_mb, registry->switch_attn_frac,
        registry->switch_head_mb, registry->switch_pcie_gb_s,
        registry->mtp_head_cpu ? 1 : 0,
        (size_t) ((registry->union_bytes + 1024 * 1024 - 1) / (1024 * 1024)));  // whole MiB, rounded up
    if (registry->pshard_disabled) {
        fprintf(f, "pshard_disabled=1 baseline_vram=%.1f\n", registry->baseline_vram_req / (1024.0 * 1024.0));
    } else {
        for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
            const auto & plan = registry->best_plans[t];
            if (!plan.is_viable) {
                fprintf(f, "[tier %zu bs=%u] not_viable\n", t, registry->tier_sizes[t]);
                continue;
            }
            fprintf(f, "[tier %zu bs=%u]\n", t, registry->tier_sizes[t]);
            fprintf(f, "strategy=%s n_pinned=%u n_attn_pinned=%u overflow=%s tps=%.2f vram=%.1f output_on_gpu=%d pin_from_back=%d overlap=%d switch_ms=%.2f ids_cross=%d\n",
                llama_pshard_strategy_name(plan.strategy),
                plan.n_pinned, plan.n_attn_pinned,
                pshard_overflow_name(plan.overflow),
                plan.tps, plan.total_vram_req / (1024.0 * 1024.0),
                (int)plan.output_on_gpu, (int)plan.pin_from_back, (int)plan.overlap,
                plan.switch_ms, (int)plan.ids_cross);
            fprintf(f, "ot=%s\n", pshard_plan_to_ot(plan, host_buft).c_str());
        }
    }

    fclose(f);
    LLAMA_LOG_INFO("%s: saved budget=%u MiB cache_ubatch=%u variant with %zu tier plans to %s\n",
        __func__, budget_mib, cache_ubatch, registry->tier_sizes.size(), cache_path);
    return true;
}

bool pshard_registry_load(
        llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        size_t current_budget, bool require_exact_budget) {
    if (!registry || !cache_path) return false;

    FILE * f = fopen(cache_path, "r");
    if (!f) return false;

    char line[8192];
    bool in_section = false;

    char fp_header[64];
    snprintf(fp_header, sizeof(fp_header), "[fingerprint=0x%016" PRIx64 "]", fingerprint);

    struct tier_data {
        uint32_t bs = 0;
        bool viable = false;
        llama_pshard_strategy strategy = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
        uint32_t n_pinned = 0;
        uint32_t n_attn_pinned = 0;
        int overflow = 0;
        float tps = 0.0f;
        double vram_mib = 0.0;
        int output_on_gpu = 0;
        int pin_from_back = 0;
        int overlap = 1;
        int ids_cross = 0;
        float switch_ms = 0.0f;
        std::string ot_line;
    };
    struct variant_data {
        uint32_t budget_mib = 0;
        uint32_t cache_ubatch = 0;
        bool pshard_disabled = false;
        double baseline_vram_mib = 0.0;
        float switch_layer_mb = 0.0f;
        float switch_attn_frac = 0.0f;
        float switch_head_mb = 0.0f;
        float switch_pcie_gb_s = 0.0f;
        bool  mtp_head_cpu     = false;
        size_t union_bytes     = 0;
        std::vector<tier_data> tiers;
    };
    std::vector<variant_data> variants;
    variant_data * cur_variant = nullptr;

    while (fgets(line, sizeof(line), f)) {
        std::string s = line;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();

        if (s.compare(0, 13, "[fingerprint=") == 0) {
            if (in_section) break; // hit next section, stop
            in_section = (s == fp_header);
            continue;
        }
        if (!in_section) continue;

        uint32_t variant_budget = 0;
        uint32_t variant_cache_ubatch = 0;
        if (pshard_parse_variant_header(s, variant_budget, variant_cache_ubatch)) {
            variants.push_back({});
            cur_variant = &variants.back();
            cur_variant->budget_mib = variant_budget;
            cur_variant->cache_ubatch = variant_cache_ubatch;
            // optional switch-cost estimate constants (absent in older caches)
            const char * p;
            if ((p = strstr(s.c_str(), "switch_mb=")) != NULL) cur_variant->switch_layer_mb  = (float)atof(p + 10);
            if ((p = strstr(s.c_str(), "attn_frac=")) != NULL) cur_variant->switch_attn_frac = (float)atof(p + 10);
            if ((p = strstr(s.c_str(), "head_mb="))   != NULL) cur_variant->switch_head_mb   = (float)atof(p + 8);
            if ((p = strstr(s.c_str(), "pcie="))      != NULL) cur_variant->switch_pcie_gb_s = (float)atof(p + 5);
            if ((p = strstr(s.c_str(), "mtp_head_cpu=")) != NULL) cur_variant->mtp_head_cpu = atoi(p + 13) != 0;
            if ((p = strstr(s.c_str(), "union_mb="))    != NULL) cur_variant->union_bytes = (size_t)(atof(p + 9) * 1024.0 * 1024.0);
            continue;
        }
        if (!cur_variant) continue;

        if (s.rfind("pshard_disabled=1", 0) == 0) {
            double baseline_mib = 0.0;
            if (sscanf(s.c_str(), "pshard_disabled=1 baseline_vram=%lf", &baseline_mib) == 1) {
                cur_variant->pshard_disabled = true;
                cur_variant->baseline_vram_mib = baseline_mib;
            } else {
                LLAMA_LOG_WARN("%s: malformed pshard_disabled line: %s\n", __func__, s.c_str());
            }
            continue;
        }

        if (s.compare(0, 5, "[tier") == 0) {
            tier_data td = {};
            size_t tier_idx = 0;
            if (s.find("not_viable") != std::string::npos) {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) {
                    LLAMA_LOG_WARN("%s: malformed tier header (not_viable): %s\n", __func__, s.c_str());
                    continue;
                }
                td.viable = false;
            } else {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) {
                    LLAMA_LOG_WARN("%s: malformed tier header: %s\n", __func__, s.c_str());
                    continue;
                }
                td.viable = true;
            }
            cur_variant->tiers.push_back(td);
        } else if (s.compare(0, 9, "strategy=") == 0 && !cur_variant->tiers.empty()) {
            auto & td = cur_variant->tiers.back();
            char strat_name[64] = {}, overflow_name[16] = {};
            if (sscanf(s.c_str(), "strategy=%63s n_pinned=%u n_attn_pinned=%u overflow=%15s tps=%f vram=%lf",
                   strat_name, &td.n_pinned, &td.n_attn_pinned, overflow_name, &td.tps, &td.vram_mib) < 4) {
                LLAMA_LOG_WARN("%s: malformed strategy line: %s\n", __func__, s.c_str());
                td.viable = false;
                continue;
            }

            const char * ogg = strstr(s.c_str(), "output_on_gpu=");
            const char * pfb = strstr(s.c_str(), "pin_from_back=");
            if (!ogg || !pfb) {
                LLAMA_LOG_WARN("%s: missing output_on_gpu/pin_from_back, invalidating cache: %s\n", __func__, s.c_str());
                td.viable = false;
                continue;
            }
            td.output_on_gpu = atoi(ogg + 14);
            td.pin_from_back = atoi(pfb + 14);
            const char * ovl = strstr(s.c_str(), "overlap=");
            td.overlap = ovl ? atoi(ovl + 8) : 1;
            const char * swm = strstr(s.c_str(), "switch_ms=");
            td.switch_ms = swm ? (float)atof(swm + 10) : 0.0f;
            const char * idc = strstr(s.c_str(), "ids_cross=");
            td.ids_cross = idc ? atoi(idc + 10) : 0;

            td.overflow = pshard_overflow_from_name(overflow_name);
            bool found_strategy = false;
            for (int i = 0; i < LLAMA_PSHARD_COUNT; i++) {
                if (strcmp(strat_name, llama_pshard_strategy_name((llama_pshard_strategy)i)) == 0) {
                    td.strategy = (llama_pshard_strategy)i;
                    found_strategy = true;
                    break;
                }
            }
            if (!found_strategy) {
                LLAMA_LOG_WARN("%s: unknown strategy in cache, invalidating tier: %s\n", __func__, strat_name);
                td.viable = false;
            }
        } else if (s.compare(0, 3, "ot=") == 0 && !cur_variant->tiers.empty()) {
            cur_variant->tiers.back().ot_line = s.substr(3);
        }
    }
    fclose(f);

    if (variants.empty() || !in_section) return false;

    auto make_plan = [&](const tier_data & td) {
        llama_pshard_plan plan;
        plan.strategy      = td.strategy;
        plan.batch_size    = td.bs;
        plan.n_pinned      = td.n_pinned;
        plan.n_attn_pinned = td.n_attn_pinned;
        plan.overflow      = td.overflow;
        plan.tps           = td.tps;
        plan.switch_ms     = td.switch_ms;
        plan.total_vram_req = (size_t)(td.vram_mib * 1024 * 1024);
        plan.is_viable     = td.viable;
        plan.overlap       = td.overlap != 0;
        plan.ids_cross     = td.ids_cross != 0;
        plan.ids_cross     = td.ids_cross != 0;
        plan.output_on_gpu = (bool)td.output_on_gpu;
        plan.pin_from_back = (bool)td.pin_from_back;

        if (!td.ot_line.empty()) {
            std::string remaining = td.ot_line;
            while (!remaining.empty()) {
                size_t comma = remaining.find(',');
                std::string token = (comma != std::string::npos) ? remaining.substr(0, comma) : remaining;
                remaining = (comma != std::string::npos) ? remaining.substr(comma + 1) : "";

                size_t eq = token.find('=');
                if (eq == std::string::npos) continue;

                std::string pattern = token.substr(0, eq);
                std::string buft_bid = token.substr(eq + 1);

                int32_t backend_id = -1;
                size_t colon = buft_bid.rfind(':');
                if (colon != std::string::npos) {
                    backend_id = atoi(buft_bid.c_str() + colon + 1);
                }

                plan.overrides.push_back({ pattern, host_buft, backend_id });
            }
        }
        return plan;
    };

    const uint32_t current_budget_mib = pshard_bytes_to_mib_ceil(current_budget);
    const uint32_t requested_cache_ubatch = registry->cache_ubatch;
    bool skipped_cache_ubatch = false;
    auto variant_cache_ubatch = [&](const variant_data & variant) {
        return variant.cache_ubatch ? variant.cache_ubatch : requested_cache_ubatch;
    };
    auto cache_ubatch_ok = [&](const variant_data & variant) {
        if (requested_cache_ubatch == 0) return true;
        if (variant.cache_ubatch == 0) return true;
        if (variant.cache_ubatch <= requested_cache_ubatch) return true;
        skipped_cache_ubatch = true;
        return false;
    };

    for (const auto & variant : variants) {
        if (!variant.pshard_disabled) continue;
        if (!cache_ubatch_ok(variant)) continue;
        const size_t baseline_vram = pshard_mib_to_bytes(variant.baseline_vram_mib);
        if (baseline_vram <= current_budget) {
            registry->tier_sizes.clear();
            registry->best_plans.clear();
            registry->pshard_disabled = true;
            registry->baseline_vram_req = baseline_vram;
            registry->budget_mib = variant.budget_mib;
            registry->cache_ubatch = variant_cache_ubatch(variant);
            LLAMA_LOG_INFO("%s: loaded pshard_disabled variant budget=%u MiB cache_ubatch=%u baseline=%.1f MiB from %s\n",
                __func__, variant.budget_mib, registry->cache_ubatch, variant.baseline_vram_mib, cache_path);
            return true;
        }
    }

    // deterministic variant selection (the accumulate + first-match era produced plans
    // from mixed planning sessions - see the phantom q8d 2.84 t/s incident):
    //   1. a variant planned for EXACTLY this budget always wins
    //   2. otherwise the largest stored budget <= requested (safe: only leaves VRAM idle)
    //   3. NEVER a variant planned for a bigger budget, and no per-tier salvage from
    //      one - its reserves were validated against headroom this run does not have
    //   4. remaining ties: larger cache_ubatch, then the NEWER entry (later in file)
    const variant_data * best_whole = nullptr;
    bool best_is_exact = false;
    for (const auto & variant : variants) {
        if (variant.pshard_disabled || variant.tiers.empty()) continue;
        if (!cache_ubatch_ok(variant)) continue;
        const bool exact = variant.budget_mib == current_budget_mib;
        if (require_exact_budget && !exact) continue;
        if (!exact && pshard_mib_to_bytes(variant.budget_mib) > current_budget) continue;
        if (!best_whole) {
            best_whole = &variant; best_is_exact = exact;
            continue;
        }
        if (exact != best_is_exact) {
            if (exact) { best_whole = &variant; best_is_exact = true; }
            continue;
        }
        const uint32_t vu = variant_cache_ubatch(variant);
        const uint32_t bu = variant_cache_ubatch(*best_whole);
        if (vu != bu) {
            if (vu > bu) { best_whole = &variant; }
            continue;
        }
        if (variant.budget_mib != best_whole->budget_mib) {
            if (variant.budget_mib > best_whole->budget_mib) { best_whole = &variant; }
            continue;
        }
        best_whole = &variant; // all keys equal: later in file = newer wins
    }

    std::vector<std::pair<uint32_t, llama_pshard_plan>> selected;
    if (best_whole) {
        for (const auto & td : best_whole->tiers) {
            llama_pshard_plan plan = make_plan(td);
            auto it = std::find_if(selected.begin(), selected.end(),
                [&](const auto & p) { return p.first == td.bs; });
            if (it == selected.end()) {
                selected.push_back({td.bs, std::move(plan)});
            } else {
                it->second = std::move(plan);
            }
        }
        if (!best_is_exact) {
            LLAMA_LOG_WARN("%s: no budget=%u MiB variant; using the budget=%u MiB plan (safe but leaves VRAM idle - replan at this budget for full utilization)\n",
                __func__, current_budget_mib, best_whole->budget_mib);
        }
    }

    const uint32_t selected_cache_ubatch = best_whole ? variant_cache_ubatch(*best_whole) : requested_cache_ubatch;
    selected.erase(std::remove_if(selected.begin(), selected.end(),
        [&](const auto & p) {
            if (p.first == 0) return true;
            return selected_cache_ubatch > 0 && p.first > selected_cache_ubatch;
        }), selected.end());
    if (selected.empty()) {
        if (require_exact_budget && !variants.empty()) {
            LLAMA_LOG_INFO("%s: cache miss, no exact budget=%u MiB variant in %s\n",
                __func__, current_budget_mib, cache_path);
        }
        if (skipped_cache_ubatch) {
            LLAMA_LOG_INFO("%s: cache miss, no variant with cache_ubatch <= target cache_ubatch=%u in %s\n",
                __func__, requested_cache_ubatch, cache_path);
        }
        return false;
    }

    std::sort(selected.begin(), selected.end(),
        [](const auto & a, const auto & b) { return a.first < b.first; });

    registry->tier_sizes.clear();
    registry->best_plans.clear();
    registry->pshard_disabled = false;
    registry->baseline_vram_req = 0;
    registry->budget_mib = best_whole ? best_whole->budget_mib : 0;
    registry->cache_ubatch = selected_cache_ubatch;
    if (best_whole) {
        registry->switch_layer_mb  = best_whole->switch_layer_mb;
        registry->switch_attn_frac = best_whole->switch_attn_frac;
        registry->switch_head_mb   = best_whole->switch_head_mb;
        registry->switch_pcie_gb_s = best_whole->switch_pcie_gb_s;
        registry->mtp_head_cpu     = best_whole->mtp_head_cpu;
        registry->union_bytes      = best_whole->union_bytes;
    }

    for (auto & item : selected) {
        const auto & p = item.second;
        if (p.is_viable && p.overrides.empty()) {
            LLAMA_LOG_WARN("%s: plan cache corrupt: tier bs=%u viable but has no overrides\n", __func__, item.first);
            registry->tier_sizes.clear();
            registry->best_plans.clear();
            return false;
        }
        registry->tier_sizes.push_back(item.first);
        registry->best_plans.push_back(std::move(item.second));
    }

    LLAMA_LOG_INFO("%s: loaded %zu tier plans from %s budget=%u MiB cache_ubatch=%u variant (current budget=%u MiB) in %s\n",
        __func__, registry->tier_sizes.size(), best_is_exact ? "exact" : "smaller",
        registry->budget_mib, registry->cache_ubatch, current_budget_mib, cache_path);
    return true;
}

static llama_pshard_plan llama_pshard_search_tier(
        const llama_pshard_search_ctx & ctx,
        int force_strategy,
        const std::vector<llama_device_memory_data> & dmds,
        llama_pshard_tier_prune & prune);

// ---- plan-time canonical-union accounting ----
//
// The loader builds ONE canonical GPU layout for all tier plans: the COMMON set
// (tensors resident in EVERY viable plan) is packed first, each plan's extras are
// stamped above it, and the pinned KV cache sits at the top of the buffer. Each
// plan was budget-checked ALONE by the search probes, so at tight budgets the
// combined layout can overshoot although every tier fits individually - the
// runtime then degrades to stock (llama-context-pshard.cpp pshard_setup_sched)
// or trips the per-tier assert at apply. Replicate the loader's arithmetic here
// (same sort, alignment and alloc-size rules) and demote tiers until every
// viable plan satisfies: stamped_scratch_off(plan) + pinned_cache(plan) <= budget.
static bool pshard_union_weight_less(const std::pair<std::string, size_t> & a,
                                     const std::pair<std::string, size_t> & b) {
    // mirrors llama-model.cpp pshard_weight_less: category, then layer, then name
    auto category = [](const std::string & n) -> int {
        if (n.find("attn_") != std::string::npos) return 0;
        if (n.find("exps")  != std::string::npos) return 4;
        if (n.find("ffn_")  != std::string::npos) return 1;
        if (n.find("norm")  != std::string::npos) return 2;
        return 3;
    };
    auto layer = [](const std::string & n) -> int {
        const size_t p = n.find("blk.");
        return p != std::string::npos ? atoi(n.c_str() + p + 4) : 9999;
    };
    const int ca = category(a.first), cb = category(b.first);
    if (ca != cb) return ca < cb;
    const int la = layer(a.first), lb = layer(b.first);
    if (la != lb) return la < lb;
    return a.first < b.first;
}

static void pshard_enforce_union_budget(
        llama_pshard_plan_registry * registry,
        llama_pshard_search_ctx    & ctx,
        const llama_context_params * cparams_base,
        const std::vector<llama_device_memory_data> & dmds,
        int force_strategy,
        const char * path_model,
        const llama_model_params * mparams) {
    if (!registry || registry->best_plans.empty() || mparams->max_vram_alloc == 0 || !ctx.gpu_buft) {
        return;
    }

    // tensor metadata (names + device alloc sizes), one cheap metadata-only load
    std::vector<std::pair<std::string, size_t>> tensors;  // name -> gpu alloc size
    {
        llama_model_params mp = *mparams;
        mp.no_alloc        = true;
        mp.load_mode       = LLAMA_LOAD_MODE_NONE;
        mp.pshard          = false;
        mp.pshard_registry = nullptr;
        llama_model * model = llama_model_load_from_file(path_model, mp);
        if (!model) {
            LLAMA_LOG_WARN("%s: metadata load failed, skipping union accounting\n", __func__);
            return;
        }
        tensors.reserve(model->tensors_by_name.size());
        for (const auto & [name, tensor] : model->tensors_by_name) {
            tensors.emplace_back(name, ggml_backend_buft_get_alloc_size(ctx.gpu_buft, tensor));
        }
        llama_model_free(model);
    }

    const size_t alignment    = ggml_backend_buft_get_alignment(ctx.gpu_buft);
    const size_t budget_bytes = (size_t) mparams->max_vram_alloc * 1024ULL * 1024ULL;
    auto align_up = [alignment](size_t off) { return ((off + alignment - 1) / alignment) * alignment; };

    // resident set of a plan: first override (emission order) with backend_id >= 0
    // whose pattern matches the tensor name; resident iff that bid == 0 (compute)
    auto resident_set = [&](const llama_pshard_plan & plan) {
        std::vector<std::regex> res;
        res.reserve(plan.overrides.size());
        for (const auto & ov : plan.overrides) { res.emplace_back(ov.pattern); }
        std::set<std::string> out;
        for (const auto & [name, sz] : tensors) {
            for (size_t i = 0; i < plan.overrides.size(); i++) {
                if (plan.overrides[i].backend_id < 0) continue;
                if (std::regex_search(name, res[i])) {
                    if (plan.overrides[i].backend_id == 0) { out.insert(name); }
                    break;
                }
            }
        }
        return out;
    };

    // demotion levers, cheapest first:
    //   pass 0: shave the violating tiers' own pins - re-plan each violator at an
    //           escalating reduced budget (every violator per round, so tied decode
    //           tiers converge together; doubling beats a search granularity coarser
    //           than the overshoot)
    //   pass 1: MTP only - the pinned head could not be paid for by trunk shaving;
    //           demote it to CPU variant-wide, re-plan every tier at the full budget,
    //           then shave again
    const int    max_rounds = 6;
    const size_t margin     = 32ULL * 1024 * 1024;
    std::vector<bool> orig_viable(registry->best_plans.size(), false);
    for (size_t t = 0; t < registry->best_plans.size(); t++) { orig_viable[t] = registry->best_plans[t].is_viable; }

    auto replan_tier = [&](size_t t, size_t vram_free) {
        llama_pshard_search_ctx ctx_t = ctx;
        ctx_t.vram_free = vram_free;
        // probes must not take the canonical-preload path (best_plans is non-empty now)
        llama_model_params mp_replan = *mparams;
        mp_replan.pshard_registry = nullptr;
        ctx_t.mparams = &mp_replan;
        llama_context_params cp_tier = *cparams_base;
        cp_tier.n_batch  = registry->tier_sizes[t];
        cp_tier.n_ubatch = cp_tier.n_batch;
        ctx_t.cparams = &cp_tier;
        llama_pshard_tier_prune prune;
        prune.init(ctx.n_layers);
        llama_pshard_plan p = llama_pshard_search_tier(ctx_t, force_strategy, dmds, prune);
        registry->best_plans[t] = p;
        registry->best_plans[t].is_viable = p.is_viable;
    };

    for (int pass = 0; pass < 2; pass++) {
        std::vector<size_t> reductions(registry->best_plans.size(), 0);  // per-tier escalating budget cut
        for (int round = 0; round < max_rounds; round++) {
            // 1. common = intersection of viable plans' resident sets
            std::vector<size_t> viable;
            std::vector<std::set<std::string>> residents(registry->best_plans.size());
            for (size_t t = 0; t < registry->best_plans.size(); t++) {
                if (!registry->best_plans[t].is_viable) continue;
                residents[t] = resident_set(registry->best_plans[t]);
                viable.push_back(t);
            }
            if (viable.empty()) return;

            std::set<std::string> common = residents[viable[0]];
            for (size_t vi = 1; vi < viable.size(); vi++) {
                std::set<std::string> next;
                for (const auto & n : common) {
                    if (residents[viable[vi]].count(n)) next.insert(n);
                }
                common.swap(next);
            }

            // 2. pack the common set (loader order): common_end = unpadded end of last tensor
            std::vector<std::pair<std::string, size_t>> common_sorted;
            for (const auto & [name, sz] : tensors) {
                if (common.count(name)) common_sorted.emplace_back(name, sz);
            }
            std::sort(common_sorted.begin(), common_sorted.end(), pshard_union_weight_less);
            size_t cursor = 0, common_end = 0;
            for (const auto & [name, sz] : common_sorted) {
                common_end = cursor + sz;
                cursor    += align_up(sz);
            }

            // 3. per viable plan: stamp extras above common_end, add its pinned cache
            std::vector<std::pair<size_t, size_t>> violators;  // tier -> overshoot bytes
            size_t max_need = 0;  // the variant's canonical union (largest tier need)
            for (size_t t : viable) {
                const auto & plan = registry->best_plans[t];
                std::vector<std::pair<std::string, size_t>> extras;
                for (const auto & [name, sz] : tensors) {
                    if (residents[t].count(name) && !common.count(name)) extras.emplace_back(name, sz);
                }
                std::sort(extras.begin(), extras.end(), pshard_union_weight_less);
                size_t so = align_up(common_end);
                for (const auto & [name, sz] : extras) { so = align_up(so + sz); }

                // the tier's compute scratch (streaming slots + graph temporaries, probe-measured)
                // must fit above the packed weights and the pinned cache too: otherwise the
                // runtime reserve falls back to constrained packing and galloc spills into an
                // overflow chunk OUTSIDE the arena (measured +496 MiB at a 3929 MiB budget)
                // + margin: the runtime canonical packing (pshard_compute_scratch_off) rounds
                // differently from this metadata pass by up to a few tens of MiB (seen +29.7 MiB
                // at a 2024 MiB budget: a tier that passed here by 1.2 MiB was marked unviable at
                // load, and every verify batch then ran in the 512-token streaming plan)
                const size_t need = so + plan.cache_measured + plan.scratch_measured + margin;
                LLAMA_LOG_INFO("%s: [tier bs=%u] union scratch_off=%.2f MiB + pinned cache=%.2f MiB + compute=%.2f MiB + margin=%.0f MiB = %.2f / %.2f MiB budget %s\n",
                    __func__, plan.batch_size, so / (1024.0*1024.0), plan.cache_measured / (1024.0*1024.0),
                    plan.scratch_measured / (1024.0*1024.0), margin / (1024.0*1024.0),
                    need / (1024.0*1024.0), budget_bytes / (1024.0*1024.0),
                    need <= budget_bytes ? "OK" : "OVERSHOOT");
                if (need > budget_bytes) { violators.emplace_back(t, need - budget_bytes); }
                max_need = std::max(max_need, need);
            }
            if (violators.empty()) {
                // every viable tier fits the canonical layout: this is what the arena must hold.
                // A tier without probe measurements (cache-seeded) would undercount it: leave 0
                // (= whole budget) rather than shrink the arena on a guess.
                bool measured = true;
                for (size_t t : viable) {
                    if (registry->best_plans[t].scratch_measured == 0) { measured = false; }
                }
                registry->union_bytes = measured ? max_need : 0;
                LLAMA_LOG_INFO("%s: canonical union %.2f MiB of the %.2f MiB budget (arena %.2f MiB, %.2f MiB leftover for a spec draft)\n",
                    __func__, max_need / (1024.0*1024.0), budget_bytes / (1024.0*1024.0),
                    registry->arena_bytes(budget_bytes) / (1024.0*1024.0),
                    (budget_bytes - registry->arena_bytes(budget_bytes)) / (1024.0*1024.0));
                return;
            }

            // 4. demote every violator: re-plan it at its (escalating) reduced budget
            for (const auto & [t, over] : violators) {
                auto & plan = registry->best_plans[t];
                reductions[t] = std::max(reductions[t] * 2, over);  // `over` already carries the margin
                const size_t new_bud = budget_bytes > reductions[t] ? budget_bytes - reductions[t] : 0;
                LLAMA_LOG_WARN("%s: canonical union overshoots by %.2f MiB; re-planning tier bs=%u at %.0f MiB\n",
                    __func__, over / (1024.0*1024.0), plan.batch_size, new_bud / (1024.0*1024.0));
                if (new_bud == 0) { plan.is_viable = false; continue; }
                replan_tier(t, new_bud);
            }
        }

        // trunk shaving did not converge: MTP head lever (once), then shave again
        if (g_pshard_n_layers_mtp > 0 && !g_pshard_mtp_head_cpu) {
            LLAMA_LOG_WARN("%s: union still overshoots after %d rounds with the MTP head pinned; re-planning all tiers with the head on CPU\n",
                __func__, max_rounds);
            g_pshard_mtp_head_cpu  = true;
            registry->mtp_head_cpu = true;
            for (size_t t = 0; t < registry->best_plans.size(); t++) {
                if (orig_viable[t]) { replan_tier(t, ctx.vram_free); }
            }
            continue;
        }
        break;
    }
    LLAMA_LOG_WARN("%s: union still overshoots after %d demotion rounds; leaving runtime degrade as backstop\n",
        __func__, max_rounds);
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

// true when every layer runs on the compute backend with no per layer override
// token_embd on host is allowed
// callers use this to skip pshard when baseline already fits
static bool pshard_plan_is_baseline_fit(
        const llama_pshard_plan & plan,
        uint32_t                  n_layers,
        int32_t                   compute_bid) {
    if (!plan.is_viable || plan.n_pinned < n_layers) return false;
    for (const auto & ov : plan.overrides) {
        // token_embd routes to CPU host buffer
        if (ov.pattern.find("token_embd") != std::string::npos) continue;
        if (ov.backend_id != compute_bid) return false;
    }
    return true;
}

static llama_pshard_plan llama_pshard_search_baseline_fit_tier(
        const llama_pshard_search_ctx & ctx,
        const std::vector<llama_device_memory_data> & dmds) {
    const auto * path_model = ctx.path_model;
    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;

    llama_pshard_plan plan;
    plan.batch_size = cparams->n_batch;

    llama_model_tensor_buft_override local_overrides[4096];
    local_overrides[0] = { nullptr, nullptr, -1 };

    llama_model_params mp_copy = *mparams;
    mp_copy.pshard = false;
    mp_copy.pshard_delegate_compute = false;
    mp_copy.tensor_buft_overrides = nullptr;
    mp_copy.max_vram_alloc = std::max<uint32_t>(1, pshard_bytes_to_mib_ceil(vram_free));

    llama_context_params cp_copy = *cparams;
    cp_copy.pshard = false;

    float ts[16] = {};
    size_t margins[16] = {};

    try {
        llama_params_fit_impl(path_model, &mp_copy, &cp_copy, ts, local_overrides, margins, 0,
            GGML_LOG_LEVEL_ERROR);

        const uint32_t ngl = (mp_copy.n_gpu_layers < 0)
            ? (n_layers + 1)
            : std::min((uint32_t)mp_copy.n_gpu_layers, n_layers + 1);
        plan.n_pinned      = (ngl > 0) ? (ngl - 1) : 0;
        plan.pin_from_back = false;
        plan.output_on_gpu = (ngl > 0);

        for (auto * ov = local_overrides; ov->pattern; ++ov) {
            if (ov->buft == gpu_buft) {
                ov->backend_id = layout.compute;
            } else {
                ov->buft       = host_buft;
                ov->backend_id = layout.cpu;
            }
            plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
        }

        mp_copy.tensor_buft_overrides = local_overrides[0].pattern ? local_overrides : nullptr;
        const auto d = llama_pshard_probe_memory(ctx, mp_copy, cp_copy, GGML_LOG_LEVEL_ERROR);
        plan.total_vram_req   = d[0].mb.total();
        plan.scratch_measured = d[0].mb.compute;
        plan.cache_measured   = d[0].mb.context;
        plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: baseline probe failed (bs=%u): %s\n",
            __func__, cparams->n_batch, e.what());
        plan.is_viable = false;
    } catch (...) {
        LLAMA_LOG_WARN("%s: baseline probe failed (bs=%u): unknown exception\n",
            __func__, cparams->n_batch);
        plan.is_viable = false;
    }

    LLAMA_LOG_INFO("%s: [bs=%-4u baseline] n_pinned=%2u/%2u, vram=%7.1f MiB, %s%s\n",
        __func__, cparams->n_batch, plan.n_pinned, n_layers,
        plan.total_vram_req / (1024.0 * 1024.0),
        plan.is_viable ? "VIABLE" : "NOT VIABLE",
        pshard_plan_is_baseline_fit(plan, n_layers, layout.compute) ? " (full fit)" : "");

    return plan;
}

// use attention priority when a forced strategy cannot fit the tier
// caller must set ctx.cparams for the target tier
static llama_pshard_plan llama_pshard_attn_pin_fallback(
        const llama_pshard_search_ctx & ctx,
        int      force_strategy,
        uint32_t hi_attn   = UINT32_MAX,
        uint32_t hi_pinned = UINT32_MAX) {
    llama_pshard_plan fallback = llama_pshard_search_attn_pin(ctx, LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS, hi_attn, hi_pinned);
    fallback.batch_size = ctx.cparams->n_batch;
    LLAMA_LOG_INFO("llama_params_fit_pshard: [bs=%-4u %-10s] forced %s non-viable, STATIC_ATTNPRIO_ALLMODELS fallback: n_pinned=%2u/%2u, %s\n",
        ctx.cparams->n_batch, llama_pshard_strategy_name(LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS),
        llama_pshard_strategy_name((llama_pshard_strategy)force_strategy),
        fallback.n_pinned, fallback.n_attn_pinned,
        fallback.is_viable ? "VIABLE" : "NOT VIABLE");
    return fallback;
}

static llama_pshard_plan llama_pshard_search_tier(
        const llama_pshard_search_ctx & ctx,
        int force_strategy,
        const std::vector<llama_device_memory_data> & dmds,
        llama_pshard_tier_prune & prune) {

    const auto * path_model = ctx.path_model;
    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan best;

    for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
        if (force_strategy >= 0 && force_strategy != s) continue;
        if (prune.skip[s]) continue;

        llama_pshard_strategy strategy = (llama_pshard_strategy)s;
        llama_pshard_plan plan;

        if (strategy == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS ||
            strategy == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) {
            plan = llama_pshard_search_attn_pin(ctx, strategy, prune.attn_hint(cparams->n_batch), prune.hi_pinned[s]);
        } else {
            plan = llama_pshard_search_strategy(ctx, strategy, prune.hi_pinned[s]);
        }

        // a strategy priced out by the overlap machinery (double slots + prefetch keepalives)
        // may still fit without it: keep the placement, drop the transport luxury
        if (!plan.is_viable && !llama_pshard_strategy_delegates_compute(strategy)) {
            llama_pshard_plan p2;
            if (strategy == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS ||
                strategy == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) {
                p2 = llama_pshard_search_attn_pin(ctx, strategy, prune.attn_hint(cparams->n_batch), prune.hi_pinned[s],
                        0, /*overlap=*/false);
            } else {
                p2 = llama_pshard_search_strategy(ctx, strategy, prune.hi_pinned[s],
                        0, /*overlap=*/false);
            }
            if (p2.is_viable) {
                LLAMA_LOG_INFO("%s: [%s] overlap machinery does not fit the budget; using overlap=0 plan\n",
                    __func__, llama_pshard_strategy_name(strategy));
                plan = p2;
            }
        }

        plan.batch_size = cparams->n_batch;

        {
            const char * status = plan.is_viable ? "VIABLE" : "NOT VIABLE";
            char tps_buf[32] = "";
            if (plan.tps > 0.0f) { snprintf(tps_buf, sizeof(tps_buf), ", tps=%.1f", plan.tps); }

            if (plan.n_attn_pinned > 0) {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u (attn=%2u), overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cparams->n_batch, llama_pshard_strategy_name(strategy),
                    plan.n_pinned, plan.n_attn_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            } else {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u, overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cparams->n_batch, llama_pshard_strategy_name(strategy),
                    plan.n_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            }
        }

        prune.update(s, plan);

        if (plan.is_viable && pshard_plan_is_better(plan, best)) {
            best = plan;
        }
    }

    if (!best.is_viable && force_strategy >= 0 && force_strategy != LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
        // before abandoning the forced strategy, try it without the overlap machinery
        llama_pshard_plan forced_noovl;
        if (force_strategy == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) {
            forced_noovl = llama_pshard_search_attn_pin(ctx, (llama_pshard_strategy)force_strategy,
                    prune.attn_hint(cparams->n_batch), UINT32_MAX, 0, /*overlap=*/false);
        } else {
            forced_noovl = llama_pshard_search_strategy(ctx, (llama_pshard_strategy)force_strategy,
                    UINT32_MAX, 0, /*overlap=*/false);
        }
        if (forced_noovl.is_viable) {
            LLAMA_LOG_INFO("%s: forced %s viable only with overlap=0\n",
                __func__, llama_pshard_strategy_name((llama_pshard_strategy)force_strategy));
            return forced_noovl;
        }
        llama_pshard_plan fallback = llama_pshard_attn_pin_fallback(
            ctx, force_strategy, prune.attn_hint(cparams->n_batch), prune.hi_pinned[LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS]);
        prune.update(LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS, fallback);
        if (fallback.is_viable) {
            best = fallback;
        }
    }

    return best;
}

// probe one strategy across tiers
static void llama_pshard_strategy_sweep(
        int strategy,
        const llama_pshard_search_ctx & ctx_template,
        const llama_context_params   & cparams_base,
        const std::vector<llama_device_memory_data> & dmds,
        const llama_pshard_plan_registry & registry,
        int force_strategy,
        llama_pshard_plan * out_plans,   // write out_plans[tier]
        size_t n_tiers,
        size_t first_tier) {

    if (force_strategy >= 0 && force_strategy != strategy) return;

    llama_model_tensor_buft_override local_overrides[4096];
    llama_pshard_search_ctx ctx = ctx_template;
    ctx.overrides = local_overrides;

    llama_pshard_tier_prune prune;
    prune.init(ctx.n_layers);

    // smaller tiers start from the previous fit
    // smaller batches usually need less scratch
    // hybrid SSM graphs are not monotonic in batch size
    uint32_t prev_n_pinned = 0;

    for (int t = (int)n_tiers - 1; t >= (int)first_tier; t--) {
        if (prune.skip[strategy]) break;

        llama_context_params cp_tier = cparams_base;
        cp_tier.n_batch  = registry.tier_sizes[t];
        cp_tier.n_ubatch = cp_tier.n_batch;
        ctx.cparams = &cp_tier;

        llama_pshard_strategy strat = (llama_pshard_strategy)strategy;
        llama_pshard_plan plan;

        if (out_plans[t].is_viable && out_plans[t].strategy == strat) {
            LLAMA_LOG_INFO("%s: === tier %d (bs=%u) [cached %s] ===\n",
                __func__, t, cp_tier.n_batch, llama_pshard_strategy_name(strat));
            plan = out_plans[t];
        } else {
            LLAMA_LOG_INFO("%s: === tier %d (bs=%u) [parallel %s] ===\n",
                __func__, t, cp_tier.n_batch, llama_pshard_strategy_name(strat));

            const uint32_t lo_hint = ctx.has_rs ? 0 : prev_n_pinned;

            if (strat == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS ||
                strat == LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE) {
                plan = llama_pshard_search_attn_pin(ctx, strat, prune.attn_hint(cp_tier.n_batch), UINT32_MAX, lo_hint);
                if (!plan.is_viable && !llama_pshard_strategy_delegates_compute(strat)) {
                    llama_pshard_plan p2 = llama_pshard_search_attn_pin(ctx, strat, prune.attn_hint(cp_tier.n_batch),
                            UINT32_MAX, lo_hint, /*overlap=*/false);
                    if (p2.is_viable) { plan = p2; }
                }
                plan.batch_size = cp_tier.n_batch;
            } else {
                plan = llama_pshard_search_strategy(ctx, strat, UINT32_MAX, lo_hint);
                if (!plan.is_viable) {
                    llama_pshard_plan p2 = llama_pshard_search_strategy(ctx, strat, UINT32_MAX,
                            lo_hint, /*overlap=*/false);
                    if (p2.is_viable) { plan = p2; }
                }
                plan.batch_size = cp_tier.n_batch;
            }
        }

        prune.update(strategy, plan);

        if (plan.is_viable && plan.n_pinned > prev_n_pinned) {
            prev_n_pinned = plan.n_pinned;
        }

        {
            const char * status = plan.is_viable ? "VIABLE" : "NOT VIABLE";
            char tps_buf[32] = "";
            if (plan.tps > 0.0f) { snprintf(tps_buf, sizeof(tps_buf), ", tps=%.1f", plan.tps); }

            if (plan.n_attn_pinned > 0) {
                LLAMA_LOG_INFO("%s: [tier=%d bs=%-4u %-10s] n_pinned=%2u (attn=%2u), overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, t, cp_tier.n_batch, llama_pshard_strategy_name(strat),
                    plan.n_pinned, plan.n_attn_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            } else {
                LLAMA_LOG_INFO("%s: [tier=%d bs=%-4u %-10s] n_pinned=%2u, overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, t, cp_tier.n_batch, llama_pshard_strategy_name(strat),
                    plan.n_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            }
        }

        out_plans[t] = plan;
    }
}

static void llama_pshard_parallel_worker(
        std::atomic<int> & next_strategy,
        const llama_pshard_search_ctx & ctx_template,
        const llama_context_params   & cparams_base,
        const std::vector<llama_device_memory_data> & dmds,
        const llama_pshard_plan_registry & registry,
        int force_strategy,
        llama_pshard_plan * all_plans,
        size_t n_tiers,
        size_t first_tier) {

    while (true) {
        int s = next_strategy.fetch_add(1);
        if (s >= LLAMA_PSHARD_COUNT) return;

        llama_pshard_plan * out = all_plans + s * n_tiers;
        llama_pshard_strategy_sweep(s, ctx_template, cparams_base, dmds, registry, force_strategy, out, n_tiers, first_tier);
    }
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

void llama_params_fit_pshard_plan(
        const char                              * path_model,
        struct llama_model_params               * mparams,
        struct llama_context_params             * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t                                    max_vram_mb,
        size_t                                    fit_target_mb) {
    const int64_t t0_us = llama_time_us();

    if (!llama_pshard_params_supported(mparams, cparams)) {
        mparams->pshard = false;
        mparams->pshard_delegate_compute = false;
        cparams->pshard = false;
        return;
    }

    mparams->pshard = true;
    mparams->pshard_delegate_compute = false;
    cparams->pshard = true;

    // step 1: probe device memory and model parameters
    std::vector<llama_device> devs;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0, hp_nr = 0;

    llama_model_params mparams_probe = *mparams;
    mparams_probe.pshard = false;
    const auto dmds = llama_get_device_memory_data(
        path_model, &mparams_probe, cparams, devs, hp_ngl, hp_nct, hp_nex, hp_nr, GGML_LOG_LEVEL_ERROR);

    if (devs.empty()) {
        LLAMA_LOG_ERROR("%s: no GPU devices found\n", __func__);
        return;
    }
    if (g_pshard_unsupported_reason) {
        LLAMA_LOG_WARN("%s: %s, disabling pshard\n", __func__, g_pshard_unsupported_reason);
        mparams->pshard = false;
        mparams->pshard_delegate_compute = false;
        cparams->pshard = false;
        return;
    }

    const uint32_t n_layers         = hp_ngl;
    const size_t   mib              = 1024ULL * 1024ULL;
    const size_t   actual_vram_free = dmds[0].free;
    const size_t   fit_target_bytes = fit_target_mb * mib;
    size_t         vram_free        = max_vram_mb > 0
        ? max_vram_mb * mib
        : (actual_vram_free > fit_target_bytes ? actual_vram_free - fit_target_bytes : 0);
    if (g_pshard_extra_device_bytes > 0 && vram_free > g_pshard_extra_device_bytes) {
        // same shrink as the runtime fit: the registry variant is keyed on this budget
        LLAMA_LOG_INFO("%s: reserving %.2f MiB for device memory kept outside the pshard arena (compressor state): budget %.1f -> %.1f MiB\n",
            __func__, g_pshard_extra_device_bytes / (1024.0 * 1024.0),
            vram_free / (1024.0 * 1024.0), (vram_free - g_pshard_extra_device_bytes) / (1024.0 * 1024.0));
        vram_free -= g_pshard_extra_device_bytes;
    }

    mparams->max_vram_alloc = std::max<size_t>(1, pshard_bytes_to_mib_ceil(vram_free));

    LLAMA_LOG_INFO("%s: probing pshard plans: %u layers, %.1f MiB VRAM free, %.1f MiB budget%s\n",
        __func__, n_layers,
        actual_vram_free / (1024.0 * 1024.0),
        vram_free / (1024.0 * 1024.0),
        max_vram_mb > 0 ? " (-mva)" : " (free - fit target)");

    // step 2: read forced strategy from env (PSHARD_STRATEGY)
    const int force_strategy = pshard_strategy_from_env();
    if (force_strategy >= 0) {
        LLAMA_LOG_INFO("%s: forcing strategy %s (PSHARD_STRATEGY=%s)\n",
            __func__, llama_pshard_strategy_name((llama_pshard_strategy)force_strategy),
            getenv("PSHARD_STRATEGY"));
    } else if (getenv("PSHARD_STRATEGY")) {
        LLAMA_LOG_WARN("%s: invalid PSHARD_STRATEGY='%s', ignoring\n",
            __func__, getenv("PSHARD_STRATEGY"));
    }

    // step 3: derive layout, buftypes, optional benchmark predictor, search ctx
    ggml_backend_buffer_type_t gpu_buft  = ggml_backend_dev_buffer_type(devs[0].dev);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(devs[0].dev);
    if (!host_buft) {
        host_buft = ggml_backend_cpu_buffer_type();
    }
    const int32_t           cpu_bid = pshard_dev_layout::compute_cpu_backend_id(devs.size());
    const pshard_dev_layout layout  = pshard_dev_layout::for_device(0, cpu_bid);
    // every variant starts with the MTP head pin-priority placement; the union-budget
    // enforcer flips this (and records it in the registry) only when the pinned head overshoots
    g_pshard_mtp_head_cpu = false;

    std::unique_ptr<llama_benchmark_predictor> predictor;
    {
        const char * env_cpu  = getenv("PSHARD_CPU_PROFILE");
        const char * env_gpu  = getenv("PSHARD_GPU_PROFILE");
        const char * cpu_path = env_cpu ? env_cpu : "cpu_profile.txt";
        const char * gpu_path = env_gpu ? env_gpu : "gpu_profile.txt";

        auto p = std::make_unique<llama_benchmark_predictor>();
        const bool has_cpu = p->load_cpu(cpu_path, cparams->n_threads);
        const bool has_gpu = p->load_gpu(gpu_path);
        if (has_cpu || has_gpu) {
            predictor = std::move(p);
            LLAMA_LOG_INFO("%s: benchmark predictor loaded (cpu=%s gpu=%s)\n",
                __func__, has_cpu ? "yes" : "no", has_gpu ? "yes" : "no");
        }
    }

    const uint32_t n_ctx_plan = cparams->n_ctx > 0 ? cparams->n_ctx : hp_nct;

    llama_pshard_search_ctx ctx = {
        path_model, mparams, cparams, tensor_buft_overrides,
        n_layers, vram_free, gpu_buft, host_buft, layout,
        /*is_moe=*/(hp_nex > 0), /*has_rs=*/(hp_nr > 0),
        predictor.get(), n_ctx_plan, 0,
    };

    // step 4: registry lookup -- try to load <model>.tensor_overrides.pshard_registry; merge cached plans by tier
    const std::string cache_path = std::string(path_model) + ".tensor_overrides.pshard_registry";

    int64_t model_file_size = 0;
    if (FILE * mf = fopen(path_model, "rb")) {
#ifdef _WIN32
        _fseeki64(mf, 0, SEEK_END);
        model_file_size = _ftelli64(mf);
#else
        fseeko(mf, 0, SEEK_END);
        model_file_size = ftello(mf);
#endif
        fclose(mf);
    }
    const uint64_t fp = pshard_registry_fingerprint(mparams, cparams, model_file_size);

    // per-mapping upload pricing: split files past the driver's host pin ceiling
    // cannot be page-locked, and their streamed copies go through the pinned
    // staging ring at a host-DRAM-bound rate. Predict which mappings lock (the
    // runtime registers whole mappings greedily in split order, all-or-nothing)
    // and reprice the predictor's weight-upload rate with the blended value.
    // Also compute the TOTAL file size: the main split of a sharded gguf can be
    // tiny (DeepSeek-V4's is 6 MB), which broke every file-size-derived
    // heuristic downstream (ids-cross expert bytes, switch-cost layer bytes).
    // NOTE: the registry fingerprint above stays on the MAIN split size - the
    // runtime computes it the same way, and changing it would strand every
    // cached registry into silent stock fallback.
    int64_t total_file_size = model_file_size;
    {
        std::vector<int64_t> map_sizes;
        map_sizes.push_back(model_file_size);
        struct gguf_init_params gip_s = { /*.no_alloc =*/ true, /*.ctx =*/ NULL };
        if (struct gguf_context * g = gguf_init_from_file(path_model, gip_s)) {
            int n_split = 0;
            const int64_t ks = gguf_find_key(g, "split.count");
            if (ks >= 0 && gguf_get_kv_type(g, ks) == GGUF_TYPE_UINT16) {
                n_split = (int) gguf_get_val_u16(g, ks);
            }
            gguf_free(g);
            if (n_split > 1) {
                char prefix[1024];
                if (llama_split_prefix(prefix, sizeof(prefix), path_model, 0, n_split) > 0) {
                    for (int idx = 1; idx < n_split; idx++) {
                        char split_path[1024];
                        llama_split_path(split_path, sizeof(split_path), prefix, idx, n_split);
                        if (FILE * sf = fopen(split_path, "rb")) {
#ifdef _WIN32
                            _fseeki64(sf, 0, SEEK_END);
                            map_sizes.push_back(_ftelli64(sf));
#else
                            fseeko(sf, 0, SEEK_END);
                            map_sizes.push_back(ftello(sf));
#endif
                            fclose(sf);
                        } else {
                            LLAMA_LOG_WARN("%s: split %d/%d not readable at %s - "
                                "file-size heuristics fall back to the main split only\n",
                                __func__, idx + 1, n_split, split_path);
                        }
                    }
                } else {
                    LLAMA_LOG_WARN("%s: model path does not match the split naming pattern - "
                        "file-size heuristics fall back to the main split only\n", __func__);
                }
            }
        }
        total_file_size = 0;
        for (int64_t s : map_sizes) {
            total_file_size += s;
        }
        if (predictor && predictor->stats.host_pin_ceiling_gb > 0.0 &&
                predictor->stats.peak_system_bw > 0.0 && predictor->stats.peak_pcie_bw > 0.0) {
            const double pcie_r = predictor->stats.peak_pcie_bw;
            // staging moves every byte three times over host DRAM (mapping read,
            // ring write, DMA read), so the staged rate is a third of DRAM BW
            const double staged_r = std::min(pcie_r, predictor->stats.peak_system_bw / 3.0);
            // pinned allocations made outside the page-lock loop (load staging,
            // the ring itself, the pinned KV shadow) share the driver's ceiling
            double ceiling_b = (predictor->stats.host_pin_ceiling_gb - 2.0) * 1e9;
            double pinned_b = 0.0, staged_b = 0.0;
            for (int64_t s : map_sizes) {
                if ((double) s <= ceiling_b) {   // greedy in split order, all-or-nothing
                    ceiling_b -= (double) s;
                    pinned_b  += (double) s;
                } else {
                    staged_b  += (double) s;
                }
            }
            if (staged_b > 0.0) {
                const double blended = (pinned_b + staged_b) / (pinned_b / pcie_r + staged_b / staged_r);
                predictor->stats.upload_bw          = blended;
                predictor->stats.upload_staged_bw   = staged_r;
                predictor->stats.upload_staged_frac = staged_b / (pinned_b + staged_b);
                LLAMA_LOG_INFO("%s: upload pricing: %.1f GB page-locks, %.1f GB staged (pin ceiling %.0f GB) -> "
                    "pinned %.1f / staged %.1f -> blended %.1f GB/s\n",
                    __func__, pinned_b / 1e9, staged_b / 1e9, predictor->stats.host_pin_ceiling_gb,
                    pcie_r, staged_r, blended);
            }
        }
    }

    // MoE geometry + model size for the per-tier ids-cross decision
    ctx.model_size = total_file_size;
    ctx.n_expert   = hp_nex;
    if (hp_nex > 0) {
        struct gguf_init_params gip = { /*.no_alloc =*/ true, /*.ctx =*/ NULL };
        if (struct gguf_context * g = gguf_init_from_file(path_model, gip)) {
            const int64_t ka = gguf_find_key(g, "general.architecture");
            if (ka >= 0) {
                char key[192];
                snprintf(key, sizeof(key), "%s.expert_used_count", gguf_get_val_str(g, ka));
                const int64_t k = gguf_find_key(g, key);
                if (k >= 0) {
                    ctx.n_expert_used = gguf_get_val_u32(g, k);
                }
            }
            gguf_free(g);
        }
        LLAMA_LOG_INFO("%s: ids-cross inputs: n_expert=%u n_expert_used=%u model_mb=%lld\n",
            __func__, ctx.n_expert, ctx.n_expert_used, (long long)(total_file_size / (1024 * 1024)));
    }

    llama_pshard_plan_registry * registry  = mparams->pshard_registry;
    bool                         needs_probe = true;

    if (registry) {
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
        registry->cache_ubatch = registry->tier_sizes.empty() ? 0 : registry->tier_sizes.back();
        ctx.cache_ubatch = registry->cache_ubatch;

        // save requested tiers before load (load overwrites tier_sizes)
        std::vector<uint32_t> requested_tiers = registry->tier_sizes;

        llama_pshard_plan_registry cached;
        cached.budget_mib = registry->budget_mib;
        cached.cache_ubatch = registry->cache_ubatch;
        if (!mparams->pshard_cache_skip_load &&
            pshard_registry_load(&cached, fp, cache_path.c_str(), host_buft, vram_free, true)) {
            // merge cached plans into registry by matching batch size
            std::unordered_map<uint32_t, llama_pshard_plan> cache_map;
            for (size_t i = 0; i < cached.tier_sizes.size(); i++) {
                cache_map[cached.tier_sizes[i]] = cached.best_plans[i];
            }

            registry->tier_sizes = requested_tiers;
            registry->best_plans.resize(requested_tiers.size());
            registry->pshard_disabled = cached.pshard_disabled;
            registry->baseline_vram_req = cached.baseline_vram_req;
            registry->budget_mib = cached.budget_mib;
            registry->cache_ubatch = cached.cache_ubatch;

            size_t n_hit = 0;
            for (size_t i = 0; i < requested_tiers.size(); i++) {
                auto it = cache_map.find(requested_tiers[i]);
                if (it != cache_map.end()) {
                    registry->best_plans[i] = it->second;
                    n_hit++;
                }
            }

            if (n_hit == requested_tiers.size()) {
                needs_probe = false;
                LLAMA_LOG_INFO("%s: loaded all %zu tiers from exact budget=%u MiB cache variant\n",
                    __func__, n_hit, registry->budget_mib);
            } else if (n_hit > 0) {
                LLAMA_LOG_INFO("%s: loaded %zu/%zu tiers from exact budget=%u MiB cache variant, %zu need probing\n",
                    __func__, n_hit, requested_tiers.size(), registry->budget_mib, requested_tiers.size() - n_hit);
            }
        }

        // common setup for cache hit and miss
    }

    // step 4b: skip pshard when a cached baseline variant fits this budget
    if (registry && registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: cache says baseline %.1f MiB fits this budget (variant budget=%u MiB cache_ubatch=%u), using baseline loading\n",
            __func__, registry->baseline_vram_req / (1024.0 * 1024.0), registry->budget_mib, registry->cache_ubatch);
        mparams->pshard = false;
        mparams->pshard_delegate_compute = false;
        cparams->pshard = false;
        mparams->n_gpu_layers = n_layers + 1;
        tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
        mparams->tensor_buft_overrides = nullptr;

        const int64_t t1_us = llama_time_us();
        LLAMA_LOG_INFO("%s: best strategy: baseline (cached), all %u layers on GPU, took %.2f s\n",
            __func__, n_layers, (t1_us - t0_us) * 1e-6);
        return;
    }

    // step 5: probe tiers largest first and skip pshard when baseline already fits
    if (registry && needs_probe) {
        const size_t n_tiers = registry->tier_sizes.size();

        // step 5a: baseline off-ramp only for global tiers
        static constexpr uint32_t GLOBAL_FIT_MIN_BATCH = 512;
        size_t first_probe_tier = 0;
        size_t min_global_tier = n_tiers;
        const bool run_baseline_offramp = force_strategy < 0;
        if (run_baseline_offramp) {
            for (size_t t = 0; t < n_tiers; t++) {
                if (registry->tier_sizes[t] >= GLOBAL_FIT_MIN_BATCH) {
                    min_global_tier = t;
                    break;
                }
            }

            if (min_global_tier < n_tiers) {
                size_t global_fit_tier = n_tiers;

                for (size_t t = n_tiers; t-- > min_global_tier; ) {
                    llama_pshard_plan baseline_plan;
                    if (pshard_plan_is_baseline_fit(registry->best_plans[t], n_layers, layout.compute)) {
                        LLAMA_LOG_INFO("%s: === tier %zu (bs=%u) [cached global-fit check] ===\n",
                            __func__, t, registry->tier_sizes[t]);
                        baseline_plan = registry->best_plans[t];
                    } else {
                        const llama_context_params * saved = ctx.cparams;
                        llama_context_params cp_tier = *cparams;
                        cp_tier.n_batch  = registry->tier_sizes[t];
                        cp_tier.n_ubatch = cp_tier.n_batch;
                        ctx.cparams = &cp_tier;

                        LLAMA_LOG_INFO("%s: === tier %zu (bs=%u) [global-fit check] ===\n",
                            __func__, t, registry->tier_sizes[t]);
                        baseline_plan = llama_pshard_search_baseline_fit_tier(ctx, dmds);

                        ctx.cparams = saved;
                    }

                    if (pshard_plan_is_baseline_fit(baseline_plan, n_layers, layout.compute)) {
                        registry->best_plans[t] = baseline_plan;
                        global_fit_tier = t;
                        break;
                    }
                }

                if (global_fit_tier < n_tiers) {
                    const llama_pshard_plan & global = registry->best_plans[global_fit_tier];
                    const uint32_t global_ubatch = registry->tier_sizes[global_fit_tier];
                    const uint32_t global_cache_ubatch = std::min(global_ubatch, GLOBAL_FIT_MIN_BATCH);
                    const size_t global_vram_req = global.total_vram_req;

                    registry->pshard_disabled = true;
                    registry->baseline_vram_req = global_vram_req;
                    registry->cache_ubatch = global_cache_ubatch;
                    registry->tier_sizes.clear();
                    registry->best_plans.clear();
                    pshard_registry_save(registry, fp, cache_path.c_str(), host_buft, cparams);

                    mparams->pshard = false;
                    mparams->pshard_delegate_compute = false;
                    cparams->pshard = false;
                    mparams->n_gpu_layers = n_layers + 1;
                    tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
                    mparams->tensor_buft_overrides = nullptr;

                    const int64_t t1_us = llama_time_us();
                    LLAMA_LOG_INFO("%s: full baseline fit at tier %zu (bs=%u, %.1f MiB); using baseline loading with cache_ubatch=%u, took %.2f s\n",
                        __func__, global_fit_tier, global_ubatch,
                        global_vram_req / (1024.0 * 1024.0),
                        global_cache_ubatch, (t1_us - t0_us) * 1e-6);
                    return;
                }
            }

            LLAMA_LOG_INFO("%s: no full-fit global plan found down to bs=%u\n",
                __func__, GLOBAL_FIT_MIN_BATCH);
        } else {
            LLAMA_LOG_INFO("%s: skipping baseline global-fit check for forced strategy %s\n",
                __func__, llama_pshard_strategy_name((llama_pshard_strategy) force_strategy));
        }

        // step 5b: probe remaining tiers
        if (first_probe_tier < n_tiers)
        {
            std::vector<llama_pshard_plan> all_plans(LLAMA_PSHARD_COUNT * n_tiers);
            for (size_t t = 0; t < n_tiers; t++) {
                const llama_pshard_plan & plan = registry->best_plans[t];
                if (plan.is_viable && plan.strategy >= 0 && plan.strategy < LLAMA_PSHARD_COUNT) {
                    all_plans[(int) plan.strategy * n_tiers + t] = plan;
                }
            }

            int n_workers = std::min((int)cparams->n_threads, (int)LLAMA_PSHARD_COUNT);
            n_workers = std::max(n_workers, 1);

            int n_pshard_strategies = LLAMA_PSHARD_COUNT;
            n_workers = std::min(n_workers, n_pshard_strategies);

            LLAMA_LOG_INFO("%s: parallel planning with %d workers (%d pshard strategies)\n",
                __func__, n_workers, n_pshard_strategies);

            std::atomic<int> next_strategy{0};

            if (n_workers <= 1) {
                llama_pshard_parallel_worker(next_strategy, ctx, *cparams, dmds,
                    *registry, force_strategy, all_plans.data(), n_tiers, first_probe_tier);
            } else {
                std::vector<std::thread> threads;
                for (int w = 1; w < n_workers; w++) {
                    threads.emplace_back(llama_pshard_parallel_worker,
                        std::ref(next_strategy), std::cref(ctx), std::cref(*cparams),
                        std::cref(dmds), std::cref(*registry), force_strategy,
                        all_plans.data(), n_tiers, first_probe_tier);
                }
                llama_pshard_parallel_worker(next_strategy, ctx, *cparams, dmds,
                    *registry, force_strategy, all_plans.data(), n_tiers, first_probe_tier);

                for (auto & t : threads) t.join();
            }

            for (size_t t = first_probe_tier; t < n_tiers; t++) {
                llama_pshard_plan best;
                for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
                    auto & p = all_plans[s * n_tiers + t];
                    if (p.is_viable && pshard_plan_is_better(p, best)) {
                        best = p;
                    }
                }
                registry->best_plans[t] = best;
            }

            // step 5d: try attention priority fallback for forced strategies that cannot fit
            // forced strategy sweeps can leave smaller tiers unfilled
            if (force_strategy >= 0 && force_strategy != LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
                for (size_t t = first_probe_tier; t < n_tiers; t++) {
                    if (registry->best_plans[t].is_viable) continue;

                    const llama_context_params * saved = ctx.cparams;
                    llama_context_params cp_tier = *cparams;
                    cp_tier.n_batch  = registry->tier_sizes[t];
                    cp_tier.n_ubatch = cp_tier.n_batch;
                    ctx.cparams = &cp_tier;

                    llama_pshard_plan fallback = llama_pshard_attn_pin_fallback(ctx, force_strategy);
                    if (fallback.is_viable) {
                        registry->best_plans[t] = fallback;
                    }

                    ctx.cparams = saved;
                }
            }
        }

        pshard_enforce_union_budget(registry, ctx, cparams, dmds, force_strategy,
            path_model, mparams);
        pshard_compute_switch_costs(registry, total_file_size, n_layers, ctx.is_moe,
            predictor ? (predictor->stats.upload_bw > 0.0
                ? predictor->stats.upload_bw : predictor->stats.peak_pcie_bw) : 25.0);
        pshard_registry_save(registry, fp, cache_path.c_str(), host_buft, cparams);
    }

    // step 6: pick the active plan
    if (registry) {
        for (size_t t = registry->tier_sizes.size(); t-- > 0; ) {
            auto * best = registry->get_best(t);
            if (best && best->is_viable) {
                registry->active_plan = best;
                break;
            }
        }
    }

    llama_pshard_plan best_plan;
    if (registry && registry->active_plan) {
        best_plan = *registry->active_plan;
    } else {
        llama_pshard_tier_prune prune_single;
        prune_single.init(n_layers);
        best_plan = llama_pshard_search_tier(ctx, force_strategy, dmds, prune_single);
        if (best_plan.is_viable && registry) {
            for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
                if (registry->tier_sizes[t] == best_plan.batch_size) {
                    registry->best_plans[t] = best_plan;
                    registry->active_plan = &registry->best_plans[t];
                    pshard_enforce_union_budget(registry, ctx, cparams, dmds, force_strategy,
                        path_model, mparams);
                    pshard_compute_switch_costs(registry, total_file_size, n_layers, ctx.is_moe,
                        predictor ? (predictor->stats.upload_bw > 0.0
                            ? predictor->stats.upload_bw : predictor->stats.peak_pcie_bw) : 25.0);
                    pshard_registry_save(registry, fp, cache_path.c_str(), host_buft, cparams);
                    break;
                }
            }
        }
    }

    // step 7: fall back to all cpu when no plan is viable
    if (!best_plan.is_viable) {
        LLAMA_LOG_WARN("%s: no viable plan found, falling back to STATIC_ATTNPRIO_ALLMODELS with n_pinned=0\n", __func__);
        llama_pshard_generate_overrides(0, n_layers, gpu_buft, host_buft, tensor_buft_overrides,
            LLAMA_LAYER_FRACTION_NONE, LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS, layout,
            /*pin_from_back=*/false, /*output_on_gpu=*/false, /*n_attn_pinned=*/0);
        mparams->pshard_delegate_compute = true;
        mparams->n_gpu_layers = n_layers + 1;
        mparams->tensor_buft_overrides = tensor_buft_overrides;
        return;
    }

    // step 8: apply best plan to tensor_buft_overrides
    llama_pshard_generate_overrides(best_plan.n_pinned, n_layers, gpu_buft, host_buft,
        tensor_buft_overrides, (llama_layer_fraction)best_plan.overflow, best_plan.strategy, layout,
        best_plan.pin_from_back, best_plan.output_on_gpu, best_plan.n_attn_pinned, best_plan.overlap, best_plan.ids_cross);

    for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
        if (tensor_buft_overrides[i].backend_id == layout.compute) {
            tensor_buft_overrides[i].buft = host_buft;
        }
    }

    mparams->pshard_delegate_compute = llama_pshard_strategy_delegates_compute(best_plan.strategy);
    mparams->n_gpu_layers = n_layers + 1;
    mparams->tensor_buft_overrides = tensor_buft_overrides;

    const int64_t t1_us = llama_time_us();
    LLAMA_LOG_INFO("%s: best strategy: %s, n_pinned=%u, n_attn=%u/%u%s%s, took %.2f s\n",
        __func__, llama_pshard_strategy_name(best_plan.strategy),
        best_plan.n_pinned, best_plan.n_attn_pinned, n_layers,
        best_plan.overflow ? " (partial: " : "",
        best_plan.overflow ? PSHARD_FRAC_NAMES[best_plan.overflow] : "",
        (t1_us - t0_us) * 1e-6);

    {
        int n_bid0 = 0, n_total = 0;
        for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
            n_total++;
            if (ov->backend_id == layout.compute) n_bid0++;
            LLAMA_LOG_DEBUG("%s:   override: %-25s -> %-15s backend_id=%d\n",
                __func__, ov->pattern, ggml_backend_buft_name(ov->buft), ov->backend_id);
        }
        LLAMA_LOG_INFO("%s: %d overrides, %d with bid=%d (compute)\n",
            __func__, n_total, n_bid0, layout.compute);
    }
}
