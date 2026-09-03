#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

enum llama_pshard_strategy {
    LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM        = 0,
    LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM           = 1,
    LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM           = 2,
    LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS           = 3,
    // unpinned FFNs alternate CPU compute / GPU streaming: the PCIe copy of the next streamed
    // FFN overlaps CPU-FFN + pinned-attn compute, so DDR and PCIe bandwidth add up
    LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE               = 4,
    // routed experts become a managed VRAM cache (docs/expert-pool-design.md):
    // per-tier variables n_attn_pinned / K / miss_policy / prefill_mode; the
    // pool region serves prefill A/B streaming and decode LRU slots
    LLAMA_PSHARD_EXPERT_POOL                         = 5,
    LLAMA_PSHARD_COUNT
};

// strategy name for logging
inline const char * llama_pshard_strategy_name(llama_pshard_strategy s) {
    switch (s) {
        case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM: return "GPUONLY_LAYERPIN_LAYERSTREAM";
        case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:    return "GPUONLY_ATTNPIN_FFNSTREAM";
        case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:    return "DYNAMIC_FFNCPU_ATTNSTREAM";
        case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:    return "STATIC_ATTNPRIO_ALLMODELS";
        case LLAMA_PSHARD_DYNAMIC_FFN_ALTERNATE:        return "DYNAMIC_FFN_ALTERNATE";
        case LLAMA_PSHARD_EXPERT_POOL:                  return "EXPERT_POOL";
        default:                                        return "UNKNOWN";
    }
}

inline bool llama_pshard_strategy_delegates_compute(llama_pshard_strategy s) {
    return s == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
}

// PSHARD_STRATEGY accepts a name or numeric id
inline int pshard_strategy_from_env() {
    const char * env = getenv("PSHARD_STRATEGY");
    if (!env || !*env) return -1;
    for (int i = 0; i < LLAMA_PSHARD_COUNT; i++) {
        if (strcmp(env, llama_pshard_strategy_name((llama_pshard_strategy)i)) == 0) {
            return i;
        }
    }
    char * end = nullptr;
    long v = strtol(env, &end, 10);
    if (end != env && *end == '\0' && v >= 0 && v < LLAMA_PSHARD_COUNT) {
        return (int)v;
    }
    return -1;
}

// EXPERT_POOL: what a cache miss does at decode / small batch (per-tier field,
// all values planner-priced - no per-strategy allowed-sets)
enum llama_pshard_miss_policy {
    LLAMA_PSHARD_MISS_FETCH        = 0,  // copy the expert RAM -> LRU victim slot, compute on GPU
    LLAMA_PSHARD_MISS_CPU_EXEC     = 1,  // compute the miss on CPU from host weights (split-op)
    LLAMA_PSHARD_MISS_FETCH_ON_2ND = 2,  // first miss cpu_exec (no admit), repeat miss fetches
    LLAMA_PSHARD_MISS_HYBRID       = 3,  // q* split: fetch m*B_P/B_H by recency, CPU runs the rest
    LLAMA_PSHARD_MISS_COUNT
};

// EXPERT_POOL: how a prefill tier moves an unpinned layer's experts
enum llama_pshard_prefill_mode {
    LLAMA_PSHARD_PREFILL_AB_STREAM = 0,  // whole expert set through the A/B span, hidden under GEMMs
    LLAMA_PSHARD_PREFILL_CPU_TAIL  = 1,  // ab_stream + the ubatch's coldest experts computed on CPU
    LLAMA_PSHARD_PREFILL_COUNT
};

inline const char * llama_pshard_miss_policy_name(llama_pshard_miss_policy p) {
    switch (p) {
        case LLAMA_PSHARD_MISS_FETCH:        return "fetch";
        case LLAMA_PSHARD_MISS_CPU_EXEC:     return "cpu_exec";
        case LLAMA_PSHARD_MISS_FETCH_ON_2ND: return "fetch_on_2nd_miss";
        case LLAMA_PSHARD_MISS_HYBRID:       return "hybrid";
        default:                             return "unknown";
    }
}

inline const char * llama_pshard_prefill_mode_name(llama_pshard_prefill_mode m) {
    switch (m) {
        case LLAMA_PSHARD_PREFILL_AB_STREAM: return "ab_stream";
        case LLAMA_PSHARD_PREFILL_CPU_TAIL:  return "cpu_tail";
        default:                             return "unknown";
    }
}

// parse a policy/mode name at the head of s (registry token: delimited by
// space / eol). "fetch" is a prefix of "fetch_on_2nd_miss", so require the
// terminator, not just the prefix.
inline int pshard_miss_policy_from_name(const char * s) {
    for (int i = LLAMA_PSHARD_MISS_COUNT - 1; i >= 0; i--) {
        const char * n = llama_pshard_miss_policy_name((llama_pshard_miss_policy)i);
        const size_t l = strlen(n);
        if (strncmp(s, n, l) == 0 && (s[l] == '\0' || s[l] == ' ' || s[l] == '\n' || s[l] == '\r')) {
            return i;
        }
    }
    return 0;
}

inline int pshard_prefill_mode_from_name(const char * s) {
    for (int i = LLAMA_PSHARD_PREFILL_COUNT - 1; i >= 0; i--) {
        const char * n = llama_pshard_prefill_mode_name((llama_pshard_prefill_mode)i);
        const size_t l = strlen(n);
        if (strncmp(s, n, l) == 0 && (s[l] == '\0' || s[l] == ' ' || s[l] == '\n' || s[l] == '\r')) {
            return i;
        }
    }
    return 0;
}

// PSHARD_MISS_POLICY accepts a name or numeric id (QA override for EXPERT_POOL
// tiers, same pattern as PSHARD_STRATEGY); "second" = fetch_on_2nd_miss alias.
// Mixed into the registry fingerprint only when set.
inline int pshard_miss_policy_from_env() {
    const char * env = getenv("PSHARD_MISS_POLICY");
    if (!env || !*env) return -1;
    if (strcmp(env, "second") == 0) return LLAMA_PSHARD_MISS_FETCH_ON_2ND;
    for (int i = 0; i < LLAMA_PSHARD_MISS_COUNT; i++) {
        if (strcmp(env, llama_pshard_miss_policy_name((llama_pshard_miss_policy)i)) == 0) {
            return i;
        }
    }
    char * end = nullptr;
    long v = strtol(env, &end, 10);
    if (end != env && *end == '\0' && v >= 0 && v < LLAMA_PSHARD_MISS_COUNT) {
        return (int)v;
    }
    return -1;
}

// cached tensor override entry
// trailing nextn/MTP layers (0 when not load_mtp): the stock-sched MTP draft
// context reads these weights concurrently with the target's decode, so the
// emitters must never place them in streamed slots (slot bytes are rewritten
// under the reader -> CUDA launch failures). Set by the planning/apply entry
// points; single-threaded within a planning pass.
inline thread_local uint32_t g_pshard_n_layers_mtp = 0;
// MTP head placement lever: false = pin-priority (head on the compute GPU whenever the
// plan pins anything), true = head CPU-resident (the union-budget enforcer flips this
// variant-wide when the pinned head overshoots; persisted in the registry header)
inline thread_local bool g_pshard_mtp_head_cpu = false;

// architecture support gate: both model probes (runtime cache loader and planner) set this
// from the loaded model; nullptr = supported. pshard refuses LOUDLY (WARN + stock fallback)
// rather than run a memory layout it cannot stream. No architecture is refused today:
// DeepSeek-V4 (llama_kv_cache_dsv4) was the case until 2026-09-01 - its wrapper hid its pipe
// shards, the pshard cache constructor skipped the attention-rotation tail (so the compressed
// attention + lightning indexer were silently never built), and the scheduler let streamed
// layers read views of host weights directly. Keep the gate for the next such architecture.
inline thread_local const char * g_pshard_unsupported_reason = nullptr;

// device bytes the model's memory keeps OUTSIDE the pshard arena (today: DeepSeek-V4's
// compressor-state tensors). Both probes set this; the runtime fit and the planner shrink
// the arena budget by it so arena + extra == the user's -mva (the registry variant is keyed
// on the shrunken budget, which both sides derive identically).
inline thread_local size_t g_pshard_extra_device_bytes = 0;
size_t llama_pshard_extra_device_bytes(const llama_model & model, uint32_t n_seq_max, uint32_t n_rs_seq);

inline const char * llama_pshard_arch_unsupported(const llama_model & model) {
    // PSHARD_ALLOW_UNSUPPORTED=1: development lever - run a refused architecture anyway
    static const bool allow = getenv("PSHARD_ALLOW_UNSUPPORTED") != nullptr && getenv("PSHARD_ALLOW_UNSUPPORTED")[0] == '1';
    if (allow) {
        return nullptr;
    }
    (void) model;
    return nullptr;
}

struct llama_pshard_override {
    std::string                pattern;
    ggml_backend_buffer_type_t buft;
    int32_t                    backend_id;
};

// saved allocator and backend ids for plan switches
struct llama_pshard_alloc_state {
    std::vector<uint8_t> node_allocs;
    std::vector<uint8_t> leaf_allocs;
    std::vector<int>     node_backend_ids;
    std::vector<int>     leaf_backend_ids;
    int  n_nodes = 0;
    int  n_leafs = 0;
    bool valid   = false;
};

struct llama_pshard_plan {
    llama_pshard_strategy strategy       = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
    uint32_t             batch_size      = 0;
    uint32_t             n_pinned        = 0;   // fully pinned layers (all tensors on GPU)
    uint32_t             n_attn_pinned   = 0;   // attention priority layers on GPU (>= n_pinned)
    int                  overflow        = 0;   // llama_layer_fraction
    bool                 pin_from_back   = false;
    bool                 output_on_gpu   = false;
    bool                 overlap         = true;   // transport mode: double-buffer slots + prefetch scan-ahead
    bool                 ids_cross       = false;  // ALTERNATE only: pin routers on the compute GPU so
                                                   // expert ids cross a split boundary -> sliced uploads

    // EXPERT_POOL per-tier variables (docs/expert-pool-design.md 3b.2); legacy
    // strategies leave them at their defaults and never serialize them
    uint32_t pool_k           = 0;     // layers whose experts are all resident (s_l = E, eviction off)
    uint32_t pool_slots       = 0;     // pool slots per unpinned layer (v1 uniform s, from pool_mb)
    int      pool_miss        = 0;     // llama_pshard_miss_policy
    int      pool_prefill     = 0;     // llama_pshard_prefill_mode
    float    pool_hybrid_frac = 0.0f;  // fetched share of misses under hybrid (B_P / B_H)

    std::vector<llama_pshard_override> overrides;

    size_t total_vram_req   = 0;
    size_t scratch_measured = 0;
    size_t cache_measured   = 0;
    float  tps              = 0.0f;  // predicted tokens/sec (0 = no benchmark data)
    float  switch_ms        = 0.0f;  // est. one-way cost of switching into this plan from
                                     // the decode (tier 0) plan: pinned-residency delta / PCIe
    bool   is_viable        = false;

    // cached maps and offsets from first apply
    mutable std::unordered_map<std::string, int32_t> cached_tensor_bids;
    mutable std::unordered_map<int, int32_t>         cached_layer_bids;
    mutable std::unordered_map<std::string, size_t>  cached_weight_offsets;
    mutable size_t cached_scratch_off = 0;
    mutable bool   maps_cached       = false;
    mutable bool   addrs_cached      = false;

    mutable llama_pshard_alloc_state alloc_state;
};

enum llama_layer_fraction {
    LLAMA_LAYER_FRACTION_NONE = 0,
    LLAMA_LAYER_FRACTION_ATTN = 1,
    LLAMA_LAYER_FRACTION_UP   = 2,
    LLAMA_LAYER_FRACTION_GATE = 3,
    LLAMA_LAYER_FRACTION_MOE  = 4,
};

const char * llama_get_overflow_pattern(size_t il, llama_layer_fraction lf);

void llama_pshard_generate_overrides(
        uint32_t n_pinned,
        uint32_t n_layers,
        ggml_backend_buffer_type_t gpu_buft,
        ggml_backend_buffer_type_t host_buft,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        llama_layer_fraction overflow_type,
        llama_pshard_strategy strategy,
        const pshard_dev_layout & layout,
        bool pin_from_back = false,
        bool output_on_gpu = false,
        uint32_t n_attn_pinned = 0,
        bool overlap = true,
        bool ids_cross = false);

// llama_device_memory_data and llama_memory_breakdown_data come from
// ToT's src/llama-ext.h (included above)

// probe hook runs before context teardown
// used by TPS prediction to inspect scheduler splits
typedef void (*llama_probe_hook_t)(llama_context * ctx, void * user_data);

std::vector<llama_device_memory_data> llama_get_device_memory_data(
        const char * path_model, const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        std::vector<llama_device> & devs, uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert, uint32_t & hp_n_embd_r,
        enum ggml_log_level log_level,
        llama_probe_hook_t probe_hook = nullptr,
        void * probe_hook_data = nullptr,
        uint32_t probe_n_tokens = 0,
        uint32_t probe_n_outputs = 0);

// fit params entry point used by pshard planning (relocated from the
// base-era src/llama.cpp; upstream's generic fit moved to common/fit)
void llama_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level);

// plan cache serialization. fingerprint covers only runtime plan-compatibility params
// so the planner binary and the runtime binary can share the same cache file.
// keep this in sync with planner save
uint64_t pshard_registry_fingerprint(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        int64_t model_file_size);

bool pshard_registry_save(
        const struct llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        const struct llama_context_params * cparams = nullptr);

bool pshard_registry_load(
        struct llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        size_t current_budget, bool require_exact_budget = false);

struct llama_pshard_plan_registry {
    std::vector<uint32_t>                tier_sizes;
    std::vector<llama_pshard_plan>       best_plans;  // one best plan per tier
    llama_pshard_plan *                  active_plan = nullptr;
    uint32_t                             budget_mib = 0;
    uint32_t                             cache_ubatch = 0;

    // switch-cost estimate constants (written by the planner; 0 = not available).
    // Residency-switch cost is a pure function of two plans' pin fields, so it is
    // evaluated pairwise on demand rather than precomputed against one anchor plan.
    float switch_layer_mb  = 0.0f;  // est. weight MB of one full layer
    float switch_attn_frac = 0.0f;  // attention share of a layer's bytes
    float switch_head_mb   = 0.0f;  // est. MB of the output head
    float switch_pcie_gb_s = 0.0f;  // upload rate for pinned weights
    bool  mtp_head_cpu     = false; // MTP head demoted to CPU by union-budget enforcement
    uint32_t n_layers      = 0;     // trunk layer count (set in-memory by planner and runtime;
                                    // 0 = unknown -> structural attention pins are not priced)
    // canonical union of every viable tier (packed weights + pinned cache + compute scratch +
    // enforcer margin), the largest device footprint any plan of this variant needs. Written
    // by the planner's union enforcer, persisted as union_mb=. 0 = unknown (older registry,
    // or the enforcer did not converge) -> the arena takes the whole budget as before.
    size_t union_bytes     = 0;

    // device bytes the pshard arena is allocated with: the union plus a packing headroom
    // (the runtime's canonical packing rounds differently from the planner's metadata pass by
    // up to a few tens of MiB), capped at the budget. Whatever the budget has beyond that is
    // LEFTOVER: real device memory nobody uses, which the one-budget rule hands to a spilled
    // speculative draft's experts (common_init_from_params) instead of idling inside the arena.
    size_t arena_bytes(size_t budget_bytes) const {
        const size_t mib      = 1024ULL * 1024;
        const size_t headroom = 64 * mib;
        if (union_bytes == 0) {
            return budget_bytes;
        }
        // whole MiB: the buffer size positions the pinned cache region (buf_total - cache),
        // so a fractional size would misalign every cache tensor (CUDA misaligned address)
        const size_t want = ((union_bytes + headroom + mib - 1) / mib) * mib;
        return want >= budget_bytes ? budget_bytes : want;
    }

    // layers whose ATTENTION is device-resident under a plan. n_attn_pinned is the
    // ATTNPRIO/ALTERNATE budget knob and stays 0 for strategies that pin attention
    // structurally: ATTNPIN_FFNSTREAM keeps every layer's attention resident, the
    // LAYERSTREAM/FFNCPU_ATTNSTREAM strategies only the fully pinned layers'. Pricing a
    // switch from the raw field charged an ATTNPIN <-> ATTNPRIO(attn=40) swap as 40
    // layers of attention traffic that never moves (selector-gap audit caveat, 2026-08-31).
    uint32_t attn_resident(const llama_pshard_plan & p) const {
        uint32_t r = p.n_attn_pinned > p.n_pinned ? p.n_attn_pinned : p.n_pinned;
        if (p.strategy == LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM && n_layers > 0) {
            r = n_layers > r ? n_layers : r;
        }
        return r;
    }

    // one-way cost of switching pinned residency between two plans, in ms.
    // Falls back to the tier0-anchored per-plan estimate for legacy caches.
    float switch_cost_ms(const llama_pshard_plan & from, const llama_pshard_plan & to) const {
        if (switch_layer_mb <= 0.0f || switch_pcie_gb_s <= 0.0f) {
            return to.switch_ms;
        }
        double mb = 0.0;
        // fully pinned layers: nested sets when pinned from the same end, disjoint otherwise
        if (from.pin_from_back == to.pin_from_back) {
            mb += (to.n_pinned > from.n_pinned ? to.n_pinned - from.n_pinned
                                               : from.n_pinned - to.n_pinned) * (double)switch_layer_mb;
        } else {
            mb += ((double)to.n_pinned + (double)from.n_pinned) * (double)switch_layer_mb;
        }
        // attention-only pins: resident attention beyond the fully pinned layers (structural
        // pins included, see attn_resident)
        const double fa = (double)(attn_resident(from) - from.n_pinned);
        const double ta = (double)(attn_resident(to)   - to.n_pinned);
        mb += (ta > fa ? ta - fa : fa - ta) * (double)switch_layer_mb * (double)switch_attn_frac;
        if (from.output_on_gpu != to.output_on_gpu) {
            mb += (double)switch_head_mb;
        }
        return (float)(mb / (double)switch_pcie_gb_s);  // MB / (GB/s) == ms
    }

    // variant marker for a baseline load that fits
    // runtime still checks baseline_vram_req against the current budget
    bool                                 pshard_disabled = false;
    size_t                               baseline_vram_req = 0;

    void init(uint32_t n_ubatch, uint32_t n_parallel = 1, uint32_t n_draft = 0) {
        tier_sizes.clear();
        best_plans.clear();

        if (n_ubatch == 0) {
            cache_ubatch = 0;
            return;
        }
        best_plans.clear();

        if (n_ubatch == 0) {
            cache_ubatch = 0;
            return;
        }

        // decode tiers
        if (n_parallel <= 1) {
            tier_sizes.push_back(1);
            // speculative verify batches (n_draft+1, realistically 3-9) get their own
            // exactly-priced tier: the ceiling tier pick would otherwise execute them
            // on the bs=16 plan, whose placement was priced for 16 independent tokens
            const uint32_t verify_tier = n_draft > 0 ? n_draft + 1 : 0;
            if (verify_tier > 1 && verify_tier < 16) {
                tier_sizes.push_back(verify_tier);
            }
            tier_sizes.push_back(16);
            if (verify_tier > 16) {
                tier_sizes.push_back(verify_tier);
            }
        } else {
            for (uint32_t t = 1; t <= 64 && t < 512; t *= 4) {
                tier_sizes.push_back(t);
                if (t == 16) {
                    tier_sizes.push_back(32);
                }
            }
        }

        // prefill tiers: x2 growth from 512
        for (uint32_t t = 512; t < n_ubatch; t *= 2) {
            if (tier_sizes.empty() || tier_sizes.back() < t) {
                tier_sizes.push_back(t);
            }
        }

        if (tier_sizes.empty() || tier_sizes.back() != n_ubatch) {
            tier_sizes.push_back(n_ubatch);
        }

        cache_ubatch = tier_sizes.empty() ? 0 : tier_sizes.back();
        best_plans.resize(tier_sizes.size());
    }

    size_t tier_index(uint32_t batch_size) const {
        for (size_t i = 0; i < tier_sizes.size(); i++) {
            if (tier_sizes[i] >= batch_size) return i;
        }
        return tier_sizes.size() - 1;
    }

    llama_pshard_plan * get_best(size_t tier) {
        return best_plans[tier].is_viable ? &best_plans[tier] : nullptr;
    }

    // pick the prefill ubatch with the lowest predicted ttft
    // use max_ubatch when TPS data is missing
    uint32_t find_optimal_ubatch(uint32_t n_prompt, uint32_t max_ubatch,
                                 const llama_pshard_plan * from_plan = nullptr) const {
        // QA/debug override: evaluation shape changes numerics on shape-sensitive models
        // (gpt-oss raw-text PPL), so A/B runs need a way to pin the prefill ubatch
        if (const char * force = getenv("PSHARD_FORCE_PREFILL_UB")) {
            const long v = atol(force);
            if (v > 0) {
                return std::min<uint32_t>((uint32_t) v, max_ubatch);
            }
        }
        uint32_t best_ub  = max_ubatch;
        double   best_time = 1e30;

        // switches are pairwise: prefill leaves whatever plan is CURRENTLY active
        // (usually - but not always - the decode plan) and returns to the decode plan
        const llama_pshard_plan * decode_plan =
            (!best_plans.empty() && best_plans[0].is_viable) ? &best_plans[0] : nullptr;
        if (from_plan == nullptr) {
            from_plan = decode_plan;
        }

        for (size_t t = 0; t < tier_sizes.size(); t++) {
            uint32_t ts = tier_sizes[t];
            if (ts < 512 || ts > max_ubatch) continue;

            const auto & plan = best_plans[t];
            if (!plan.is_viable || plan.tps <= 0.0f) continue;

            double per_iter = (double)ts / (double)plan.tps;
            uint32_t n_iters = (n_prompt + ts - 1) / ts;
            // TTFT includes switching INTO this tier's plan from the active one and back
            // to the decode plan after prefill; a tier sharing that residency wins ties
            // against one that swaps pinned weights around the prompt
            double switch_total_ms = 0.0;
            if (from_plan != nullptr) {
                switch_total_ms += (double)switch_cost_ms(*from_plan, plan);
            }
            if (decode_plan != nullptr) {
                switch_total_ms += (double)switch_cost_ms(plan, *decode_plan);
            }
            double total = n_iters * per_iter + switch_total_ms / 1000.0;

            if (total < best_time) {
                best_time = total;
                best_ub   = ts;
            }
        }

        return best_ub;
    }
};
