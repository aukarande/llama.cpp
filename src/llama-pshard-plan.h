#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

enum llama_pshard_strategy {
    LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM        = 0,
    LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS           = 1,
    // attention pinned first, the rest streamed to the GPU; the unpinned FFNs split between CPU compute and GPU
    // streaming by the fraction that balances the CPU chain against the copy chain (0 = every FFN on the CPU)
    LLAMA_PSHARD_HYBRID_ATTNPRIO_FFNBALANCE          = 2,
    // routed experts become a managed VRAM cache (docs/expert-pool-design.md):
    // per-tier variables n_attn_pinned / K / miss_policy / prefill_mode; the
    // pool region serves the prefill ab_stream double buffer and decode LRU slots
    LLAMA_PSHARD_EXPERT_POOL                         = 3,
    LLAMA_PSHARD_COUNT
};

// strategy name for logging
inline const char * llama_pshard_strategy_name(llama_pshard_strategy s) {
    switch (s) {
        case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM: return "GPUONLY_LAYERPIN_LAYERSTREAM";
        case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:    return "STATIC_ATTNPRIO_ALLMODELS";
        case LLAMA_PSHARD_HYBRID_ATTNPRIO_FFNBALANCE:   return "HYBRID_ATTNPRIO_FFNBALANCE";
        case LLAMA_PSHARD_EXPERT_POOL:                  return "EXPERT_POOL";
        default:                                        return "UNKNOWN";
    }
}

inline bool llama_pshard_strategy_delegates_compute(llama_pshard_strategy s) {
    return s == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
}

// PSHARD_STRATEGY=ALL: the auto search includes the expert pool; unset = every strategy but the expert pool
constexpr int PSHARD_STRATEGY_ALL = -2;

// host memory the runtime pins outside the loader's page-lock loop (load staging, the
// staging ring, the pinned KV shadow, the CPU chain's landing buffers). The loader stops
// this far short of the driver's ceiling and the planner prices the staged share with the
// same budget
constexpr double LLAMA_PSHARD_HOST_PIN_RESERVE_GB = 2.0;

// PSHARD_STRATEGY accepts a name, a numeric id or ALL; -1 when unset or invalid
inline int pshard_strategy_from_env() {
    const char * env = getenv("PSHARD_STRATEGY");
    if (!env || !*env) return -1;
    if (strcmp(env, "ALL") == 0) return PSHARD_STRATEGY_ALL;
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

// whether the search may consider strategy s under the PSHARD_STRATEGY setting:
// a forced strategy alone, the expert pool only when forced or with ALL
inline bool pshard_strategy_allowed(int force_strategy, int s) {
    if (force_strategy >= 0) return force_strategy == s;
    if (s == LLAMA_PSHARD_EXPERT_POOL) return force_strategy == PSHARD_STRATEGY_ALL;
    return true;
}

// PSHARD_STRATEGY setting for logs and the registry header
inline const char * pshard_forced_strategy_name(int force_strategy) {
    if (force_strategy == PSHARD_STRATEGY_ALL) return "ALL";
    return force_strategy >= 0 ? llama_pshard_strategy_name((llama_pshard_strategy) force_strategy) : "auto";
}

// EXPERT_POOL: what a cache miss does at decode / small batch (per-tier field,
// all values planner-priced - no per-strategy allowed-sets)
enum llama_pshard_miss_policy {
    LLAMA_PSHARD_MISS_FETCH        = 0,  // copy the expert RAM -> LRU victim slot, compute on GPU
    LLAMA_PSHARD_MISS_CPU_EXEC     = 1,  // compute the miss on CPU from host weights (split-op)
    LLAMA_PSHARD_MISS_FETCH_ON_2ND = 2,  // first miss cpu_exec (no admit), repeat miss fetches
    LLAMA_PSHARD_MISS_HYBRID       = 3,  // q* split: fetch m*B_P/B_H by recency, CPU runs the rest
    LLAMA_PSHARD_MISS_CPU_ADMIT    = 4,  // every miss computes on CPU THIS pass while its rows upload
                                         // on the copy stream into a slot (background admission):
                                         // a GPU hit from the next pass on (RFC #24528 execution)
    LLAMA_PSHARD_MISS_COUNT
};

// EXPERT_POOL: how a prefill tier moves an unpinned layer's experts
enum llama_pshard_prefill_mode {
    LLAMA_PSHARD_PREFILL_AB_STREAM = 0,  // whole expert set streamed through the double-buffered span, hidden under GEMMs
    LLAMA_PSHARD_PREFILL_CPU_TAIL  = 1,  // ab_stream + the ubatch's coldest experts computed on CPU
    LLAMA_PSHARD_PREFILL_COUNT
};

inline const char * llama_pshard_miss_policy_name(llama_pshard_miss_policy p) {
    switch (p) {
        case LLAMA_PSHARD_MISS_FETCH:        return "fetch";
        case LLAMA_PSHARD_MISS_CPU_EXEC:     return "cpu_exec";
        case LLAMA_PSHARD_MISS_FETCH_ON_2ND: return "fetch_on_2nd_miss";
        case LLAMA_PSHARD_MISS_HYBRID:       return "hybrid";
        case LLAMA_PSHARD_MISS_CPU_ADMIT:    return "cpu_admit";
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
// rather than run a memory layout it cannot stream. No architecture is refused today; the gate
// stays for the next memory wrapper that hides its pipe shards from the pshard cache constructor
// or lets streamed layers read views of host weights directly.
inline thread_local const char * g_pshard_unsupported_reason = nullptr;

// device bytes the model's memory keeps OUTSIDE the pshard arena (today: DeepSeek-V4's
// compressor-state tensors). Both probes set this; the runtime fit and the planner shrink
// the arena budget by it so arena + extra == the user's -mva (the registry variant is keyed
// on the shrunken budget, which both sides derive identically).
inline thread_local size_t g_pshard_extra_device_bytes = 0;
size_t llama_pshard_extra_device_bytes(const llama_model & model, uint32_t n_seq_max, uint32_t n_rs_seq);

inline const char * llama_pshard_arch_unsupported(const llama_model & model) {
    // no architecture is refused today; the callers keep the hook
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
    // transport: 0 one slot, no prefetch; 1 two slots, prefetch scan-ahead; 2 one live streamed shard, prefetched
    // only from splits that stream nothing (ggml_backend_sched_set_prefetch_window)
    uint8_t              overlap         = 1;
    bool                 ids_cross       = false;  // HYBRID_ATTNPRIO_FFNBALANCE only: pin routers on the compute GPU so
                                                   // expert ids cross a split boundary -> sliced uploads

    // EXPERT_POOL per-tier variables (docs/expert-pool-design.md 3b.2); legacy
    // strategies leave them at their defaults and never serialize them
    uint32_t pool_k           = 0;     // layers whose experts are all resident (s_l = E, eviction off)
    uint32_t pool_slots       = 0;     // pool slots per unpinned layer (v1 uniform s, from pool_mb)
    int      pool_miss        = 0;     // llama_pshard_miss_policy
    int      pool_prefill     = 0;     // llama_pshard_prefill_mode
    float    pool_hybrid_frac = 0.0f;  // fetched share of misses under hybrid (B_P / B_H)
    // HYBRID_ATTNPRIO_FFNBALANCE: how many of the unpinned FFNs stream to the GPU (the rest compute on the CPU)
    uint32_t n_ffn_gpu        = 0;

    std::vector<llama_pshard_override> overrides;

    size_t total_vram_req   = 0;
    size_t scratch_measured = 0;
    size_t cache_measured   = 0;
    float  tps              = 0.0f;  // predicted tokens/sec (0 = no benchmark data)
    float  switch_ms        = 0.0f;  // est. cost of switching into this plan from the decode
                                     // (tier 0) plan, see llama_pshard_plan_registry::switch_cost_ms
    float  cold_ms          = 0.0f;  // EXPERT_POOL cache tier: est. decode ms its misses cost above the
                                     // warm rate after it lands with empty slots (0 = not a cache tier)
    bool   is_viable        = false;

    // ubatches below this many tokens copy only the used experts of a layer whose ids an earlier split computed,
    // larger ones prefetch the whole tensors (ggml_backend_sched_set_expert_slice_tokens); -1 = the scheduler's rule
    int32_t expert_slice_tokens = -1;

    // predicted ms of one ubatch of piece_n[i] tokens on this plan (ascending, below batch_size)
    std::vector<uint32_t> piece_n;
    std::vector<float>    piece_ms;

    // predicted ms of one ubatch of n tokens: the piece curve, ending at batch_size tokens at the tier's rate
    // (linear between points); without a curve, n tokens at the tier's rate
    double ubatch_ms(uint32_t n) const {
        if (tps <= 0.0f) {
            return 0.0;
        }
        const double full_ms = (double) batch_size * 1000.0 / (double) tps;
        if (piece_n.empty() || n >= batch_size) {
            return n == batch_size ? full_ms : (double) n * 1000.0 / (double) tps;
        }
        uint32_t x0 = piece_n[0];
        double   y0 = piece_ms[0];
        if (n <= x0) {
            return y0;
        }
        for (size_t i = 1; i <= piece_n.size(); i++) {
            const uint32_t x1 = i < piece_n.size() ? piece_n[i]  : batch_size;
            const double   y1 = i < piece_n.size() ? piece_ms[i] : full_ms;
            if (n <= x1) {
                return y0 + (y1 - y0) * (double) (n - x0) / (double) (x1 - x0);
            }
            x0 = x1;
            y0 = y1;
        }
        return full_ms;
    }

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
        bool ids_cross = false,
        uint32_t n_ffn_gpu = 0);

// HYBRID_ATTNPRIO_FFNBALANCE: whether the k-th of n_ffn unpinned layers streams its FFN to the GPU when n_gpu of them
// do; the GPU ones are spread evenly so each copy overlaps CPU work on the layers around it
inline bool llama_pshard_hybrid_ffn_on_gpu(uint32_t k, uint32_t n_gpu, uint32_t n_ffn) {
    return n_ffn > 0 && (uint64_t) (k + 1) * n_gpu / n_ffn > (uint64_t) k * n_gpu / n_ffn;
}

// llama_device_memory_data and llama_memory_breakdown_data come from
// ToT's src/llama-ext.h (included above)

struct llama_model_meta_cache;

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
        uint32_t probe_n_outputs = 0,
        int32_t  probe_expert_slice_tokens = -1,
        bool     probe_prefetch_window = false,
        struct llama_model_meta_cache * meta_cache = nullptr);

// fit params entry point used by pshard planning; upstream's generic fit lives in common/fit
void llama_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level);

struct llama_pshard_workload;

// share of a layer's routes that a pool of pool_slots slots misses: 1 minus the mass of its
// pool_slots most routed experts in the workload histogram. 1.0 for a layer without a
// histogram, or without a pool (every routed expert of a streamed layer is uploaded). The
// loader ranks the expert stacks it page-locks by it (copies per token) and the planner
// prices the page-lock ceiling with it, so the two agree on which bytes stay pageable
std::vector<double> llama_pshard_layer_miss_share(const llama_pshard_workload * wl, uint32_t n_layers, uint32_t pool_slots);

// plan cache serialization. fingerprint covers the runtime plan-compatibility params, the
// predictor version and the profile files' hash, so the planner binary and the runtime binary
// share the same cache file when they see the same profile files (working directory or
// PSHARD_CPU_PROFILE / PSHARD_GPU_PROFILE).
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

// one strategy's result in a tier's sweep: the pick and its runner-ups, kept so a pick can be
// audited against the prices it beat; never executed
struct llama_pshard_candidate {
    llama_pshard_strategy strategy       = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
    bool                  is_viable      = false;
    float                 tps            = 0.0f;
    uint32_t              n_pinned       = 0;
    uint32_t              n_attn_pinned  = 0;
    uint32_t              pool_slots     = 0;
    size_t                total_vram_req = 0;
};

// the state a plan switch moves per layer: MB before the decode, and MB per row the decode writes
struct llama_pshard_switch_state {
    std::vector<double> mb;
    std::vector<double> row_mb;
};

struct llama_pshard_plan_registry {
    std::vector<uint32_t>                tier_sizes;
    std::vector<llama_pshard_plan>       best_plans;  // one best plan per tier
    std::vector<std::vector<llama_pshard_candidate>> candidates;  // per tier, every strategy the sweep priced
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
    // the machine profile's kernel-copy crossover (largest transfer at which a copy kernel still beats
    // a copy-engine transfer ordered against kernels); the pool sets it as the engine's cap while active.
    // -1 = not in the profile -> the engine keeps its default
    float kernel_copy_cap_mb = -1.0f;
    // hash of the machine (gpu description | cpu brand | os) the variant was planned on; the loader ignores a
    // variant planned elsewhere. 0 = unknown (older registry)
    uint64_t machine_hash = 0;
    bool  mtp_head_cpu     = false; // MTP head demoted to CPU by union-budget enforcement
    // retired, never charged; kept so existing registry files still parse. The one-budget fit
    // measures the MTP context's device need under the fitted placement (common_pshard_fit_one_budget).
    uint32_t mtp_head_extra_mb = 0;
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
    // any viable tier pools experts: the pool region is the budget remainder by
    // design, so the arena must be the whole budget, not union + headroom
    bool has_pool() const {
        for (const auto & p : best_plans) {
            if (p.is_viable && p.strategy == LLAMA_PSHARD_EXPERT_POOL) {
                return true;
            }
        }
        return false;
    }

    size_t arena_bytes(size_t budget_bytes) const {
        const size_t mib      = 1024ULL * 1024;
        const size_t headroom = 64 * mib;
        // mtp_head_extra_mb is retired and not charged here (see the field comment)
        const size_t budget   = budget_bytes;
        if (union_bytes == 0 || has_pool()) {
            return (budget / mib) * mib; // whole MiB, see below
        }
        // whole MiB: the buffer size positions the pinned cache region (buf_total - cache),
        // so a fractional size would misalign every cache tensor (CUDA misaligned address)
        const size_t want = ((union_bytes + headroom + mib - 1) / mib) * mib;
        return want >= budget ? budget : want;
    }

    // whether a plan keeps layer il whole on the device
    bool full_resident(const llama_pshard_plan & p, uint32_t il) const {
        return p.pin_from_back ? il + p.n_pinned >= n_layers : il < p.n_pinned;
    }

    // whether a plan keeps layer il's non-FFN tensors (attention, norms, router) and its KV / recurrent state
    // on the device: its whole layers or the attention pins (from the front)
    bool attn_resident(const llama_pshard_plan & p, uint32_t il) const {
        return full_resident(p, il) || il < p.n_attn_pinned;
    }

    // cost of switching from one plan to another, in ms. Weights move one way only: what `to` keeps on the device
    // and `from` does not is uploaded, what `from` keeps and `to` does not is dropped in place. The state of a
    // layer whose attention changes residency moves either way: st->mb[il] MB before the decode plus
    // st->row_mb[il] per row the decode has written (n_rows). An expert pool cache tier that lands on slots
    // another plan used starts empty and pays its refill (cold_ms); a pool cache tier with the same slot count
    // keeps them. Legacy caches fall back to the tier0-anchored per-plan estimate
    float switch_cost_ms(const llama_pshard_plan & from, const llama_pshard_plan & to,
                         const llama_pshard_switch_state * st = nullptr, double n_rows = 0.0) const {
        if (switch_layer_mb <= 0.0f || switch_pcie_gb_s <= 0.0f || n_layers == 0) {
            return to.switch_ms;
        }
        const double attn_mb = (double) switch_layer_mb * (double) switch_attn_frac;
        const double ffn_mb  = (double) switch_layer_mb - attn_mb;
        double mb = 0.0;
        for (uint32_t il = 0; il < n_layers; il++) {
            if (full_resident(to, il) && !full_resident(from, il)) {
                mb += ffn_mb;
            }
            const bool attn_from = attn_resident(from, il);
            const bool attn_to   = attn_resident(to, il);
            if (attn_to && !attn_from) {
                mb += attn_mb;
            }
            if (attn_to != attn_from && st != nullptr && il < st->mb.size()) {
                mb += st->mb[il] + n_rows * (il < st->row_mb.size() ? st->row_mb[il] : 0.0);
            }
        }
        if (to.output_on_gpu && !from.output_on_gpu) {
            mb += (double) switch_head_mb;
        }
        double ms = mb / (double) switch_pcie_gb_s;  // MB / (GB/s) == ms
        const bool same_slots = from.strategy == LLAMA_PSHARD_EXPERT_POOL && from.cold_ms > 0.0f &&
                                from.pool_slots == to.pool_slots;
        if (to.cold_ms > 0.0f && !same_slots) {
            ms += (double) to.cold_ms;
        }
        return (float) ms;
    }

    // variant marker for a baseline load that fits
    // runtime still checks baseline_vram_req against the current budget
    bool                                 pshard_disabled = false;
    // the load found no plan for this fingerprint and budget
    bool                                 cache_missed = false;
    size_t                               baseline_vram_req = 0;

    void init(uint32_t n_ubatch, uint32_t n_parallel = 1, uint32_t n_draft = 0) {
        tier_sizes.clear();
        best_plans.clear();
        candidates.clear();

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
            // multi-sequence decode: one sequence left, the steady-state step of n_parallel tokens,
            // its speculative verify batch of n_parallel * (n_draft + 1), and one small tier for
            // short mixed batches; every tier is a full search, so nothing else
            tier_sizes.push_back(1);
            const uint32_t verify_tier = n_draft > 0 ? n_parallel * (n_draft + 1) : 0;
            const uint32_t small_tier  = n_parallel < 16 ? 16 : 0;
            for (uint32_t t : { n_parallel, small_tier, verify_tier }) {
                if (t > 1 && t < 512 && t <= n_ubatch && std::find(tier_sizes.begin(), tier_sizes.end(), t) == tier_sizes.end()) {
                    tier_sizes.push_back(t);
                }
            }
            std::sort(tier_sizes.begin(), tier_sizes.end());
        }

        // prefill tiers: 512 and the 2048..8192 doublings, capped at n_ubatch; a context too
        // small for 8192 ends the ladder at its own n_ubatch
        for (uint32_t t : { 512u, 2048u, 4096u, 8192u }) {
            if (t > n_ubatch) break;
            if (tier_sizes.empty() || tier_sizes.back() < t) {
                tier_sizes.push_back(t);
            }
        }

        if (n_ubatch < 8192 && (tier_sizes.empty() || tier_sizes.back() != n_ubatch)) {
            tier_sizes.push_back(n_ubatch);
        }

        cache_ubatch = tier_sizes.empty() ? 0 : tier_sizes.back();
        best_plans.resize(tier_sizes.size());
        candidates.resize(tier_sizes.size());
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

    // the tier that executes a batch of batch_size tokens: the smallest VIABLE tier at or
    // above it (a larger tier's reserve covers a smaller ubatch); when none is viable above,
    // the largest viable tier below (the caller clamps its ubatch to that tier's size).
    // tier_sizes.size() = no viable tier at all.
    size_t viable_tier_for(uint32_t batch_size) const {
        for (size_t i = 0; i < tier_sizes.size(); i++) {
            if (tier_sizes[i] >= batch_size && best_plans[i].is_viable) return i;
        }
        for (size_t i = tier_sizes.size(); i-- > 0;) {
            if (best_plans[i].is_viable) return i;
        }
        return tier_sizes.size();
    }

    // the largest viable tier <= max_ubatch, never max_ubatch itself: the top tier can be unviable by design (a
    // pool tier whose scratch leaves no room for its region) and a ubatch routed to it would run on the decode
    // plan and spill past its window
    uint32_t largest_viable_ubatch(uint32_t max_ubatch) const {
        for (size_t t = tier_sizes.size(); t-- > 0;) {
            if (tier_sizes[t] <= max_ubatch && best_plans[t].is_viable) {
                return tier_sizes[t];
            }
        }
        return max_ubatch;
    }

    // how a decode runs: ubatches of tier_sizes[tier] tokens on `tier`, the last partial ubatch on `tail`
    struct cut {
        size_t tier = SIZE_MAX;   // SIZE_MAX = no priced tier
        size_t tail = SIZE_MAX;
        double ms   = 0.0;
    };

    // the cheapest cut of a decode of n_tokens: one ubatch on a tier that holds it, or whole ubatches of one tier
    // and the remainder on that tier or on a smaller viable tier that holds it. Each ubatch is priced from its
    // tier's piece curve; the switches are pairwise: from the active plan, between the two tiers, and back to the
    // decode plan. min_ubatch: the smallest ubatch size the memory's splitter accepts when it cuts. st: the state
    // each layer's switch moves (see switch_cost_ms)
    cut find_cut(uint32_t n_tokens, uint32_t max_ubatch, uint32_t min_ubatch, const llama_pshard_plan * from_plan,
                 const llama_pshard_switch_state * st = nullptr) const {
        cut best;
        const llama_pshard_plan * decode_plan =
            (!best_plans.empty() && best_plans[0].is_viable) ? &best_plans[0] : nullptr;
        if (from_plan == nullptr) {
            from_plan = decode_plan;
        }
        // n_rows: the rows of this decode written before the switch
        auto sw = [&](const llama_pshard_plan * a, const llama_pshard_plan * b, double n_rows) -> double {
            return a != nullptr && b != nullptr ? (double) switch_cost_ms(*a, *b, st, n_rows) : 0.0;
        };
        auto priced = [&](size_t t) {
            return best_plans[t].is_viable && best_plans[t].tps > 0.0f;
        };
        auto consider = [&](size_t t, size_t u, double ms) {
            if (best.tier == SIZE_MAX || ms < best.ms) {
                best.tier = t;
                best.tail = u;
                best.ms   = ms;
            }
        };

        for (size_t t = 0; t < tier_sizes.size(); t++) {
            const uint32_t ts = tier_sizes[t];
            if (ts > max_ubatch || !priced(t)) {
                continue;
            }
            const llama_pshard_plan * plan = &best_plans[t];
            if (n_tokens <= ts) {
                consider(t, t, sw(from_plan, plan, 0) + plan->ubatch_ms(n_tokens) + sw(plan, decode_plan, n_tokens));
                continue;
            }
            if (ts < min_ubatch) {
                continue;
            }
            const uint32_t k = n_tokens / ts;
            const uint32_t r = n_tokens % ts;
            const double head_ms = sw(from_plan, plan, 0) + (double) k * plan->ubatch_ms(ts);
            if (r == 0) {
                consider(t, t, head_ms + sw(plan, decode_plan, n_tokens));
                continue;
            }
            for (size_t u = 0; u <= t; u++) {
                if (tier_sizes[u] < r || !priced(u)) {
                    continue;
                }
                const llama_pshard_plan * tail = &best_plans[u];
                consider(t, u, head_ms + sw(plan, tail, (double) k * ts) + tail->ubatch_ms(r) + sw(tail, decode_plan, n_tokens));
            }
        }
        return best;
    }
};
