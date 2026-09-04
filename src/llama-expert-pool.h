#pragma once

// EXPERT_POOL runtime (docs/expert-pool-design.md): routed experts stop being
// placed weights and become a managed VRAM cache. One instance per
// llama_context. It owns
//   - a device region carved from the model arena ([weights | scratch | POOL |
//     pinned KV]), never galloc-managed,
//   - per-layer expert->slot maps with LRU eviction and hit/miss counters,
//   - the consume-time serving of pool-managed graph inputs: the layer's host
//     expert tensors are registered with the scheduler as input-copy overrides
//     (ggml_backend_sched_set_input_copy_override), so their transient copies
//     become persistent pool views and the upload runs through serve().
//
// Two view sets exist per layer and the active one follows the tier:
//   - cache mode (decode / small batch): ne[2] = n_slots, ids remapped through
//     the expert->slot map, misses fetched into LRU victims;
//   - A/B mode (whole-stack prefill tiers, bs*top_k*2 >= n_expert): ne[2] =
//     n_expert over an alternating half of the region, identity ids, the whole
//     layer uploaded on first use per pass.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct llama_model;

struct llama_expert_pool {
    // one routed-expert weight tensor of one layer (fused gate_up, or up/gate, and down)
    struct tensor_entry {
        const ggml_tensor * host      = nullptr;  // mmap-backed home (model tensor)
        ggml_tensor *       view_slots = nullptr; // cache-mode view, ne[2] = n_slots
        ggml_tensor *       view_ab    = nullptr; // A/B-mode view,  ne[2] = n_expert
        size_t              row_bytes  = 0;       // one expert = host->nb[2]
        size_t              region_off = 0;       // slot 0 offset inside the region (cache mode)
        size_t              ab_off[2]  = {0, 0};  // layer-half offsets (A/B mode)
    };

    struct layer_state {
        int32_t il = -1;
        std::vector<tensor_entry> tensors;

        // slot state is shared across the layer's tensors: slot i holds expert
        // slot_expert[i] in every tensor of the layer
        std::vector<int32_t>  expert_slot;  // [n_expert] -> slot or -1
        std::vector<int32_t>  slot_expert;  // [n_slots]  -> expert or -1
        std::vector<uint64_t> slot_stamp;   // LRU stamps
        uint64_t stamp    = 0;
        uint64_t hits     = 0;
        uint64_t misses   = 0;
        uint64_t cache_passes = 0;          // cache-mode serve() passes (misses/token denominator)
        uint64_t hits_1   = 0;              // the same, restricted to single-token passes (decode)
        uint64_t misses_1 = 0;
        uint64_t passes_1 = 0;
        uint64_t demoted  = 0;              // probationary fetches restamped below the residents
        uint64_t evicted  = 0;              // residents displaced by a fetch
        uint64_t ab_pass  = 0;              // last pass this layer's A/B half was filled
        uint64_t serve_gen = 0;             // last generation serve() ran the full work
        std::vector<int32_t> mapped_buf;    // remapped-ids upload buffer: MUST outlive the
                                            // async copy (the staging worker may queue the
                                            // host pointer behind pending staged fetches)

        // per-graph-build registration (rebound every build by build_moe_ffn)
        ggml_tensor * ids_router   = nullptr; // selected_experts (device, original ids)
        ggml_tensor * ids_gpu      = nullptr; // GPU chain mm ids: slot id | -1
        ggml_tensor * ids_gpu_bias = nullptr; // GPU chain add_id ids: expert id | -1 (dual only)
        ggml_tensor * ids_cpu      = nullptr; // CPU chain ids: expert id | -1 (dual only)
        std::vector<int32_t> bias_buf;        // persistent upload buffers (async-safe)
        std::vector<int32_t> cpu_buf;
        std::vector<uint64_t> expert_last_gen; // [n_expert] recency: last generation routed
        std::vector<uint32_t> miss_count;      // [n_expert] fetch_on_2nd_miss admission counter
        std::vector<uint32_t> expert_pending;  // [n_expert] generation whose pass admitted the expert
                                               // in the background (cpu_admit: CPU route that pass)
    };

    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_slots       = 0;        // cache-mode slots per layer (v1 uniform)
    bool     ab_mode       = false;    // active tier is a whole-stack prefill tier
    bool     active        = false;    // the ACTIVE plan is EXPERT_POOL (legacy tiers
                                       // in a mixed registry must stream normally)
    int      miss_policy   = 0;        // llama_pshard_miss_policy of the active tier
    float    hybrid_frac   = 0.55f;    // fetched share of misses under hybrid (B_P / B_H)
    // probationary admission (RFC #24528's ADMIT_AFTER): an expert earns an MRU stamp
    // only from its admit_after-th miss on; earlier fetches still serve the pass but
    // land below every live resident, so one-off experts cannot evict the hot set.
    // 1 = plain LRU. PSHARD_POOL_ADMIT_AFTER overrides.
    uint32_t admit_after   = 1;
    uint64_t epoch         = 0;        // bumped on active/ab flips; joins graph reuse
    uint64_t generation    = 0;        // bumped once per decode call; dedupes serve()

    void *   region_base   = nullptr;
    size_t   region_bytes  = 0;
    size_t   layer_slot_bytes = 0;     // per-layer cache-mode footprint (all tensors)
    size_t   layer_full_bytes = 0;     // per-layer whole-expert-set footprint
    bool     ab_capable       = false; // the region holds the A/B pair (set_region)

    std::vector<layer_state> layers;   // dense by il; tensors empty for non-moe layers
    ggml_context * ctx_views = nullptr;
    ggml_backend_t backend_router = nullptr; // compute backend that produced the ids

    ~llama_expert_pool();

    // scan the model's routed-expert tensors; false when the model has none
    bool init(const llama_model & model, uint32_t n_expert, uint32_t n_expert_used);

    // bind the carved region and (re)build both view sets; slots = cache-mode
    // slots per layer from the active plan. arena = the model's managed device
    // buffer (the views carry it so async copies pass the backend checks)
    bool set_region(ggml_backend_buffer_t arena, void * base, size_t bytes, uint32_t slots_per_layer);

    // bytes the region needs for a given slot count (planner/carve agreement)
    // bytes for slots_per_layer cache slots; with_ab also requires the 2-layer
    // A/B pair (whole-stack tiers overlay it on the region start)
    size_t region_bytes_needed(uint32_t slots_per_layer, bool with_ab = true) const;

    // register every layer's host tensors as sched input-copy overrides pointing
    // at the ACTIVE view set, and install the serving callback
    void register_sched(ggml_backend_sched_t sched);

    // tier switch: flip cache/AB mode and re-register; cache contents survive a
    // mode round-trip only in the preserved span (v1: dropped - lazy refill)
    void set_ab_mode(bool ab, ggml_backend_sched_t sched);

    // the ACTIVE plan pools experts; when false the overrides are cleared and
    // pooled layers build/stream like any legacy plan
    void set_active(bool on, ggml_backend_sched_t sched);

    // true when the active policy admits CPU-executed routes (the split-op:
    // two expert chains, -1 for the other side's routes)
    bool cpu_routes() const { return miss_policy != 0; }

    // policy of the active tier; a change of chain shape (single <-> dual)
    // bumps the epoch
    void set_policy(int policy, float frac, ggml_backend_sched_t sched);

    // graph-build registration (called from build_moe_ffn via the graph channel)
    void bind_layer_ids(int32_t il, ggml_tensor * ids_router, ggml_tensor * ids_gpu,
                        ggml_tensor * ids_gpu_bias, ggml_tensor * ids_cpu);
    ggml_tensor * mm_view(int32_t il, const ggml_tensor * host) const;
    bool layer_pooled(int32_t il) const {
        return il >= 0 && il < (int32_t) layers.size() && !layers[il].tensors.empty();
    }
    // is THIS weight one of the layer's pooled tensors? A layer can carry a second,
    // unpooled expert group (GroveMoE chunk experts) that builds through the same
    // MoE FFN path: it must keep the router ids and stream normally.
    bool tensor_pooled(int32_t il, const ggml_tensor * w) const {
        if (!layer_pooled(il) || w == nullptr) {
            return false;
        }
        for (const auto & e : layers[il].tensors) {
            if (e.host == w) {
                return true;
            }
        }
        return false;
    }

    // consume-time service (sched callback): reads the router ids, remaps,
    // fetches misses into victim slots (cache mode) or fills the layer half
    // (A/B mode), uploads ids_gpu
    static bool sched_input_cb(const ggml_tensor * src, ggml_tensor * view,
                               ggml_backend_t split_backend, void * user_data);
    bool serve(const ggml_tensor * src, ggml_tensor * view, ggml_backend_t split_backend);

    // background admission (cpu_admit): admitted experts upload on the pool's own
    // copy backend (a second backend instance on the split device = its own stream,
    // like the sched's prefetch copy backends) while the CPU chain computes them this
    // pass; the GPU reads the slot from the next pass on. One event, recorded after
    // each layer's uploads (FIFO stream: the latest record covers them all), waited
    // on by the split stream at the first service of the next pass.
    ggml_backend_t       admit_backend = nullptr;
    ggml_backend_event_t admit_event   = nullptr;
    bool                 admit_pending = false;

    // prefetch-time service (whole-stack tiers only): fill the layer's A/B half on
    // the copy backend while the previous layer computes; serve() then finds the
    // half already filled for this generation and only uploads the ids
    static bool sched_prefetch_cb(const ggml_tensor * src, ggml_tensor * view,
                                  ggml_backend_t copy_backend, void * user_data);
    bool prefetch(const ggml_tensor * src, ggml_backend_t copy_backend);

    void reset_slots();
    void log_counters() const;

    // scratch for the device->host ids read (persistent: async-safe lifetime)
    std::vector<char> ids_read_buf;
    // pass-local distinct-expert marker (union guard)
    std::vector<uint8_t> seen_gen;
};
