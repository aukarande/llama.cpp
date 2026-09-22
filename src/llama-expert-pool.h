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
#include <algorithm>
#include <string>
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

    // int32 id buffer: lives in the pool's page-locked arena when it fits (the CUDA backend then uploads
    // it with a kernel copy instead of a copy-engine transfer), else in an owned vector. Same call surface
    // as std::vector; a copy gets its own vector storage.
    struct ids_buf {
        int32_t * pin     = nullptr;   // arena slice (may be null)
        size_t    pin_cap = 0;         // elements
        std::vector<int32_t> vec;      // fallback storage
        int32_t * p = nullptr;
        size_t    n = 0;
        ids_buf() = default;
        ids_buf(const ids_buf & o) { *this = o; }
        ids_buf & operator=(const ids_buf & o) {
            if (this != &o) {
                assign(o.n, 0);
                std::copy(o.p, o.p + o.n, p);
            }
            return *this;
        }
        void assign(size_t count, int32_t v) {
            n = count;
            if (pin != nullptr && count <= pin_cap) {
                p = pin;
                std::fill(p, p + n, v);
            } else {
                vec.assign(count, v);
                p = vec.data();
            }
        }
        int32_t &       operator[](size_t i)       { return p[i]; }
        const int32_t & operator[](size_t i) const { return p[i]; }
        int32_t *       data()       { return p; }
        const int32_t * data() const { return p; }
        size_t          size() const { return n; }
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
        uint64_t evicted  = 0;              // residents displaced by a fetch
        uint64_t ab_pass  = 0;              // last pass this layer's A/B half was filled
        uint64_t serve_gen = 0;             // last generation serve() ran the full work
        bool     cpu_dead     = false;      // dual chain: this generation routes nothing to the CPU
        uint64_t cpu_dead_gen = 0;          // generation cpu_dead was decided for
        uint64_t cpu_dead_passes = 0;       // passes whose CPU chain had no routes (skippable)
        ids_buf mapped_buf;                 // remapped-ids upload buffer: MUST outlive the
                                            // async copy (the staging worker may queue the
                                            // host pointer behind pending staged fetches)

        // per-graph-build registration (rebound every build by build_moe_ffn)
        ggml_tensor * ids_router   = nullptr; // selected_experts (device, original ids)
        ggml_tensor * ids_gpu      = nullptr; // GPU chain mm ids: slot id | -1
        ggml_tensor * ids_gpu_bias = nullptr; // GPU chain add_id ids: expert id | -1 (dual only)
        ggml_tensor * ids_cpu      = nullptr; // CPU chain ids: expert id | -1 (dual only)
        ggml_tensor * ids_pred     = nullptr; // routing predicted predict_k layers earlier (output leaf)
        // router ids landing: the graph copies selected_experts into this layer's pinned slice; serve() polls it
        ggml_tensor * ids_host     = nullptr; // the graph's landing tensor (nullptr: the layer reads the ids back from the device)
        int32_t *     ids_host_pin = nullptr; // host view of the slice
        size_t        ids_host_cap = 0;       // elements
        size_t        ids_host_n   = 0;       // elements the bound landing tensor holds (n_expert_used x n_tokens)
        bool          ids_host_armed = false; // -1 sentinels written this pass and not consumed yet

        // this layer's router (for the predictor built in an earlier layer)
        const ggml_tensor * gate_inp    = nullptr;
        const ggml_tensor * gate_inp_b  = nullptr;
        const ggml_tensor * exp_probs_b = nullptr;

        // prediction quality: routes named by the prediction / all routes, and
        // misses named by the prediction / all misses (what a prefetch could hide)
        uint64_t pred_total   = 0;
        uint64_t pred_hit     = 0;
        uint64_t pred_misses  = 0;
        uint64_t pred_covered = 0;
        // prefetch: experts uploaded for this layer ahead of its service (from the
        // prediction) and how many of them the layer then actually routed to
        uint64_t pf_issued = 0;
        uint64_t pf_used   = 0;
        std::vector<uint64_t> slot_pf_gen;   // [n_slots] pass that prefetched the slot

        // this layer's slot count (= the pool's uniform n_slots, the same in every layer)
        uint32_t n_slots_l = 0;
        ids_buf bias_buf;                     // persistent upload buffers (async-safe)
        std::vector<int32_t> cpu_buf;
        std::vector<uint64_t> expert_last_gen; // [n_expert] recency: last generation routed
        std::vector<uint32_t> miss_count;      // [n_expert] fetch_on_2nd_miss admission counter
        std::vector<uint64_t> use_count;       // [n_expert] cache-mode routes per expert (the routing workload histogram)
        std::vector<uint32_t> expert_pending;  // [n_expert] generation whose pass admitted the expert
                                               // in the background (cpu_admit: CPU route that pass)
    };

    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_slots       = 0;        // cache-mode slots per layer (the same count in every layer)
    std::string workload_path;         // <model>.pshard_workload: the routing skew refit written at exit (empty = not persisted)
    bool     ab_mode       = false;    // active tier is a whole-stack prefill tier
    bool     active        = false;    // the ACTIVE plan is EXPERT_POOL (legacy tiers
                                       // in a mixed registry must stream normally)
    int      miss_policy   = 0;        // llama_pshard_miss_policy of the active tier
    float    hybrid_frac   = 0.55f;    // fetched share of misses under hybrid (B_P / B_H)
    // prefetch predictor depth: layer il's FFN input through layer il+k's router
    // predicts layer il+k's experts (PSHARD_POOL_PREDICT=k, 0 = off)
    int32_t  predict_k     = 0;
    bool     prefetch_on   = true;   // PSHARD_POOL_PREFETCH=0: predict and score only
    // dead CPU chains (every route of the pass resident or promoted): the sched skips the
    // chain's compute and zero-fills its merge input instead of the host join copy.
    uint64_t skipped_splits = 0;     // CPU chains the sched skipped on our word
    int32_t  prefetch_n    = 1;      // PSHARD_POOL_PREFETCH_N: at most this many of the predicted
                                     // experts per layer, highest predicted score first (0 = all).
                                     // Mispredicted uploads share the PCIe link with the critical-path misses.
    std::vector<char> pred_read_buf;
    // page-locked staging: the ids / prediction downloads and every layer's id upload buffers.
    // Device-accessible, so with kernel copies on the backend moves them with kernels instead
    // of copy-engine transfers (a DMA ordered behind kernels idles the GPU under WDDM whether
    // the host side is pinned or pageable). Allocated while kernel copies are on; larger batches
    // fall back to the vectors. Also holds the per-layer router-ids landing slices (ids_host_pin),
    // written by the graph via ids_host_buf
    ggml_backend_buffer_t read_staging = nullptr;
    bool   read_staging_tried = false;
    bool   kernel_copies      = false;   // pinned copies run as kernels (the context's transfer mode)
    ggml_backend_copy_segments_async_t copy_segments = nullptr;   // batched upload proc (CUDA), else per-tensor
    bool   procs_looked_up = false;
    void lookup_backend_procs();
    struct upload_seg { ggml_tensor * view; size_t off; const void * src; size_t size; };
    std::vector<upload_seg> upload_segs;   // one pass's admitted rows, issued as one launch
    char * read_ids  = nullptr;
    char * read_pred = nullptr;
    size_t read_cap  = 0;
    void ensure_read_staging(ggml_backend_t backend);
    // router-ids landing: device alias of the pinned staging, its host base, counters
    ggml_backend_wrap_host_buffer_t wrap_host_buffer = nullptr;
    ggml_backend_buffer_t ids_host_buf  = nullptr;
    char *                ids_host_base = nullptr;   // host address of ids_host_buf's base
    uint32_t ids_host_unconsumed = 0;   // layers armed this pass whose serve() has not read them
    uint64_t ids_host_polls      = 0;   // readbacks that went through the landing slice
    uint64_t ids_host_drains     = 0;   // of those, polls that drained the stream before the slice filled
    uint64_t ids_host_wait_us    = 0;   // host time spent polling for them
    uint64_t ids_host_fallbacks  = 0;   // slices that never filled (stream drained, then the device readback)
    uint32_t ids_host_drain_streak = 0;     // consecutive decode-sized polls that filled only after a drain
    bool     ids_host_sync_first   = false; // this device's writes reach the landing slice only after a drain
    bool wait_ids_host(layer_state & L, size_t n);
    uint64_t epoch         = 0;        // bumped on active/ab flips; joins graph reuse
    uint64_t generation    = 0;        // bumped once per decode call; dedupes serve()

    void *   region_base   = nullptr;
    size_t   region_bytes  = 0;
    ggml_backend_buffer_t region_arena = nullptr;
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

    // tier switch: flip cache/AB mode and re-register; cache contents do not survive
    // a mode round-trip (the halves alias the slot arrays): maps dropped, lazy refill
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
    // the target layer's router tensors for the predictor (false: layer not pooled)
    bool router_of(int32_t il, const ggml_tensor *& gate_inp, const ggml_tensor *& gate_inp_b,
                   const ggml_tensor *& exp_probs_b) const;
    void bind_pred_ids(int32_t il, ggml_tensor * ids_pred);
    // the pool's copy backend + event (background admission, prefetch); no-op once created
    void ensure_admit_backend(ggml_backend_t split_backend);
    void bind_layer_ids(int32_t il, ggml_tensor * ids_router, ggml_tensor * ids_gpu,
                        ggml_tensor * ids_gpu_bias, ggml_tensor * ids_cpu, ggml_tensor * ids_host);
    // graph build: device tensor aliasing this layer's landing slice (nullptr: unmapped or too large)
    ggml_tensor * ids_host_tensor(int32_t il, ggml_context * ctx, const ggml_tensor * ids);
    // pass start: write the -1 sentinels the landing slices are polled against
    void arm_ids_host();
    // finished graph: move nodes independent of a pooled layer's routed split ahead of its boundary
    void hoist_independent(ggml_cgraph * gf);
    uint32_t hoisted_nodes = 0;   // last build: nodes moved ahead of a boundary
    uint32_t hoist_regions = 0;   // last build: routed boundaries seen
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
    bool                 admit_tried   = false;
    ggml_backend_event_t admit_event   = nullptr;
    bool                 admit_pending = false;

    // prefetch-time service (whole-stack tiers only): fill the layer's A/B half on
    // the copy backend while the previous layer computes; serve() then finds the
    // half already filled for this generation and only uploads the ids
    static bool sched_prefetch_cb(const ggml_tensor * src, ggml_tensor * view,
                                  ggml_backend_t copy_backend, void * user_data);
    bool prefetch(const ggml_tensor * src, ggml_backend_t copy_backend);

    // compute-time service (sched callback): true when the split is one of our CPU chains
    // and serve() routed nothing to it this generation, so every node is zeros
    static bool sched_split_skip_cb(const ggml_cgraph * split_graph, ggml_backend_t backend, void * user_data);
    bool split_is_zero(const ggml_cgraph * split_graph);

    void reset_slots();
    void log_counters() const;

    // scratch for the device->host ids read (persistent: async-safe lifetime)
    std::vector<char> ids_read_buf;
    // pass-local distinct-expert marker (union guard)
    std::vector<uint8_t> seen_gen;
};

// finished graph of a pshard context: nodes that follow a layer's first read of a host-resident weight (a CPU-computed
// or streamed layer) or of a pool-served tensor, without depending on it, move ahead of that boundary; the pool's ids
// leaves count as late when the pool is active (pool may be nullptr)
// sched (may be nullptr): a region whose boundary computes on the CPU gets a split cut before its first hoisted
// node, so the scheduler fetches the CPU split's inputs before the hoisted nodes launch
void llama_pshard_hoist_independent(ggml_cgraph * gf, llama_expert_pool * pool, ggml_backend_sched_t sched,
                                    uint32_t * hoisted, uint32_t * regions);
