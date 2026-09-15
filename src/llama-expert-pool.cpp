#include "llama-expert-pool.h"
#include "llama-pshard-plan.h"
#include "llama-pshard-workload.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

llama_expert_pool::~llama_expert_pool() {
    if (admit_backend != nullptr) {
        ggml_backend_synchronize(admit_backend);
    }
    if (admit_event != nullptr) {
        ggml_backend_event_free(admit_event);
        admit_event = nullptr;
    }
    if (admit_backend != nullptr) {
        ggml_backend_free(admit_backend);
        admit_backend = nullptr;
    }
    if (ids_host_buf != nullptr) {
        ggml_backend_buffer_free(ids_host_buf);   // the alias first: it frees nothing, the staging owns the memory
        ids_host_buf = nullptr;
    }
    if (read_staging != nullptr) {
        ggml_backend_buffer_free(read_staging);
        read_staging = nullptr;
    }
    if (ctx_views != nullptr) {
        ggml_free(ctx_views);
    }
}

bool llama_expert_pool::init(const llama_model & model, uint32_t n_expert_, uint32_t n_expert_used_) {
    workload_path = model.get_path_model().empty() ? std::string() : llama_pshard_workload::path_for(model.get_path_model());
    if (const char * pk = getenv("PSHARD_POOL_PREDICT")) {
        predict_k = (int32_t) std::min<long>(8, std::max<long>(0, strtol(pk, nullptr, 10)));
    }
    if (const char * pf = getenv("PSHARD_POOL_PREFETCH")) {
        prefetch_on = strtol(pf, nullptr, 10) != 0;
    }
    if (const char * pn = getenv("PSHARD_POOL_PREFETCH_N")) {
        prefetch_n = (int32_t) std::max<long>(0, strtol(pn, nullptr, 10));
    }
    n_expert      = n_expert_;
    n_expert_used = n_expert_used_;
    if (n_expert == 0 || n_expert_used == 0) {
        return false;
    }

    layers.clear();
    layers.resize(model.layers.size());
    layer_full_bytes = 0;

    size_t n_pooled = 0;
    for (size_t il = 0; il < model.layers.size(); il++) {
        const auto & ml = model.layers[il];
        layer_state & L = layers[il];
        L.il = (int32_t) il;

        // routed experts only: ne[2] == n_expert; shared experts (shexp) and dense
        // FFNs keep classic placement
        const ggml_tensor * cands[4] = { ml.ffn_gate_up_exps, ml.ffn_up_exps, ml.ffn_gate_exps, ml.ffn_down_exps };
        size_t full = 0;
        for (const ggml_tensor * t : cands) {
            if (t == nullptr || t->ne[2] != (int64_t) n_expert) {
                continue;
            }
            if (t->buffer != nullptr && !ggml_backend_buffer_is_host(t->buffer)) {
                continue;   // pinned on the device (e.g. the MTP head's experts): not a pool home
            }
            tensor_entry e;
            e.host      = t;
            e.row_bytes = t->nb[2];
            full += e.row_bytes * n_expert;
            L.tensors.push_back(e);
        }
        L.gate_inp    = ml.ffn_gate_inp;
        L.gate_inp_b  = ml.ffn_gate_inp_b;
        L.exp_probs_b = ml.ffn_exp_probs_b;
        if (!L.tensors.empty()) {
            n_pooled++;
            layer_full_bytes = std::max(layer_full_bytes, full);
            L.expert_slot.assign(n_expert, -1);
            L.use_count.assign(n_expert, 0);
        }
    }

    // quantized-padding contract: the sliced/prefetch upload paths reserve and
    // zero MMQ row padding for tensors whose ne0 is not a multiple of 512; the
    // pool's slot views do not (yet) - refuse such models instead of computing
    // with garbage tail scales
    for (const auto & L : layers) {
        for (const auto & e : L.tensors) {
            if (ggml_is_quantized(e.host->type) && e.host->ne[0] % 512 != 0) {
                LLAMA_LOG_WARN("%s: expert pool: %s has ne0=%lld %% 512 != 0 (MMQ padding "
                    "contract unhandled) - pool disabled for this model\n",
                    __func__, e.host->name, (long long) e.host->ne[0]);
                return false;
            }
        }
    }

    LLAMA_LOG_INFO("%s: expert pool: %zu pooled layers, %u experts (%u used), max layer %.1f MiB\n",
        __func__, n_pooled, n_expert, n_expert_used, layer_full_bytes / (1024.0 * 1024.0));
    return n_pooled > 0;
}

size_t llama_expert_pool::region_bytes_needed(uint32_t slots_per_layer, bool with_ab) const {
    // cache mode: slots_per_layer rows of every pooled tensor, per layer;
    // ab_mode reuses the same span for two whole layers - take the max when asked
    size_t cache_bytes = 0;
    for (const auto & L : layers) {
        for (const auto & e : L.tensors) {
            cache_bytes += ((size_t) slots_per_layer * e.row_bytes + 255) & ~(size_t) 255;
        }
    }
    return with_ab ? std::max(cache_bytes, 2 * layer_full_bytes) : cache_bytes;
}

bool llama_expert_pool::set_region(ggml_backend_buffer_t arena, void * base, size_t bytes, uint32_t slots_per_layer) {
    region_base  = base;
    region_bytes = bytes;
    region_arena = arena;
    n_slots      = slots_per_layer;

    if (ctx_views != nullptr) {
        ggml_free(ctx_views);
        ctx_views = nullptr;
    }

    size_t n_tensors = 0;
    for (const auto & L : layers) {
        n_tensors += L.tensors.size();
    }
    if (n_tensors == 0 || base == nullptr) {
        return false;
    }

    // per-layer slot counts are uniform; the region must hold every layer's slot arrays
    size_t need = 0;
    for (auto & L : layers) {
        L.n_slots_l = slots_per_layer;
        for (const auto & e : L.tensors) {
            need += ((size_t) L.n_slots_l * e.row_bytes + 255) & ~(size_t) 255;
        }
    }
    if (need > bytes) {
        LLAMA_LOG_WARN("%s: expert pool region too small: need %.1f MiB (s=%u), have %.1f MiB\n",
            __func__, need / (1024.0 * 1024.0), slots_per_layer, bytes / (1024.0 * 1024.0));
        return false;
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ 2 * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_views = ggml_init(ip);

    // cache-mode layout: layer-major, per-tensor slot arrays; the ab_mode halves overlay the region start
    // (they are only live on whole-stack prefill tiers, where the cache contents are volatile by design)
    size_t off = 0;
    for (auto & L : layers) {
        L.slot_expert.assign(L.n_slots_l, -1);
        L.slot_stamp.assign(L.n_slots_l, 0);
        L.slot_pf_gen.assign(L.n_slots_l, 0);
        if (!L.expert_slot.empty()) {
            std::fill(L.expert_slot.begin(), L.expert_slot.end(), -1);
        }
        for (auto & e : L.tensors) {
            e.region_off = off;
            off += ((size_t) L.n_slots_l * e.row_bytes + 255) & ~(size_t) 255;

            e.view_slots = ggml_new_tensor_3d(ctx_views, e.host->type,
                e.host->ne[0], e.host->ne[1], L.n_slots_l);
            e.view_slots->data   = (char *) base + e.region_off;
            e.view_slots->buffer = arena;
            ggml_format_name(e.view_slots, "pool_s#%s", e.host->name);
        }
    }

    // ab_mode halves: parity-alternating whole-layer sets at the region start, only when the region holds the pair;
    // a cache tier's region can be smaller, and the whole-stack tiers get their own larger carve when they are applied
    ab_capable = bytes >= 2 * layer_full_bytes;
    for (auto & L : layers) {
        size_t sub = 0;
        const size_t half = layer_full_bytes;
        for (auto & e : L.tensors) {
            e.view_ab = nullptr;
            e.ab_off[0] = e.ab_off[1] = 0;
            if (!ab_capable) {
                continue;
            }
            e.ab_off[0] = 0    + sub;
            e.ab_off[1] = half + sub;
            sub += e.row_bytes * n_expert;

            e.view_ab = ggml_new_tensor_3d(ctx_views, e.host->type,
                e.host->ne[0], e.host->ne[1], n_expert);
            e.view_ab->data   = (char *) base + e.ab_off[L.il & 1];
            e.view_ab->buffer = arena;
            ggml_format_name(e.view_ab, "pool_ab#%s", e.host->name);
        }
    }

    LLAMA_LOG_INFO("%s: expert pool region %.1f MiB: s=%u slots/layer (cache)%s\n",
        __func__, bytes / (1024.0 * 1024.0), n_slots,
        ab_capable ? " + 2-layer A/B pair" : " (no A/B pair: cache tiers only)");
    return true;
}

void llama_expert_pool::register_sched(ggml_backend_sched_t sched) {
    ggml_backend_sched_clear_input_copy_overrides(sched);
    ggml_backend_sched_set_split_skip_cb(sched, nullptr, nullptr);
    if (!active) {
        // legacy tier active: pooled layers stream through the standard paths
        return;
    }
    for (auto & L : layers) {
        for (auto & e : L.tensors) {
            ggml_tensor * view = ab_mode ? e.view_ab : e.view_slots;
            if (view != nullptr) {
                // const_cast: the sched keys the override map on the pointer only
                ggml_backend_sched_set_input_copy_override(sched, e.host, view);
            }
        }
    }
    ggml_backend_sched_set_pool_input_cb(sched, sched_input_cb, this);
    ggml_backend_sched_set_pool_prefetch_cb(sched, sched_prefetch_cb);
    if (cpu_routes()) {
        ggml_backend_sched_set_split_skip_cb(sched, sched_split_skip_cb, this);
    }
}

void llama_expert_pool::lookup_backend_procs() {
    if (procs_looked_up) {
        return;
    }
    procs_looked_up = true;
    ggml_backend_dev_t dev = backend_router != nullptr ? ggml_backend_get_device(backend_router)
                                                       : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr) {
        return;
    }
    copy_segments    = (ggml_backend_copy_segments_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_segments_async");
    wrap_host_buffer = (ggml_backend_wrap_host_buffer_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_wrap_host_buffer");
}

void llama_expert_pool::set_active(bool on, ggml_backend_sched_t sched) {
    if (on == active) {
        return;
    }
    active = on;
    lookup_backend_procs();
    epoch++;   // pooled-layer graph topology changes with this flag
    reset_slots();
    if (!on) {
        // the next graphs will not bind these; stale pointers would alias whatever
        // tensor the rebuilt graph places at the same address
        for (auto & L : layers) {
            L.ids_router = L.ids_gpu = L.ids_gpu_bias = L.ids_cpu = L.ids_pred = L.ids_host = nullptr;
            L.ids_host_armed = false;
            L.ids_host_n = 0;
        }
    }
    if (sched != nullptr) {
        register_sched(sched);
    }
}

void llama_expert_pool::set_ab_mode(bool ab, ggml_backend_sched_t sched) {
    if (ab && active && !ab_capable) {
        LLAMA_LOG_ERROR("%s: whole-stack tier on a region without the A/B pair (%.1f MiB) - staying in cache mode\n",
            __func__, region_bytes / (1024.0 * 1024.0));
        ab = false;
    }
    if (ab == ab_mode) {
        return;
    }
    ab_mode = ab;
    epoch++;   // stale view bindings must not survive a graph-reuse pass
    // cache contents do not survive the whole-layer overlay (the halves alias the slot arrays): drop the maps and refill lazily
    reset_slots();
    if (sched != nullptr) {
        register_sched(sched);
    }
}

void llama_expert_pool::set_policy(int policy, float frac, ggml_backend_sched_t sched) {
    const bool was_dual = cpu_routes();
    miss_policy = policy;
    if (frac > 0.0f && frac <= 1.0f) {
        hybrid_frac = frac;
    }
    if (cpu_routes() != was_dual) {
        epoch++;   // single <-> dual chain: graph topology changes
        if (sched != nullptr && active) {
            register_sched(sched);
        }
    }
}

void llama_expert_pool::ensure_admit_backend(ggml_backend_t split_backend) {
    if (admit_backend != nullptr || admit_tried) {
        return;
    }
    admit_tried   = true;
    admit_backend = ggml_backend_dev_init(ggml_backend_get_device(split_backend), nullptr);
    if (admit_backend != nullptr) {
        admit_event = ggml_backend_event_new(ggml_backend_get_device(split_backend));
        if (admit_event == nullptr) {
            // no events on this device: an upload nobody can wait on must not leave
            // the split stream - background admission / prefetch stay off
            ggml_backend_free(admit_backend);
            admit_backend = nullptr;
            LLAMA_LOG_WARN("%s: no device events - copy-stream uploads disabled\n", __func__);
        }
    }
}

bool llama_expert_pool::router_of(int32_t il, const ggml_tensor *& gate_inp, const ggml_tensor *& gate_inp_b,
                                  const ggml_tensor *& exp_probs_b) const {
    if (!layer_pooled(il)) {
        return false;
    }
    gate_inp    = layers[il].gate_inp;
    gate_inp_b  = layers[il].gate_inp_b;
    exp_probs_b = layers[il].exp_probs_b;
    return true;
}

void llama_expert_pool::bind_pred_ids(int32_t il, ggml_tensor * ids_pred) {
    if (!layer_pooled(il)) {
        return;
    }
    layers[il].ids_pred = ids_pred;
}

void llama_expert_pool::bind_layer_ids(int32_t il, ggml_tensor * ids_router, ggml_tensor * ids_gpu,
                                       ggml_tensor * ids_gpu_bias, ggml_tensor * ids_cpu, ggml_tensor * ids_host) {
    if (!layer_pooled(il)) {
        return;
    }
    layer_state & L = layers[il];
    L.ids_router   = ids_router;
    L.ids_gpu      = ids_gpu;
    L.ids_gpu_bias = ids_gpu_bias;
    L.ids_cpu      = ids_cpu;
    L.ids_host     = ids_host;
    L.ids_host_n   = ids_host != nullptr ? (size_t) ggml_nelements(ids_host) : 0;
    L.ids_host_armed = false;
    L.serve_gen    = 0;
    // ids_pred is NOT touched here: within build_moe_ffn(il) the predictor for
    // layer il+k binds BEFORE this call, and this layer's own prediction was bound
    // k layers ago; every rebuild rebinds all reachable targets, deactivation clears
    if (L.expert_last_gen.size() != n_expert) {
        L.expert_last_gen.assign(n_expert, 0);
        L.miss_count.assign(n_expert, 0);
        L.expert_pending.assign(n_expert, 0);
    }
}

ggml_tensor * llama_expert_pool::mm_view(int32_t il, const ggml_tensor * host) const {
    if (!layer_pooled(il)) {
        return nullptr;
    }
    for (const auto & e : layers[il].tensors) {
        if (e.host == host) {
            return ab_mode ? e.view_ab : e.view_slots;
        }
    }
    return nullptr;
}

bool llama_expert_pool::sched_input_cb(const ggml_tensor * src, ggml_tensor * view,
                                       ggml_backend_t split_backend, void * user_data) {
    return ((llama_expert_pool *) user_data)->serve(src, view, split_backend);
}

bool llama_expert_pool::sched_prefetch_cb(const ggml_tensor * src, ggml_tensor * view,
                                          ggml_backend_t copy_backend, void * user_data) {
    GGML_UNUSED(view);
    return ((llama_expert_pool *) user_data)->prefetch(src, copy_backend);
}

bool llama_expert_pool::sched_split_skip_cb(const ggml_cgraph * split_graph, ggml_backend_t backend, void * user_data) {
    GGML_UNUSED(backend);
    return ((llama_expert_pool *) user_data)->split_is_zero(split_graph);
}

bool llama_expert_pool::split_is_zero(const ggml_cgraph * g) {
    ggml_cgraph * gg = const_cast<ggml_cgraph *>(g);   // the public accessors take a mutable graph
    // the split must be exactly a CPU expert chain served this generation with every route -1:
    // MUL_MAT_IDs over one of our ids_cpu leaves (the CPU op writes zero rows for -1 routes),
    // plus ops that map zeros to zeros and whose every source is an earlier node of the split
    if (ggml_graph_n_nodes(gg) == 0 || generation == 0) {
        return false;
    }
    bool any_chain = false;
    for (int i = 0; i < ggml_graph_n_nodes(gg); i++) {
        const ggml_tensor * t = ggml_graph_node(gg, i);
        auto in_split = [&](const ggml_tensor * s) {
            for (int j = 0; j < i; j++) {
                if (ggml_graph_node(gg, j) == s) {
                    return true;
                }
            }
            return false;
        };
        switch (t->op) {
            case GGML_OP_MUL_MAT_ID: {
                const ggml_tensor * ids = t->src[2];
                const layer_state * L = nullptr;
                for (const auto & C : layers) {
                    if (C.ids_cpu != nullptr && C.ids_cpu == ids) {
                        L = &C;
                        break;
                    }
                }
                if (L == nullptr || !L->cpu_dead || L->cpu_dead_gen != generation || L->serve_gen != generation) {
                    return false;
                }
                any_chain = true;
            } break;
            case GGML_OP_GLU:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (t->src[s] != nullptr && !in_split(t->src[s])) {
                        return false;
                    }
                }
                break;
            default:
                return false;
        }
    }
    if (any_chain) {
        skipped_splits++;
    }
    return any_chain;
}

bool llama_expert_pool::prefetch(const ggml_tensor * src, ggml_backend_t copy_backend) {
    if (!active || !ab_mode) {
        return false; // cache tiers: the router ids are not computed yet
    }
    layer_state * Lp = nullptr;
    for (auto & L : layers) {
        for (const auto & e : L.tensors) {
            if (e.host == src) {
                Lp = &L;
                break;
            }
        }
        if (Lp != nullptr) {
            break;
        }
    }
    if (Lp == nullptr) {
        return false;
    }
    layer_state & L = *Lp;
    if (L.ab_pass == generation && generation > 0) {
        return true; // the layer's other tensors: already filled by the first call
    }
    // the sched already waited on the compute fence for this copy stream, so the
    // half (last read by layer il-2) is free; the consumer waits on the copy event
    for (const auto & e : L.tensors) {
        ggml_backend_tensor_set_async(copy_backend, e.view_ab,
            e.host->data, 0, (size_t) n_expert * e.row_bytes);
    }
    L.ab_pass = generation;
    return true;
}

void llama_expert_pool::ensure_read_staging(ggml_backend_t backend) {
    if (read_staging_tried) {
        return;
    }
    read_staging_tried = true;
    if (!kernel_copies) {
        return;   // without kernel copies a pinned transfer is a DMA behind kernels: no gain
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (buft == nullptr) {
        return;
    }
    read_cap = 512 * 1024;           // ids or prediction download of a 2048-token ubatch x 16 routes x 4 B = 128 KB; headroom
    const size_t per_layer = 8192;   // per-layer id upload buffers: 1024 tokens x 8 routes; larger batches use the vectors
    const size_t landing   = layers.size() * per_layer * sizeof(int32_t);   // per-layer router-ids landing slices
    const size_t bytes = 2 * read_cap + layers.size() * 2 * per_layer * sizeof(int32_t) + landing;
    read_staging = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (read_staging == nullptr) {
        read_cap = 0;
        return;
    }
    char * base = (char *) ggml_backend_buffer_get_base(read_staging);
    read_ids  = base;
    read_pred = base + read_cap;
    int32_t * slice = (int32_t *) (base + 2 * read_cap);
    for (auto & L : layers) {
        L.mapped_buf.pin = slice; L.mapped_buf.pin_cap = per_layer; slice += per_layer;
        L.bias_buf.pin   = slice; L.bias_buf.pin_cap   = per_layer; slice += per_layer;
    }
    // landing slices: a device alias lets the graph write the router ids into host memory; without the mapping
    // the layers read the ids back with a stream drain
    ids_host_base = (char *) slice;
    lookup_backend_procs();
    ids_host_buf = wrap_host_buffer != nullptr ? wrap_host_buffer(backend, ids_host_base, landing) : nullptr;
    if (ids_host_buf != nullptr) {
        std::fill(slice, slice + layers.size() * per_layer, -1);
        for (auto & L : layers) {
            L.ids_host_pin = slice; L.ids_host_cap = per_layer; slice += per_layer;
        }
    }
    LLAMA_LOG_INFO("%s: expert pool: %.1f MiB page-locked staging in %s (kernel-copy downloads and id uploads%s)\n",
        __func__, bytes / (1024.0 * 1024.0), ggml_backend_buft_name(buft),
        ids_host_buf != nullptr ? ", router-ids landing mapped for the device" : "; router-ids landing NOT mappable");
}

ggml_tensor * llama_expert_pool::ids_host_tensor(int32_t il, ggml_context * ctx, const ggml_tensor * ids) {
    if (!layer_pooled(il) || backend_router == nullptr || ids == nullptr) {
        return nullptr;
    }
    ensure_read_staging(backend_router);
    layer_state & L = layers[il];
    if (ids_host_buf == nullptr || L.ids_host_pin == nullptr || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || (size_t) (ids->ne[0] * ids->ne[1]) > L.ids_host_cap) {
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
    t->buffer = ids_host_buf;
    t->data   = (char *) ggml_backend_buffer_get_base(ids_host_buf) + ((char *) L.ids_host_pin - ids_host_base);
    return t;
}

void llama_expert_pool::arm_ids_host() {
    if (ids_host_buf == nullptr) {
        return;
    }
    if (ids_host_unconsumed > 0) {
        // a previous pass left armed slices (aborted graph): let its copies land before re-arming
        ggml_backend_synchronize(backend_router);
        ids_host_unconsumed = 0;
    }
    for (auto & L : layers) {
        if (L.ids_host == nullptr || L.ids_host_pin == nullptr) {
            continue;
        }
        std::fill(L.ids_host_pin, L.ids_host_pin + L.ids_host_n, -1);
        L.ids_host_armed = true;
        ids_host_unconsumed++;
    }
}

void llama_expert_pool::hoist_independent(ggml_cgraph * gf) {
    llama_pshard_hoist_independent(gf, this, nullptr, &hoisted_nodes, &hoist_regions);
}

void llama_pshard_hoist_independent(ggml_cgraph * gf, llama_expert_pool * pool, ggml_backend_sched_t sched,
                                    uint32_t * hoisted, uint32_t * regions) {
    if (sched != nullptr) {
        ggml_backend_sched_clear_split_before(sched);
    }
    uint32_t hoisted_nodes = 0;
    uint32_t hoist_regions = 0;
    if (hoisted) { *hoisted = 0; }
    if (regions) { *regions = 0; }
    if (gf == nullptr) {
        return;
    }
    const int n = ggml_graph_n_nodes(gf);
    if (n == 0) {
        return;
    }
    ggml_tensor ** nodes = ggml_graph_nodes(gf);

    auto root = [](const ggml_tensor * t) {
        while (t->view_src != nullptr) {
            t = t->view_src;
        }
        return t;
    };

    // late tensors: host-resident weights (a CPU-computed or streamed layer's first read of one is the boundary
    // the scheduler cuts at), keyed by their layer, plus the pool-served weights and the ids leaves serve() writes
    std::unordered_map<const ggml_tensor *, int32_t> late;   // late tensor -> its layer
    for (int i = 0; i < n; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = nodes[i]->src[s];
            if (src == nullptr) {
                continue;
            }
            const ggml_tensor * r = root(src);
            if (r->op != GGML_OP_NONE || r->buffer == nullptr || late.count(r)) {
                continue;
            }
            if (!ggml_backend_buffer_is_host(r->buffer) || ggml_backend_buffer_get_usage(r->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            int il = -1;
            if (sscanf(r->name, "blk.%d.", &il) == 1 && il >= 0) {
                late[r] = il;
            }
        }
    }
    std::unordered_set<const ggml_tensor *> pool_late;   // the pool's own late tensors: the pool service cuts there itself
    if (pool != nullptr && pool->active) {
        for (const auto & L : pool->layers) {
            for (const auto & e : L.tensors) {
                late[e.host] = L.il;
                pool_late.insert(e.host);
            }
            if (L.ids_gpu      != nullptr) { late[L.ids_gpu]      = L.il; pool_late.insert(L.ids_gpu); }
            if (L.ids_gpu_bias != nullptr) { late[L.ids_gpu_bias] = L.il; pool_late.insert(L.ids_gpu_bias); }
            if (L.ids_cpu      != nullptr) { late[L.ids_cpu]      = L.il; pool_late.insert(L.ids_cpu); }
        }
    }
    if (late.empty()) {
        return;
    }

    auto is_view_op = [](ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    };

    // one region per pooled layer's boundary: dependents of the boundary are held in order; nodes whose inputs
    // are all available before it are hoisted ahead of the held ones
    std::unordered_set<const ggml_tensor *> done;      // results already emitted in the new order (or leaves)
    std::unordered_set<const ggml_tensor *> tainted;   // results downstream of this region's boundary
    std::vector<ggml_tensor *> out;
    std::vector<ggml_tensor *> held;
    out.reserve(n);
    ggml_tensor * region_first_hoisted = nullptr;   // first node hoisted in the open region
    ggml_tensor * region_boundary      = nullptr;   // the node that opened it
    bool          region_pool          = false;     // it opened on a pool-served tensor

    // the layer whose late tensor this node reads directly (-1: none)
    auto direct_late = [&](const ggml_tensor * t) -> int32_t {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = t->src[s];
            if (src == nullptr) {
                continue;
            }
            auto it = late.find(src);
            if (it == late.end()) {
                it = late.find(root(src));
            }
            if (it != late.end()) {
                return it->second;
            }
        }
        return -1;
    };
    // downstream of this region's boundary through a result, or an in-place op on such a tensor
    auto reads_tainted = [&](const ggml_tensor * t) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = t->src[s];
            if (src != nullptr && (tainted.count(src) || tainted.count(root(src)))) {
                return true;
            }
        }
        return t->view_src != nullptr && tainted.count(root(t)) > 0;
    };
    auto available = [&](const ggml_tensor * src) {
        if (src->op == GGML_OP_NONE) {
            return !late.count(src) && !late.count(root(src));   // a leaf that is not written by the service
        }
        return done.count(src) > 0;
    };
    auto hoistable = [&](const ggml_tensor * t) {
        if (t->view_src != nullptr && !is_view_op(t->op)) {
            return false;   // an in-place op: it writes into a tensor the held nodes may still read
        }
        if (t->view_src != nullptr) {
            // a pure view moves only when its root is an emitted result
            const ggml_tensor * r = root(t);
            if (r->op == GGML_OP_NONE || !done.count(r)) {
                return false;
            }
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = t->src[s];
            if (src != nullptr && !available(src)) {
                return false;
            }
        }
        // write-after-read: a held node that writes in place into a tensor this node reads must stay before it
        for (const ggml_tensor * h : held) {
            if (h->view_src == nullptr || is_view_op(h->op)) {
                continue;
            }
            const ggml_tensor * w = root(h);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                const ggml_tensor * src = t->src[s];
                if (src != nullptr && root(src) == w) {
                    return false;
                }
            }
        }
        return true;
    };
    auto flush = [&]() {
        for (ggml_tensor * h : held) {
            out.push_back(h);
            done.insert(h);
        }
        held.clear();
        tainted.clear();
        // a region whose boundary computes on the CPU: cut a split before its first hoisted node, so the scheduler
        // fetches the CPU split's inputs before the hoisted nodes launch and the CPU works while they run
        if (sched != nullptr && region_first_hoisted != nullptr && region_boundary != nullptr && !region_pool) {
            ggml_backend_sched_set_split_before(sched, region_first_hoisted, region_boundary);
        }
        region_first_hoisted = nullptr;
        region_boundary      = nullptr;
        region_pool          = false;
    };

    int32_t region_il = -1;   // the layer whose routed boundary opened the current region (-1: none open)
    for (int i = 0; i < n; i++) {
        ggml_tensor * t = nodes[i];
        const int32_t il = direct_late(t);
        if (il >= 0 && il != region_il) {
            // a new layer's boundary: flush the previous region in its original order
            flush();
            region_il = il;
            hoist_regions++;
            region_boundary = t;
            for (int s = 0; s < GGML_MAX_SRC && !region_pool; s++) {
                const ggml_tensor * src = t->src[s];
                region_pool = src != nullptr && (pool_late.count(src) || pool_late.count(root(src)));
            }
        }
        if (il >= 0 || (region_il >= 0 && reads_tainted(t))) {
            held.push_back(t);   // the routed chain and everything downstream of it
            tainted.insert(t);
            continue;
        }
        if (region_il < 0) {
            out.push_back(t);    // before the first boundary: untouched
            done.insert(t);
            continue;
        }
        if (hoistable(t)) {
            out.push_back(t);    // independent of the routed experts: runs while the host decides
            done.insert(t);
            hoisted_nodes++;
            if (region_first_hoisted == nullptr) {
                region_first_hoisted = t;
            }
        } else {
            held.push_back(t);   // depends on a held node, or writes in place: keeps its order
        }
    }
    flush();
    GGML_ASSERT((int) out.size() == n);
    for (int i = 0; i < n; i++) {
        nodes[i] = out[i];
    }
    if (hoisted) { *hoisted = hoisted_nodes; }
    if (regions) { *regions = hoist_regions; }
    if (hoisted_nodes > 0) {
        LLAMA_LOG_INFO("%s: pshard: %u nodes hoisted ahead of %u host-weight boundaries\n",
            __func__, hoisted_nodes, hoist_regions);
    }
}

bool llama_expert_pool::wait_ids_host(layer_state & L, size_t n) {
    // ids are >= 0, so -1 marks an entry the graph copy has not written yet
    const volatile int32_t * p = L.ids_host_pin;
    const int64_t t0 = ggml_time_us();
    size_t i = 0;
    bool drained = false;
    for (;;) {
        while (i < n && p[i] != -1) {
            i++;
        }
        if (i == n) {
            break;
        }
        if (!drained && ggml_time_us() - t0 > 50000) {
            ggml_backend_synchronize(backend_router);   // large batches: the router is far down the queue
            drained = true;
            ids_host_drains++;
            continue;
        }
        if (drained) {
            ids_host_fallbacks++;
            return false;
        }
    }
    ids_host_polls++;
    ids_host_wait_us += (uint64_t) (ggml_time_us() - t0);
    return true;
}

bool llama_expert_pool::serve(const ggml_tensor * src, ggml_tensor * view, ggml_backend_t split_backend) {
    GGML_UNUSED(view);
    layer_state * Lp = nullptr;
    for (auto & L : layers) {
        for (const auto & e : L.tensors) {
            if (e.host == src) {
                Lp = &L;
                break;
            }
        }
        if (Lp != nullptr) {
            break;
        }
    }
    if (Lp == nullptr) {
        return false;
    }
    layer_state & L = *Lp;

    if (L.ids_router == nullptr || L.ids_gpu == nullptr) {
        LLAMA_LOG_WARN("%s: pool layer %d has no bound ids tensors\n", __func__, L.il);
        return false;
    }
    // one full pass per layer per generation; the layer's other expert tensors
    // arrive as further inputs of the same split and are already served
    if (generation > 0 && L.serve_gen == generation) {
        return true;
    }

    // read the router ids (device, produced by an earlier split on the compute
    // backend; same read pattern as the sched's sliced-expert path)
    const ggml_tensor * ids = L.ids_router;
    const int64_t n_ids_0 = ids->ne[0]; // n_expert_used
    const int64_t n_ids_1 = ids->ne[1]; // n_tokens
    ensure_read_staging(backend_router);
    char * idbuf = nullptr;
    size_t nb0 = ids->nb[0];
    size_t nb1 = ids->nb[1];
    bool landed = false;
    if (L.ids_host != nullptr && L.ids_host_armed && L.ids_host_pin != nullptr &&
        (size_t) (n_ids_0 * n_ids_1) == L.ids_host_n) {
        // the graph copied the ids into the pinned slice right after the router: poll instead of draining the stream
        L.ids_host_armed = false;
        landed = wait_ids_host(L, (size_t) (n_ids_0 * n_ids_1));
        if (landed && ids_host_unconsumed > 0) {
            ids_host_unconsumed--;   // an unlanded slice stays counted so the next arm drains first
        }
        if (landed) {
            idbuf = (char *) L.ids_host_pin;
            nb0 = sizeof(int32_t);
            nb1 = (size_t) n_ids_0 * sizeof(int32_t);
        }
    }
    if (!landed) {
        if (read_ids != nullptr && ggml_nbytes(ids) <= read_cap) {
            idbuf = read_ids;
        } else {
            ids_read_buf.resize(ggml_nbytes(ids));
            idbuf = ids_read_buf.data();
        }
        ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(ids), idbuf, 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend_router);
    }

    // background admission: slots filled on the copy stream last pass must have
    // landed before any kernel of this pass reads them (one wait per pass, first
    // pooled layer; every later kernel queues behind it on the split stream)
    if (admit_pending && admit_event != nullptr) {
        ggml_backend_event_wait(split_backend, admit_event);
        admit_pending = false;
    }

    // per-layer persistent upload buffer: the async staging worker may queue this
    // host pointer behind pending staged fetches and read it after serve() returns
    L.mapped_buf.assign((size_t) n_ids_0 * n_ids_1, 0);
    ids_buf & mapped = L.mapped_buf;

    if (ab_mode) {
        // whole-stack tier: fill this layer's half once per pass, identity ids
        if (L.ab_pass != generation || generation == 0) {
            for (const auto & e : L.tensors) {
                ggml_backend_tensor_set_async(split_backend, e.view_ab,
                    e.host->data, 0, (size_t) n_expert * e.row_bytes);
            }
            L.ab_pass = generation;
        }
        for (int64_t i1 = 0; i1 < n_ids_1; i1++) {
            for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                const int32_t e = *(const int32_t *) (idbuf + i1*nb1 + i0*nb0);
                mapped[i1*n_ids_0 + i0] = e;
            }
        }
        if (cpu_routes() && L.ids_cpu != nullptr) {
            // dual chain on a whole-stack tier: everything is resident, nothing goes to CPU
            L.bias_buf = mapped;
            L.cpu_buf.assign(mapped.size(), -1);
            L.cpu_dead     = true;
            L.cpu_dead_gen = generation;
            L.cpu_dead_passes++;
            if (L.ids_gpu_bias != nullptr) {
                ggml_backend_tensor_set_async(split_backend, L.ids_gpu_bias,
                    L.bias_buf.data(), 0, L.bias_buf.size() * sizeof(int32_t));
            }
            ggml_backend_tensor_set(L.ids_cpu, L.cpu_buf.data(), 0, L.cpu_buf.size() * sizeof(int32_t));
        }
    } else {
        // 1. the pass's distinct experts: hits stay; misses are decided PER EXPERT
        //    by the tier's miss policy (all of an expert's routes go the same way)
        seen_gen.assign(n_expert, 0);
        std::vector<int32_t> miss_list;
        uint32_t n_hit = 0;
        for (int64_t i1 = 0; i1 < n_ids_1; i1++) {
            for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                const int32_t e = *(const int32_t *)
                    (idbuf + i1*nb1 + i0*nb0);
                GGML_ASSERT(e >= 0 && e < (int32_t) n_expert);
                if (!L.use_count.empty()) {
                    L.use_count[e]++;   // routing workload histogram: every route, before the per-pass dedup
                }
                if (seen_gen[e] != 0) {
                    continue;
                }
                seen_gen[e] = 1;
                if (L.expert_slot[e] >= 0) {
                    // refresh NOW: the fetch loop below picks LRU victims, and a
                    // same-pass hit must never be one
                    L.slot_stamp[L.expert_slot[e]] = ++L.stamp;
                    n_hit++;
                    if (!L.slot_pf_gen.empty() && L.slot_pf_gen[L.expert_slot[e]] == generation) {
                        L.pf_used++;   // a prefetch from this pass paid off
                    }
                } else {
                    miss_list.push_back(e);
                }
            }
        }
        // prediction made predict_k layers earlier for THIS layer: how many of the
        // routes did it name, and how many of the misses (residency as of now)
        if (L.ids_pred != nullptr && predict_k > 0) {
            const ggml_tensor * pt = L.ids_pred;
            char * predbuf = read_pred;
            if (predbuf == nullptr || ggml_nbytes(pt) > read_cap) {
                pred_read_buf.resize(ggml_nbytes(pt));
                predbuf = pred_read_buf.data();
            }
            ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(pt), predbuf, 0, ggml_nbytes(pt));
            ggml_backend_synchronize(backend_router);
            const int64_t np1 = std::min<int64_t>(pt->ne[1], n_ids_1);
            for (int64_t i1 = 0; i1 < np1; i1++) {
                for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                    const int32_t e = *(const int32_t *) (idbuf + i1*nb1 + i0*nb0);
                    bool in_pred = false;
                    for (int64_t j0 = 0; j0 < pt->ne[0] && !in_pred; j0++) {
                        in_pred = *(const int32_t *) (predbuf + i1*pt->nb[1] + j0*pt->nb[0]) == e;
                    }
                    L.pred_total++;
                    if (in_pred) {
                        L.pred_hit++;
                    }
                    if (e >= 0 && e < (int32_t) n_expert && L.expert_slot[e] < 0) {
                        L.pred_misses++;
                        if (in_pred) {
                            L.pred_covered++;
                        }
                    }
                }
            }
        }

        const bool dual = cpu_routes() && L.ids_cpu != nullptr;
        L.cache_passes++;

        // 2. which misses get fetched (admitted) vs computed on CPU
        //    fetch:             all (the floor guarantees the slots)
        //    cpu_exec:          none
        //    fetch_on_2nd_miss: only experts that missed before (admission filter)
        //    hybrid:            the q* = round(m * B_P/B_H) most recently active misses
        std::vector<uint8_t> admit(miss_list.size(), 1);
        if (dual) {
            const size_t m = miss_list.size();
            if (miss_policy == 1) {                       // cpu_exec
                std::fill(admit.begin(), admit.end(), 0);
            } else if (miss_policy == 2) {                // fetch_on_2nd_miss
                for (size_t i = 0; i < m; i++) {
                    const int32_t e = miss_list[i];
                    admit[i] = L.miss_count[e] > 0 ? 1 : 0;   // counted in the fetch loop below
                }
            } else if (miss_policy == 3) {                // hybrid
                size_t q = (size_t) (hybrid_frac * (double) m + 0.5);
                if (m > 0 && q == 0) q = 1;
                if (q > m) q = m;
                // rank misses by recency (most recently active first), fetch the top q
                std::vector<size_t> order(m);
                for (size_t i = 0; i < m; i++) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return L.expert_last_gen[miss_list[a]] > L.expert_last_gen[miss_list[b]];
                });
                std::fill(admit.begin(), admit.end(), 0);
                for (size_t i = 0; i < q; i++) admit[order[i]] = 1;
            }
        }
        // slot capacity: this pass's hits + admitted misses must all be resident at
        // once (one MUL_MAT_ID per chain). Overflow spills to CPU when a CPU chain
        // exists; a fetch-only tier fails loudly instead of evicting same-pass rows.
        {
            uint32_t need = n_hit;
            for (size_t i = 0; i < miss_list.size(); i++) {
                if (admit[i] && need < L.n_slots_l) {
                    need++;
                } else if (admit[i]) {
                    if (!dual) {
                        LLAMA_LOG_ERROR("%s: pool layer %d: distinct experts this pass exceed %u slots - "
                            "the runtime clamp undercut the plan floor\n", __func__, L.il, L.n_slots_l);
                        return false;
                    }
                    admit[i] = 0;
                }
            }
        }

        // 3. fetch the admitted misses into LRU victims (same-pass residents carry
        //    the newest stamps, so they are never chosen). cpu_admit uploads on the
        //    pool's copy backend (off the critical path) and routes the expert to the
        //    CPU chain THIS pass; the slot is read from the next pass on (event below).
        //    The victim is safe to overwrite concurrently: it is not routed this pass
        //    and the synchronous ids read above drained every earlier kernel.
        const bool background = miss_policy == LLAMA_PSHARD_MISS_CPU_ADMIT;
        if (background) {
            ensure_admit_backend(split_backend);
        }
        ggml_backend_t up_backend  = (background && admit_backend != nullptr && admit_event != nullptr) ? admit_backend : split_backend;
        bool           uploaded_bg = false;
        for (size_t i = 0; i < miss_list.size(); i++) {
            const int32_t e = miss_list[i];
            L.expert_last_gen[e] = generation;
            ++L.miss_count[e];   // fetch_on_2nd_miss admission reads this
            if (!admit[i]) {
                L.misses++;
                continue;
            }
            int32_t slot = 0;
            for (uint32_t s = 1; s < L.n_slots_l; s++) {
                if (L.slot_stamp[s] < L.slot_stamp[slot]) {
                    slot = (int32_t) s;
                }
            }
            if (L.slot_expert[slot] >= 0) {
                L.expert_slot[L.slot_expert[slot]] = -1;
                L.evicted++;
            }
            L.slot_expert[slot] = e;
            L.expert_slot[e]    = slot;
            L.slot_stamp[slot]  = ++L.stamp;   // protected for THIS pass
            for (const auto & te : L.tensors) {
                // collected: the pass's rows go up in one launch below (per-tensor set_async fallback)
                upload_segs.push_back({ te.view_slots, (size_t) slot * te.row_bytes,
                    (const char *) te.host->data + (size_t) e * te.row_bytes, te.row_bytes });
            }
            if (background) {
                L.expert_pending[e] = generation;   // CPU route this pass, GPU hit from the next
                uploaded_bg = uploaded_bg || up_backend == admit_backend;
            }
            L.misses++;
        }
        if (!upload_segs.empty()) {
            bool batched = false;
            if (copy_segments != nullptr && kernel_copies) {
                std::vector<ggml_backend_copy_segment> segs(upload_segs.size());
                for (size_t i = 0; i < upload_segs.size(); i++) {
                    segs[i] = { (char *) upload_segs[i].view->data + upload_segs[i].off, upload_segs[i].src, upload_segs[i].size };
                }
                batched = copy_segments(up_backend, segs.data(), (int) segs.size());
            }
            if (!batched) {
                for (const auto & sg : upload_segs) {
                    ggml_backend_tensor_set_async(up_backend, sg.view, sg.src, sg.off, sg.size);
                }
            }
            upload_segs.clear();
        }
        if (uploaded_bg && admit_event != nullptr) {
            ggml_backend_event_record(admit_event, admit_backend);
            admit_pending = true;
        }

        // 4. per-route ids: GPU mm gets slot | -1, GPU bias gets expert | -1, the
        //    CPU chain gets expert | -1 (complement)
        if (dual) {
            L.bias_buf.assign(mapped.size(), -1);
            L.cpu_buf.assign(mapped.size(), -1);
        }
        for (int64_t i1 = 0; i1 < n_ids_1; i1++) {
            for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                const int32_t e = *(const int32_t *)
                    (idbuf + i1*nb1 + i0*nb0);
                const int32_t slot = L.expert_slot[e];
                const size_t  k    = (size_t) (i1*n_ids_0 + i0);
                L.expert_last_gen[e] = generation;
                const bool pending = miss_policy == LLAMA_PSHARD_MISS_CPU_ADMIT &&
                                     !L.expert_pending.empty() && L.expert_pending[e] == generation;
                if (slot >= 0 && !pending) {
                    L.slot_stamp[slot] = ++L.stamp;
                    mapped[k] = slot;
                    if (dual) {
                        L.bias_buf[k] = e;
                    }
                } else {
                    GGML_ASSERT(dual && "unadmitted miss without a CPU chain");
                    mapped[k] = -1;
                    L.cpu_buf[k] = e;
                }
            }
        }
        L.hits += n_hit;
        if (n_ids_1 == 1) {
            L.passes_1++;
            L.hits_1   += n_hit;
            L.misses_1 += miss_list.size();
        }
        if (dual) {
            if (L.ids_gpu_bias != nullptr) { // only exists when an expert bias consumes it
                ggml_backend_tensor_set_async(split_backend, L.ids_gpu_bias,
                    L.bias_buf.data(), 0, L.bias_buf.size() * sizeof(int32_t));
            }
            // the CPU chain's leaf lives in host memory: plain synchronous set
            ggml_backend_tensor_set(L.ids_cpu, L.cpu_buf.data(), 0, L.cpu_buf.size() * sizeof(int32_t));
            // nothing routed to the CPU this pass: the sched may skip the chain (split_is_zero)
            L.cpu_dead     = std::all_of(L.cpu_buf.begin(), L.cpu_buf.end(), [](int32_t e) { return e < 0; });
            L.cpu_dead_gen = generation;
            if (L.cpu_dead) {
                L.cpu_dead_passes++;
            }
        }

        // 5. prefetch for layer il+k. The prediction for that layer was computed in
        //    THIS layer's compute split (ready: the ids read above drained the stream).
        //    Upload its non-resident experts into that layer's LRU victims on the copy
        //    stream now, about one layer ahead of their use; at il+k's service they are
        //    hits once the split stream has waited on the admit event (top of serve()).
        //    The victim is safe: the target layer's previous pass has completed (the
        //    drain above) and nothing reads its slots before its own service.
        if (predict_k > 0 && prefetch_on && n_ids_1 <= 8) {   // decode and small verify batches
            const int32_t tgt = L.il + predict_k;
            if (layer_pooled(tgt) && layers[tgt].ids_pred != nullptr) {
                layer_state & T = layers[tgt];
                if (T.slot_expert.size() == T.n_slots_l && T.n_slots_l > 0 && !T.expert_slot.empty()) {
                    const ggml_tensor * pt = T.ids_pred;
                    char * predbuf = read_pred;
                    if (predbuf == nullptr || ggml_nbytes(pt) > read_cap) {
                        pred_read_buf.resize(ggml_nbytes(pt));
                        predbuf = pred_read_buf.data();
                    }
                    ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(pt), predbuf, 0, ggml_nbytes(pt));
                    ggml_backend_synchronize(backend_router);
                    ensure_admit_backend(split_backend);
                    if (admit_backend != nullptr && admit_event != nullptr) {
                        bool issued = false;
                        int32_t n_issued = 0;
                        // the prediction is in descending score order per token: with a cap,
                        // rank j0 of every token goes before rank j0+1 (mispredicted uploads
                        // compete with the critical-path misses for the same PCIe link)
                        const int64_t np1 = std::min<int64_t>(pt->ne[1], n_ids_1);
                        for (int64_t j0 = 0; j0 < pt->ne[0]; j0++) {
                        for (int64_t i1 = 0; i1 < np1; i1++) {
                            if (prefetch_n > 0 && n_issued >= prefetch_n * np1) {
                                break;
                            }
                            const int32_t e = *(const int32_t *) (predbuf + i1*pt->nb[1] + j0*pt->nb[0]);
                            if (e < 0 || e >= (int32_t) n_expert || T.expert_slot[e] >= 0) {
                                continue;
                            }
                            n_issued++;
                            int32_t slot = 0;
                            for (uint32_t s2 = 1; s2 < T.n_slots_l; s2++) {
                                if (T.slot_stamp[s2] < T.slot_stamp[slot]) {
                                    slot = (int32_t) s2;
                                }
                            }
                            if (T.slot_expert[slot] >= 0) {
                                T.expert_slot[T.slot_expert[slot]] = -1;
                                T.evicted++;
                            }
                            T.slot_expert[slot] = e;
                            T.expert_slot[e]    = slot;
                            T.slot_stamp[slot]  = ++T.stamp;
                            T.slot_pf_gen[slot] = generation;
                            for (const auto & te : T.tensors) {
                                ggml_backend_tensor_set_async(admit_backend, te.view_slots,
                                    (const char *) te.host->data + (size_t) e * te.row_bytes,
                                    (size_t) slot * te.row_bytes, te.row_bytes);
                            }
                            T.pf_issued++;
                            issued = true;
                        }
                        }
                        if (issued) {
                            ggml_backend_event_record(admit_event, admit_backend);
                            admit_pending = true;
                        }
                    }
                }
            }
        }
    }

    ggml_backend_tensor_set_async(split_backend, L.ids_gpu,
        mapped.data(), 0, mapped.size() * sizeof(int32_t));

    L.serve_gen = generation;
    return true;
}

void llama_expert_pool::reset_slots() {
    // background admission: uploads still in flight target the OLD slot layout;
    // let them land before the maps (and possibly the region) change under them
    if (admit_backend != nullptr) {
        ggml_backend_synchronize(admit_backend);
    }
    admit_pending = false;
    for (auto & L : layers) {
        if (L.tensors.empty()) {
            continue;
        }
        std::fill(L.expert_slot.begin(), L.expert_slot.end(), -1);
        std::fill(L.slot_expert.begin(), L.slot_expert.end(), -1);
        std::fill(L.slot_stamp.begin(),  L.slot_stamp.end(),  0);
        L.stamp   = 0;
        L.ab_pass = 0;
    }
}

void llama_expert_pool::log_counters() const {
    // the headline lines print at WARN so runs without -lv still report them (the pricing model is
    // calibrated from these); per-layer detail stays at INFO
    uint64_t hits = 0, misses = 0, passes = 0;
    uint64_t hits_1 = 0, misses_1 = 0, passes_1 = 0, evicted = 0;
    std::string per_layer;
    for (const auto & L : layers) {
        if (L.tensors.empty()) {
            continue;
        }
        hits     += L.hits;
        misses   += L.misses;
        passes    = std::max(passes, L.cache_passes);
        hits_1   += L.hits_1;
        misses_1 += L.misses_1;
        passes_1  = std::max(passes_1, L.passes_1);
        evicted  += L.evicted;
        const uint64_t n = L.hits + L.misses;
        char buf[16];
        snprintf(buf, sizeof(buf), " %.2f", n > 0 ? (double) L.hits / (double) n : 0.0);
        per_layer += buf;
    }
    if (ids_host_polls + ids_host_fallbacks > 0) {
        // before the cache-mode return: whole-stack passes have no hits/misses
        LLAMA_LOG_WARN("%s: expert pool: router ids landed in host memory %llu times (mean poll %.1f us, %llu drained the stream first), %llu fell back to the stream drain; last graph: %u nodes hoisted ahead of %u routed boundaries\n",
            __func__, (unsigned long long) ids_host_polls,
            ids_host_polls > 0 ? (double) ids_host_wait_us / (double) ids_host_polls : 0.0,
            (unsigned long long) ids_host_drains, (unsigned long long) ids_host_fallbacks, hoisted_nodes, hoist_regions);
    }
    if (hits + misses == 0) {
        return; // never served in cache mode (or pool never engaged)
    }
    const double h   = (double) hits / (double) (hits + misses);
    const double mpt = passes > 0 ? (double) misses / (double) passes : 0.0;
    // cache-mode decode counters: distinct experts per layer per pass; misses/token
    // is per pass (= per token at bs=1), summed over the pooled layers
    LLAMA_LOG_WARN("%s: expert pool: %llu hits / %llu misses over %llu passes: h=%.3f misses/token=%.1f (s=%u)\n",
        __func__, (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) passes, h, mpt, n_slots);
    LLAMA_LOG_INFO("%s: expert pool h per layer:%s\n", __func__, per_layer.c_str());
    LLAMA_LOG_INFO("%s: expert pool admission: %llu residents evicted\n",
        __func__, (unsigned long long) evicted);
    if (cpu_routes()) {
        uint64_t dead = 0;
        for (const auto & L : layers) {
            dead += L.cpu_dead_passes;
        }
        LLAMA_LOG_INFO("%s: expert pool CPU chains: %llu of %llu passes routed nothing to the CPU, %llu chains skipped\n",
            __func__, (unsigned long long) dead, (unsigned long long) passes, (unsigned long long) skipped_splits);
    }
    if (predict_k > 0) {
        uint64_t pt = 0, ph = 0, pm = 0, pc = 0;
        std::string per_layer_cov;
        for (const auto & L : layers) {
            if (L.pred_total == 0) {
                continue;
            }
            pt += L.pred_total; ph += L.pred_hit; pm += L.pred_misses; pc += L.pred_covered;
            char buf[16];
            snprintf(buf, sizeof(buf), " %.2f", L.pred_misses > 0 ? (double) L.pred_covered / (double) L.pred_misses : 0.0);
            per_layer_cov += buf;
        }
        LLAMA_LOG_INFO("%s: expert pool prediction (k=%d): recall %.3f (%llu/%llu routes), miss coverage %.3f (%llu/%llu misses)\n",
            __func__, predict_k, pt > 0 ? (double) ph / (double) pt : 0.0, (unsigned long long) ph, (unsigned long long) pt,
            pm > 0 ? (double) pc / (double) pm : 0.0, (unsigned long long) pc, (unsigned long long) pm);
        LLAMA_LOG_INFO("%s: expert pool prediction miss coverage per layer:%s\n", __func__, per_layer_cov.c_str());
        uint64_t pfi = 0, pfu = 0;
        for (const auto & L : layers) {
            pfi += L.pf_issued;
            pfu += L.pf_used;
        }
        LLAMA_LOG_WARN("%s: expert pool prefetch: %s, %llu uploads issued, %llu used by the target layer (%.3f)\n",
            __func__, prefetch_on ? "on" : "off", (unsigned long long) pfi, (unsigned long long) pfu,
            pfi > 0 ? (double) pfu / (double) pfi : 0.0);
    }
    if (passes_1 > 0 && passes_1 != passes) {
        // decode-only view (single-token passes): what the planner's h(s) and
        // misses/token model; the all-passes line above includes any multi-token
        // ubatch a cache-mode tier served (e.g. the prompt when the prefill tiers
        // fell back onto the decode plan). The QA ledger takes the LAST line.
        const double h1   = hits_1 + misses_1 > 0 ? (double) hits_1 / (double) (hits_1 + misses_1) : 0.0;
        const double mpt1 = (double) misses_1 / (double) passes_1;
        LLAMA_LOG_WARN("%s: expert pool decode: %llu hits / %llu misses over %llu passes: h=%.3f misses/token=%.1f (s=%u)\n",
            __func__, (unsigned long long) hits_1, (unsigned long long) misses_1, (unsigned long long) passes_1, h1, mpt1, n_slots);
    }
    // routing workload: this run's cache-mode route histogram goes into <model>.pshard_workload. Real runs
    // accumulate (one short run has too few routes per expert; the split-half fit is unbiased at small samples
    // and the store grows with every run); a file holding only the plan-time calibration stand-in is replaced by
    // the first real run. The 64-pass floor keeps trivial runs out. The planner prices h(s) with the fitted exponent.
    if (passes >= 64) {
        std::vector<std::pair<uint32_t, std::vector<uint64_t>>> mine;
        uint64_t mine_routes = 0;
        for (const auto & L : layers) {
            if (!L.tensors.empty() && !L.use_count.empty()) {
                mine.push_back({ (uint32_t) L.il, L.use_count });
                for (uint64_t v : L.use_count) {
                    mine_routes += v;
                }
            }
        }
        llama_pshard_workload wl;
        const bool had = !workload_path.empty() && wl.load(workload_path);
        const bool ok  = (had && wl.source == "runtime" && wl.n_expert == n_expert)
            ? wl.merge_counts(mine)
            : wl.set_counts(mine, n_expert, "runtime");
        if (ok) {
            const bool saved = !workload_path.empty() && wl.save(workload_path);
            LLAMA_LOG_WARN("%s: expert pool workload: zipf_alpha=%.3f (rms %.4f, %llu routes accumulated, %llu this run, %u layers): model h(%u)=%.3f vs observed %.3f%s%s\n",
                __func__, wl.zipf_alpha, wl.fit_rms, (unsigned long long) wl.samples, (unsigned long long) mine_routes, wl.n_layers,
                n_slots, llama_pshard_workload::zipf_h(wl.zipf_alpha, n_slots, n_expert), h,
                saved ? " -> " : " (not saved)", saved ? workload_path.c_str() : "");
        }
    }
}
