#include "llama-expert-pool.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>

llama_expert_pool::~llama_expert_pool() {
    if (ctx_views != nullptr) {
        ggml_free(ctx_views);
    }
}

bool llama_expert_pool::init(const llama_model & model, uint32_t n_expert_, uint32_t n_expert_used_) {
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
            tensor_entry e;
            e.host      = t;
            e.row_bytes = t->nb[2];
            full += e.row_bytes * n_expert;
            L.tensors.push_back(e);
        }
        if (!L.tensors.empty()) {
            n_pooled++;
            layer_full_bytes = std::max(layer_full_bytes, full);
            L.expert_slot.assign(n_expert, -1);
        }
    }

    LLAMA_LOG_INFO("%s: expert pool: %zu pooled layers, %u experts (%u used), max layer %.1f MiB\n",
        __func__, n_pooled, n_expert, n_expert_used, layer_full_bytes / (1024.0 * 1024.0));
    return n_pooled > 0;
}

size_t llama_expert_pool::region_bytes_needed(uint32_t slots_per_layer) const {
    // cache mode: slots_per_layer rows of every pooled tensor, per layer;
    // A/B mode reuses the same span (2 whole layers) - take the max
    size_t cache_bytes = 0;
    for (const auto & L : layers) {
        for (const auto & e : L.tensors) {
            cache_bytes += ((size_t) slots_per_layer * e.row_bytes + 255) & ~(size_t) 255;
        }
    }
    return std::max(cache_bytes, 2 * layer_full_bytes);
}

bool llama_expert_pool::set_region(ggml_backend_buffer_t arena, void * base, size_t bytes, uint32_t slots_per_layer) {
    region_base  = base;
    region_bytes = bytes;
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

    const size_t need = region_bytes_needed(slots_per_layer);
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

    // cache-mode layout: layer-major, per-tensor slot arrays; A/B halves overlay
    // the region start (they are only live on whole-stack prefill tiers, where the
    // cache contents are volatile by design)
    size_t off = 0;
    for (auto & L : layers) {
        L.slot_expert.assign(n_slots, -1);
        L.slot_stamp.assign(n_slots, 0);
        if (!L.expert_slot.empty()) {
            std::fill(L.expert_slot.begin(), L.expert_slot.end(), -1);
        }
        for (auto & e : L.tensors) {
            e.region_off = off;
            off += ((size_t) n_slots * e.row_bytes + 255) & ~(size_t) 255;

            e.view_slots = ggml_new_tensor_3d(ctx_views, e.host->type,
                e.host->ne[0], e.host->ne[1], n_slots);
            e.view_slots->data   = (char *) base + e.region_off;
            e.view_slots->buffer = arena;
            ggml_format_name(e.view_slots, "pool_s#%s", e.host->name);
        }
    }

    // A/B halves: parity-alternating whole-layer sets at the region start
    for (auto & L : layers) {
        size_t sub = 0;
        const size_t half = layer_full_bytes;
        for (auto & e : L.tensors) {
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

    LLAMA_LOG_INFO("%s: expert pool region %.1f MiB: s=%u slots/layer (cache) or 2x%.1f MiB A/B halves\n",
        __func__, bytes / (1024.0 * 1024.0), n_slots, layer_full_bytes / (1024.0 * 1024.0));
    return true;
}

void llama_expert_pool::register_sched(ggml_backend_sched_t sched) {
    ggml_backend_sched_clear_input_copy_overrides(sched);
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
}

void llama_expert_pool::set_ab_mode(bool ab, ggml_backend_sched_t sched) {
    if (ab == ab_mode) {
        return;
    }
    ab_mode = ab;
    // v1 relabel: cache contents do not survive the A/B overlay (the halves alias
    // the slot arrays); drop the maps and refill lazily
    reset_slots();
    if (sched != nullptr) {
        register_sched(sched);
    }
}

void llama_expert_pool::bind_layer_ids(int32_t il, ggml_tensor * ids_router, ggml_tensor * ids_gpu) {
    if (!layer_pooled(il)) {
        return;
    }
    layers[il].ids_router = ids_router;
    layers[il].ids_gpu    = ids_gpu;
    layers[il].serve_gen  = 0;
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
    std::vector<char> idbuf(ggml_nbytes(ids));
    ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(ids), idbuf.data(), 0, idbuf.size());
    ggml_backend_synchronize(backend_router);

    std::vector<int32_t> mapped((size_t) n_ids_0 * n_ids_1);

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
                mapped[i1*n_ids_0 + i0] = *(const int32_t *)
                    (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
            }
        }
    } else {
        for (int64_t i1 = 0; i1 < n_ids_1; i1++) {
            for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                const int32_t e = *(const int32_t *)
                    (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
                GGML_ASSERT(e >= 0 && e < (int32_t) n_expert);
                int32_t slot = L.expert_slot[e];
                if (slot < 0) {
                    // LRU victim; the floor (n_slots >= min(E, top_k*bs)) keeps this
                    // pass's residents un-evictable (their stamps are the newest)
                    slot = 0;
                    for (uint32_t s = 1; s < n_slots; s++) {
                        if (L.slot_stamp[s] < L.slot_stamp[slot]) {
                            slot = (int32_t) s;
                        }
                    }
                    if (L.slot_expert[slot] >= 0) {
                        L.expert_slot[L.slot_expert[slot]] = -1;
                    }
                    L.slot_expert[slot] = e;
                    L.expert_slot[e]    = slot;
                    for (const auto & te : L.tensors) {
                        ggml_backend_tensor_set_async(split_backend, te.view_slots,
                            (const char *) te.host->data + (size_t) e * te.row_bytes,
                            (size_t) slot * te.row_bytes, te.row_bytes);
                    }
                    L.misses++;
                } else {
                    L.hits++;
                }
                L.slot_stamp[slot] = ++L.stamp;
                mapped[i1*n_ids_0 + i0] = slot;
            }
        }
    }

    ggml_backend_tensor_set_async(split_backend, L.ids_gpu,
        mapped.data(), 0, mapped.size() * sizeof(int32_t));

    L.serve_gen = generation;
    return true;
}

void llama_expert_pool::reset_slots() {
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
    uint64_t hits = 0, misses = 0;
    for (const auto & L : layers) {
        hits   += L.hits;
        misses += L.misses;
    }
    const double h = hits + misses > 0 ? (double) hits / (double) (hits + misses) : 0.0;
    LLAMA_LOG_INFO("%s: expert pool: %llu hits / %llu misses (h=%.3f)\n",
        __func__, (unsigned long long) hits, (unsigned long long) misses, h);
}
