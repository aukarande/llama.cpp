#include "llama-expert-pool.h"
#include "llama-pshard-plan.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>

llama_expert_pool::~llama_expert_pool() {
    if (admit_backend != nullptr) {
        ggml_backend_synchronize(admit_backend);
    }
    for (auto & L : layers) {
        if (L.warm_event != nullptr) {
            ggml_backend_event_free(L.warm_event);
            L.warm_event = nullptr;
        }
    }
    if (admit_event != nullptr) {
        ggml_backend_event_free(admit_event);
        admit_event = nullptr;
    }
    if (admit_backend != nullptr) {
        ggml_backend_free(admit_backend);
        admit_backend = nullptr;
    }
    if (ctx_views != nullptr) {
        ggml_free(ctx_views);
    }
}

bool llama_expert_pool::init(const llama_model & model, uint32_t n_expert_, uint32_t n_expert_used_) {
    if (const char * pk = getenv("PSHARD_POOL_PREDICT")) {
        predict_k = (int32_t) std::min<long>(8, std::max<long>(0, strtol(pk, nullptr, 10)));
    }
    if (const char * pf = getenv("PSHARD_POOL_PREFETCH")) {
        prefetch_on = strtol(pf, nullptr, 10) != 0;
    }
    if (const char * pn = getenv("PSHARD_POOL_PREFETCH_N")) {
        prefetch_n = (int32_t) std::max<long>(0, strtol(pn, nullptr, 10));
    }
    if (const char * w = getenv("PSHARD_POOL_WARM")) {
        warm_n = (int32_t) std::max<long>(0, strtol(w, nullptr, 10));
    }
    if (const char * a = getenv("PSHARD_POOL_ALLOC")) {
        alloc_on = strtol(a, nullptr, 10) != 0;
    }
    if (const char * aa = getenv("PSHARD_POOL_ADMIT_AFTER")) {
        const long v = strtol(aa, nullptr, 10);
        admit_after = (uint32_t) std::min<long>(64, std::max<long>(1, v));
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
    // A/B mode reuses the same span (2 whole layers) - take the max when asked
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

    // per-layer counts: the last warm start's plan when it was made for this uniform
    // count (same region), else uniform
    const bool use_plan = alloc_on && layer_slots_plan.size() == layers.size() && layer_slots_plan_base == slots_per_layer;
    size_t need = 0;
    for (auto & L : layers) {
        L.n_slots_l = use_plan ? layer_slots_plan[L.il] : slots_per_layer;
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

    // cache-mode layout: layer-major, per-tensor slot arrays; A/B halves overlay
    // the region start (they are only live on whole-stack prefill tiers, where the
    // cache contents are volatile by design)
    size_t off = 0;
    for (auto & L : layers) {
        L.slot_expert.assign(L.n_slots_l, -1);
        L.slot_stamp.assign(L.n_slots_l, 0);
        L.slot_pf_gen.assign(L.n_slots_l, 0);
        L.warm_pending = false;
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

    // A/B halves: parity-alternating whole-layer sets at the region start. Only
    // when the region holds the pair - a cache tier's region can be smaller, and
    // the whole-stack tiers get their own (larger) carve when they are applied
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

    uint32_t smin = UINT32_MAX, smax = 0;
    for (const auto & L : layers) {
        if (L.tensors.empty()) {
            continue;
        }
        smin = std::min(smin, L.n_slots_l);
        smax = std::max(smax, L.n_slots_l);
    }
    if (smin == UINT32_MAX) {
        smin = smax = n_slots;
    }
    LLAMA_LOG_INFO("%s: expert pool region %.1f MiB: s=%u slots/layer (cache%s)%s\n",
        __func__, bytes / (1024.0 * 1024.0), n_slots,
        use_plan ? (std::string(", per layer ") + std::to_string(smin) + ".." + std::to_string(smax)).c_str() : "",
        ab_capable ? " + 2-layer A/B pair" : " (no A/B pair: cache tiers only)");
    return true;
}

void llama_expert_pool::register_sched(ggml_backend_sched_t sched) {
    ggml_backend_sched_clear_input_copy_overrides(sched);
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
}

void llama_expert_pool::set_active(bool on, ggml_backend_sched_t sched) {
    if (on == active) {
        return;
    }
    active = on;
    epoch++;   // pooled-layer graph topology changes with this flag
    reset_slots();
    if (!on) {
        // the next graphs will not bind these; stale pointers would alias whatever
        // tensor the rebuilt graph places at the same address
        for (auto & L : layers) {
            L.ids_router = L.ids_gpu = L.ids_gpu_bias = L.ids_cpu = L.ids_pred = nullptr;
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
    // v1 relabel: cache contents do not survive the A/B overlay (the halves alias
    // the slot arrays); drop the maps and refill lazily
    reset_slots();
    if (ab) {
        // a new prefill: start its histogram
        prompt_seen = false;
        for (auto & L : layers) {
            if (!L.tensors.empty()) {
                L.prompt_count.assign(n_expert, 0);
            }
        }
    } else {
        // back to decode: the prompt just processed sizes and seeds the cache
        warm_start();
    }
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

void llama_expert_pool::warm_start() {
    if (!prompt_seen || region_base == nullptr || n_slots == 0 || backend_router == nullptr) {
        return;
    }
    // experts of each layer ranked by prompt count (descending), counts > 0 only
    std::vector<std::vector<int32_t>> order(layers.size());
    for (const auto & L : layers) {
        if (L.tensors.empty() || L.prompt_count.empty()) {
            continue;
        }
        auto & o = order[L.il];
        for (uint32_t e = 0; e < n_expert; e++) {
            if (L.prompt_count[e] > 0) {
                o.push_back((int32_t) e);
            }
        }
        std::sort(o.begin(), o.end(), [&](int32_t a, int32_t b) { return L.prompt_count[a] > L.prompt_count[b]; });
    }

    // (1) per-layer allocation: same region bytes, slots redistributed by demand.
    //     Every layer keeps the decode fetch floor (top_k distinct experts per
    //     pass); the remaining slots go, one at a time, to the layer whose next
    //     most-used prompt expert has the highest count (greedy water-filling on
    //     the histogram: the marginal slot buys the most expected hits there).
    if (alloc_on) {
        const uint32_t floor_l = std::min<uint32_t>(n_expert, n_expert_used);
        std::vector<uint32_t> alloc(layers.size(), 0);
        std::vector<size_t>   slot_bytes(layers.size(), 0);   // bytes of ONE slot in this layer
        size_t used = 0;
        bool ok = true;
        for (const auto & L : layers) {
            if (L.tensors.empty()) {
                continue;
            }
            for (const auto & e : L.tensors) {
                slot_bytes[L.il] += e.row_bytes;
            }
            slot_bytes[L.il] += 256 * L.tensors.size();   // per-tensor rounding slack
            alloc[L.il] = floor_l;
            used += (size_t) floor_l * slot_bytes[L.il];
        }
        if (used > region_bytes) {
            ok = false;   // the floors alone do not fit: keep the uniform layout
        }
        while (ok) {
            int32_t  best = -1;
            uint32_t best_gain = 0;
            for (const auto & L : layers) {
                if (L.tensors.empty() || alloc[L.il] >= n_expert) {
                    continue;
                }
                const auto & o = order[L.il];
                const uint32_t gain = alloc[L.il] < o.size() ? L.prompt_count[o[alloc[L.il]]] : 0;
                if (best < 0 || gain > best_gain) {
                    best = L.il;
                    best_gain = gain;
                }
            }
            if (best < 0 || used + slot_bytes[best] > region_bytes) {
                break;
            }
            alloc[best]++;
            used += slot_bytes[best];
        }
        if (ok) {
            layer_slots_plan      = alloc;
            layer_slots_plan_base = n_slots;
            if (!set_region(region_arena, region_base, region_bytes, n_slots)) {
                layer_slots_plan.clear();
                set_region(region_arena, region_base, region_bytes, n_slots);
            }
        }
    }

    // (2) seeding: each layer's most-used prompt experts go into its slots now, on
    //     the copy stream; the layer's first service waits for its own seeds. Popular
    //     first, stamped most-recent so the LRU keeps them longest.
    const int32_t wn = warm_n;
    if (wn <= 0) {
        return;
    }
    ensure_admit_backend(backend_router);
    if (admit_backend == nullptr) {
        return;
    }
    uint64_t seeded = 0;
    size_t   seeded_bytes = 0;
    for (auto & L : layers) {
        if (L.tensors.empty() || L.slot_expert.empty()) {
            continue;
        }
        const auto & o = order[L.il];
        const uint32_t m = std::min<uint32_t>({ (uint32_t) wn, L.n_slots_l, (uint32_t) o.size() });
        if (m == 0) {
            continue;
        }
        for (uint32_t i = 0; i < m; i++) {
            const int32_t e = o[i];
            L.slot_expert[i] = e;
            L.expert_slot[e] = (int32_t) i;
            L.slot_stamp[i]  = (uint64_t) (m - i);
            for (const auto & te : L.tensors) {
                ggml_backend_tensor_set_async(admit_backend, te.view_slots,
                    (const char *) te.host->data + (size_t) e * te.row_bytes,
                    (size_t) i * te.row_bytes, te.row_bytes);
                seeded_bytes += te.row_bytes;
            }
        }
        L.stamp = m;
        if (L.warm_event == nullptr) {
            L.warm_event = ggml_backend_event_new(ggml_backend_get_device(backend_router));
        }
        if (L.warm_event != nullptr) {
            ggml_backend_event_record(L.warm_event, admit_backend);
            L.warm_pending = true;
        } else {
            ggml_backend_synchronize(admit_backend);   // no events: land them now
        }
        seeded += m;
    }
    warm_seeded += seeded;
    warm_starts++;
    LLAMA_LOG_INFO("%s: expert pool warm start: seeded %llu experts (%.1f MiB) from the prompt histogram%s\n",
        __func__, (unsigned long long) seeded, seeded_bytes / (1024.0 * 1024.0),
        alloc_on && !layer_slots_plan.empty() ? ", slots redistributed per layer" : "");
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
                                       ggml_tensor * ids_gpu_bias, ggml_tensor * ids_cpu) {
    if (!layer_pooled(il)) {
        return;
    }
    layer_state & L = layers[il];
    L.ids_router   = ids_router;
    L.ids_gpu      = ids_gpu;
    L.ids_gpu_bias = ids_gpu_bias;
    L.ids_cpu      = ids_cpu;
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
    ids_read_buf.resize(ggml_nbytes(ids));
    std::vector<char> & idbuf = ids_read_buf;
    ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(ids), idbuf.data(), 0, idbuf.size());
    ggml_backend_synchronize(backend_router);

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
    std::vector<int32_t> & mapped = L.mapped_buf;

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
                const int32_t e = *(const int32_t *) (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
                mapped[i1*n_ids_0 + i0] = e;
                // prompt histogram for the warm start (the ids are read anyway)
                if (!L.prompt_count.empty() && e >= 0 && e < (int32_t) n_expert) {
                    L.prompt_count[e]++;
                    prompt_seen = true;
                }
            }
        }
        if (cpu_routes() && L.ids_cpu != nullptr) {
            // dual chain on a whole-stack tier: everything is resident, nothing goes to CPU
            L.bias_buf = mapped;
            L.cpu_buf.assign(mapped.size(), -1);
            if (L.ids_gpu_bias != nullptr) {
                ggml_backend_tensor_set_async(split_backend, L.ids_gpu_bias,
                    L.bias_buf.data(), 0, L.bias_buf.size() * sizeof(int32_t));
            }
            ggml_backend_tensor_set(L.ids_cpu, L.cpu_buf.data(), 0, L.cpu_buf.size() * sizeof(int32_t));
        }
    } else {
        // warm start: this layer's seeded slots were uploaded on the copy stream at the
        // A/B -> cache flip; the split stream waits for them before the first read
        if (L.warm_pending && L.warm_event != nullptr) {
            ggml_backend_event_wait(split_backend, L.warm_event);
            L.warm_pending = false;
        }

        // 1. the pass's distinct experts: hits stay; misses are decided PER EXPERT
        //    by the tier's miss policy (all of an expert's routes go the same way)
        seen_gen.assign(n_expert, 0);
        std::vector<int32_t> miss_list;
        uint32_t n_hit = 0;
        for (int64_t i1 = 0; i1 < n_ids_1; i1++) {
            for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                const int32_t e = *(const int32_t *)
                    (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
                GGML_ASSERT(e >= 0 && e < (int32_t) n_expert);
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
            pred_read_buf.resize(ggml_nbytes(pt));
            ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(pt), pred_read_buf.data(), 0, pred_read_buf.size());
            ggml_backend_synchronize(backend_router);
            const int64_t np1 = std::min<int64_t>(pt->ne[1], n_ids_1);
            for (int64_t i1 = 0; i1 < np1; i1++) {
                for (int64_t i0 = 0; i0 < n_ids_0; i0++) {
                    const int32_t e = *(const int32_t *) (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
                    bool in_pred = false;
                    for (int64_t j0 = 0; j0 < pt->ne[0] && !in_pred; j0++) {
                        in_pred = *(const int32_t *) (pred_read_buf.data() + i1*pt->nb[1] + j0*pt->nb[0]) == e;
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
        std::vector<int32_t> probation;
        for (size_t i = 0; i < miss_list.size(); i++) {
            const int32_t e = miss_list[i];
            L.expert_last_gen[e] = generation;
            const uint32_t seen_misses = ++L.miss_count[e];
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
            if (seen_misses < admit_after) {
                probation.push_back(slot);
            }
            for (const auto & te : L.tensors) {
                ggml_backend_tensor_set_async(up_backend, te.view_slots,
                    (const char *) te.host->data + (size_t) e * te.row_bytes,
                    (size_t) slot * te.row_bytes, te.row_bytes);
            }
            if (background) {
                L.expert_pending[e] = generation;   // CPU route this pass, GPU hit from the next
                uploaded_bg = uploaded_bg || up_backend == admit_backend;
            }
            L.misses++;
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
                    (idbuf.data() + i1*ids->nb[1] + i0*ids->nb[0]);
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
        // probationary admission: experts below admit_after misses served this pass
        // but must not displace the established residents next time - restamp them
        // below every live entry (in fetch order). Placed AFTER the per-route touch
        // above, which refreshes every current slot to MRU (it undid the demotion when
        // this sat before it: h was invariant to admit_after across 44k evictions)
        if (!probation.empty()) {
            uint64_t floor = UINT64_MAX;
            for (uint32_t s = 0; s < L.n_slots_l; s++) {
                bool on_probation = false;
                for (int32_t ps : probation) {
                    if (ps == (int32_t) s) { on_probation = true; break; }
                }
                if (!on_probation) {
                    floor = std::min(floor, L.slot_stamp[s]);
                }
            }
            const uint64_t n    = probation.size();
            const uint64_t base = (floor != UINT64_MAX && floor > n) ? floor - n : 0;
            for (uint64_t i = 0; i < n; i++) {
                L.slot_stamp[probation[i]] = base + i;
            }
            L.demoted += n;
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
        }

        // 5. prefetch for layer il+k. The prediction for that layer was computed in
        //    THIS layer's compute split (ready: the ids read above drained the stream).
        //    Upload its non-resident experts into that layer's LRU victims on the copy
        //    stream now, about one layer ahead of their use; at il+k's service they are
        //    hits once the split stream has waited on the admit event (top of serve()).
        //    The victim is safe: the target layer's previous pass has completed (the
        //    drain above) and nothing reads its slots before its own service.
        if (predict_k > 0 && prefetch_on && n_ids_1 == 1) {
            const int32_t tgt = L.il + predict_k;
            if (layer_pooled(tgt) && layers[tgt].ids_pred != nullptr) {
                layer_state & T = layers[tgt];
                if (T.slot_expert.size() == T.n_slots_l && T.n_slots_l > 0 && !T.expert_slot.empty()) {
                    const ggml_tensor * pt = T.ids_pred;
                    pred_read_buf.resize(ggml_nbytes(pt));
                    ggml_backend_tensor_get_async(backend_router, const_cast<ggml_tensor *>(pt), pred_read_buf.data(), 0, pred_read_buf.size());
                    ggml_backend_synchronize(backend_router);
                    ensure_admit_backend(split_backend);
                    if (admit_backend != nullptr && admit_event != nullptr) {
                        bool issued = false;
                        int32_t n_issued = 0;
                        // the prediction is in descending score order: with a cap, the
                        // most confident experts go first (mispredicted uploads compete
                        // with the critical-path misses for the same PCIe link)
                        for (int64_t j0 = 0; j0 < pt->ne[0]; j0++) {
                            if (prefetch_n > 0 && n_issued >= prefetch_n) {
                                break;
                            }
                            const int32_t e = *(const int32_t *) (pred_read_buf.data() + j0*pt->nb[0]);
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
    uint64_t hits = 0, misses = 0, passes = 0;
    uint64_t hits_1 = 0, misses_1 = 0, passes_1 = 0, demoted = 0, evicted = 0;
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
        demoted  += L.demoted;
        evicted  += L.evicted;
        const uint64_t n = L.hits + L.misses;
        char buf[16];
        snprintf(buf, sizeof(buf), " %.2f", n > 0 ? (double) L.hits / (double) n : 0.0);
        per_layer += buf;
    }
    if (hits + misses == 0) {
        return; // never served in cache mode (or pool never engaged)
    }
    const double h   = (double) hits / (double) (hits + misses);
    const double mpt = passes > 0 ? (double) misses / (double) passes : 0.0;
    // cache-mode decode counters: distinct experts per layer per pass; misses/token
    // is per pass (= per token at bs=1), summed over the pooled layers
    LLAMA_LOG_INFO("%s: expert pool: %llu hits / %llu misses over %llu passes: h=%.3f misses/token=%.1f (s=%u)\n",
        __func__, (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) passes, h, mpt, n_slots);
    LLAMA_LOG_INFO("%s: expert pool h per layer:%s\n", __func__, per_layer.c_str());
    LLAMA_LOG_INFO("%s: expert pool admission: admit_after=%u, %llu fetches demoted, %llu residents evicted\n",
        __func__, admit_after, (unsigned long long) demoted, (unsigned long long) evicted);
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
        LLAMA_LOG_INFO("%s: expert pool warm start: %u flips, %llu experts seeded\n",
            __func__, warm_starts, (unsigned long long) warm_seeded);
        LLAMA_LOG_INFO("%s: expert pool prefetch: %s, %llu uploads issued, %llu used by the target layer (%.3f)\n",
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
        LLAMA_LOG_INFO("%s: expert pool decode: %llu hits / %llu misses over %llu passes: h=%.3f misses/token=%.1f (s=%u)\n",
            __func__, (unsigned long long) hits_1, (unsigned long long) misses_1, (unsigned long long) passes_1, h1, mpt1, n_slots);
    }
}
