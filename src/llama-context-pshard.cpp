#include "llama-context.h"

#include "llama-impl.h"

#include <stdexcept>
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-pipe-shard.h"
#include "llama-model.h"
#include "llama-pshard-plan.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

namespace {
    struct llama_pshard_split_cb_context {
        std::vector<llama_memory_pipe_shard_i *> pipe_shards;
    };

    void pshard_pre_compute(ggml_tensor * tensor, ggml_backend_t backend, void * user_data) {
        auto * ctx = (llama_pshard_split_cb_context *) user_data;
        for (auto * ps : ctx->pipe_shards) {
            if (ps->upload_if_owned(tensor, backend)) {
                return;
            }
        }
    }

    void pshard_prefetch(ggml_tensor * tensor, ggml_backend_t copy_backend, void * user_data) {
        auto * ctx = (llama_pshard_split_cb_context *) user_data;
        for (auto * ps : ctx->pipe_shards) {
            if (ps->prefetch_if_owned(tensor, copy_backend)) {
                return;
            }
        }
    }

    void pshard_post_compute(ggml_tensor * tensor, ggml_backend_t backend, void * user_data) {
        auto * ctx = (llama_pshard_split_cb_context *) user_data;
        for (auto * ps : ctx->pipe_shards) {
            if (ps->download_if_owned(tensor, backend)) {
                return;
            }
        }
    }

    thread_local llama_pshard_split_cb_context g_split_ctx;
    thread_local std::vector<std::vector<uint32_t>> g_kv_write_cells;
    thread_local std::vector<std::vector<uint32_t>> g_swa_write_cells;
    thread_local std::vector<std::vector<uint32_t>> g_lid_write_cells;

    size_t total_pinned_cache_size(llama_memory_i * mem) {
        size_t total = 0;
        for (auto * ps : mem->get_pipe_shards()) {
            total += ps->current_pinned_size();
        }
        return total;
    }

    void zero_pinned_layers(const std::vector<llama_memory_pipe_shard_i::layer> & layers) {
        for (const auto & l : layers) {
            if (!l.is_pinned) continue;
            size_t t1_bytes = ggml_nbytes(l.t1_gpu);
            size_t t2_bytes = l.t2_gpu ? ggml_nbytes(l.t2_gpu) : 0;
            std::vector<uint8_t> zeros(std::max(t1_bytes, t2_bytes), 0);
            ggml_backend_tensor_set(l.t1_gpu, zeros.data(), 0, t1_bytes);
            if (l.t2_gpu) {
                ggml_backend_tensor_set(l.t2_gpu, zeros.data(), 0, t2_bytes);
            }
        }
    }

    void sync_pins(llama_memory_pipe_shard_i * ps,
                   const std::function<int32_t(uint32_t)> & bid_for,
                   const pshard_dev_layout & layout,
                   ggml_backend_t gpu) {
        for (const auto & l : ps->get_layers()) {
            const int32_t bid          = bid_for(l.il);
            const bool    want_pinned  = (bid == layout.compute);
            const bool    want_cpu     = (bid == layout.cpu);

            if (want_pinned && !l.is_pinned) {
                ps->pin_layer(l.il);
                ggml_backend_tensor_memset_async(gpu, l.t1_gpu, 0, 0, ggml_nbytes(l.t1_gpu));
                if (l.t2_gpu) {
                    ggml_backend_tensor_memset_async(gpu, l.t2_gpu, 0, 0, ggml_nbytes(l.t2_gpu));
                }
            } else if (!want_pinned && l.is_pinned) {
                ps->unpin_layer(l.il);
            }

            if (bid >= 0) {
                if (want_cpu) {
                    ps->activate_cpu(l.il);
                } else {
                    ps->activate_gpu(l.il);
                }
            }
        }
    }

} // namespace

void pshard_assign_tensors(
        ggml_backend_sched_t                              sched,
        const llama_model                               & model,
        llama_memory_i                                  * memory,
        const std::vector<ggml_backend_ptr>             & backends,
        const pshard_dev_layout                         & layout) {
    const auto & tbids = model.get_tensor_backend_ids();
    const auto & lbids = model.get_layer_backend_ids();

    for (const auto & [tensor, bid] : tbids) {
        if (bid >= 0 && bid < (int32_t) backends.size()) {
            ggml_backend_sched_set_tensor_backend_hint(sched, tensor, backends[bid].get());
        }
    }

    if (memory) {
        for (auto * ps : memory->get_pipe_shards()) {
            ps->assign_tensors(sched, lbids, backends, layout);
        }
    }
}

void pshard_refresh_stream_views(llama_memory_i * memory) {
    if (!memory) return;
    for (auto * ps : memory->get_pipe_shards()) {
        for (const auto & l : ps->get_layers()) {
            ps->refresh_stream_views(l.il);
        }
    }
}

void llama_context::pshard_pack_cache_region() {
    auto * buf = model.get_dev_preload_buf();
    if (!buf) return;

    auto pipe_shards = memory->get_pipe_shards();
    if (pipe_shards.empty()) return;

    size_t buf_total = ggml_backend_buffer_get_size(buf);
    size_t alignment = ggml_backend_buffer_get_alignment(buf);
    void * buf_base  = ggml_backend_buffer_get_base(buf);

    const auto & lbids = model.get_layer_backend_ids();
    const auto & layout = pshard_layout;

    auto is_pinned = [&](uint32_t il) -> bool {
        auto it = lbids.find(il);
        return it != lbids.end() && it->second == layout.compute;
    };

    struct cache_entry {
        uint32_t              il;
        llama_memory_pipe_shard_i  * ps;
        ggml_tensor         * t1;
        ggml_tensor         * t2;
    };

    std::vector<cache_entry> entries;

    for (auto * ps : pipe_shards) {
        for (const auto & l : ps->get_layers()) {
            entries.push_back({ l.il, ps, l.t1_gpu, l.t2_gpu });
        }
    }

    std::sort(entries.begin(), entries.end(), [](const cache_entry & a, const cache_entry & b) { return a.il < b.il; });

    size_t total_cache = 0;

    size_t offset_from_right = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        auto & e = entries[i];

        size_t t1_size = ((ggml_backend_buffer_get_alloc_size(buf, e.t1) + alignment - 1) / alignment) * alignment;
        size_t t2_size = e.t2 ? ((ggml_backend_buffer_get_alloc_size(buf, e.t2) + alignment - 1) / alignment) * alignment : 0;
        size_t layer_total = t1_size + t2_size;

        void * t2_addr = nullptr;
        if (t2_size > 0) {
            offset_from_right += t2_size;
            t2_addr = (char *)buf_base + buf_total - offset_from_right;
        }

        offset_from_right += t1_size;
        void * t1_addr = (char *)buf_base + buf_total - offset_from_right;

        e.ps->set_external_addrs(e.il, t1_addr, t2_addr, layer_total);
        total_cache += layer_total;
    }

    size_t n_pinned_total = 0;
    for (auto & e : entries) {
        if (!is_pinned(e.il)) continue;
        e.ps->pin_layer(e.il);
        n_pinned_total++;
    }

    for (auto * ps : pipe_shards) {
        zero_pinned_layers(ps->get_layers());
    }

    size_t preloaded_size = model.get_dev_preloaded_size();
    size_t pinned_cache = total_pinned_cache_size(memory.get());

    LLAMA_LOG_INFO("%s: %zu cache layers (%.2f MiB total), %zu pinned (%.2f MiB)\n",
        __func__, entries.size(), total_cache / (1024.0 * 1024.0),
        n_pinned_total, pinned_cache / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("%s: layout: [weights 0..%.2f | scratch %.2f..%.2f | cache %.2f..%.2f MiB]\n",
        __func__,
        preloaded_size / (1024.0 * 1024.0),
        preloaded_size / (1024.0 * 1024.0),
        (buf_total - pinned_cache) / (1024.0 * 1024.0),
        (buf_total - pinned_cache) / (1024.0 * 1024.0),
        buf_total / (1024.0 * 1024.0));
}

#include "llama-expert-pool.h"

// EXPERT_POOL: create the pool and carve its region from the arena, directly
// below the pinned KV cache: [weights | scratch | POOL | pinned KV]. The slot
// count comes from the plan; when the planner's file-size heuristic oversized
// it, clamp to what actually fits after the largest measured tier scratch.
void llama_context::pshard_setup_expert_pool() {
    expert_pool.reset();
    expert_pool_bytes = 0;
    if (!cparams.pshard || getenv("PSHARD_POOL_RUNTIME") == nullptr) {
        return;
    }
    auto * registry = const_cast<llama_pshard_plan_registry *>(model.get_plan_registry());
    if (registry == nullptr) {
        return;
    }
    uint32_t slots = 0;
    size_t scratch_need = 64ull << 20;
    bool any = false;
    for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
        const llama_pshard_plan * p = registry->get_best(t);
        if (p == nullptr || !p->is_viable) {
            continue;
        }
        scratch_need = std::max(scratch_need, p->scratch_measured + (64ull << 20));
        if (p->strategy == LLAMA_PSHARD_EXPERT_POOL) {
            any   = true;
            slots = slots > 0 ? std::min(slots, p->pool_slots) : p->pool_slots;
        }
    }
    if (!any || slots == 0) {
        return;
    }

    auto pool = std::make_unique<llama_expert_pool>();
    if (!pool->init(model, model.hparams.n_expert, model.hparams.n_expert_used)) {
        return;
    }

    ggml_backend_buffer_t buf = model.get_dev_preload_buf();
    if (buf == nullptr) {
        return;
    }
    const size_t buf_total = ggml_backend_buffer_get_size(buf);
    const size_t preloaded = model.get_dev_preloaded_size();
    const size_t cache     = total_pinned_cache_size(memory.get());
    if (preloaded + cache + scratch_need >= buf_total) {
        LLAMA_LOG_WARN("%s: no room for the expert pool (weights %.1f + cache %.1f + scratch %.1f >= %.1f MiB)\n",
            __func__, preloaded / (1024.0 * 1024.0), cache / (1024.0 * 1024.0),
            scratch_need / (1024.0 * 1024.0), buf_total / (1024.0 * 1024.0));
        return;
    }
    const size_t avail = buf_total - preloaded - cache - scratch_need;
    size_t want = pool->region_bytes_needed(slots);
    while (slots > 0 && want > avail) {
        slots--;
        want = pool->region_bytes_needed(slots);
    }
    if (slots == 0 || want == 0 || 2 * pool->layer_full_bytes > avail) {
        LLAMA_LOG_WARN("%s: expert pool does not fit: avail %.1f MiB, A/B pair needs %.1f MiB\n",
            __func__, avail / (1024.0 * 1024.0), 2 * pool->layer_full_bytes / (1024.0 * 1024.0));
        return;
    }

    // the clamp must not undercut any cache-mode POOL tier's fetch floor (one
    // MUL_MAT_ID needs the whole pass's distinct experts resident at once)
    uint32_t floor_slots = 0;
    for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
        const llama_pshard_plan * p = registry->get_best(t);
        if (p == nullptr || !p->is_viable || p->strategy != LLAMA_PSHARD_EXPERT_POOL) {
            continue;
        }
        const uint64_t bs = registry->tier_sizes[t];
        if (bs * model.hparams.n_expert_used * 2 >= model.hparams.n_expert) {
            continue; // whole-stack tier: A/B mode, no slot floor
        }
        const uint32_t f = (uint32_t) std::min<uint64_t>(model.hparams.n_expert, bs * model.hparams.n_expert_used);
        floor_slots = std::max(floor_slots, f);
    }
    if (slots < floor_slots) {
        LLAMA_LOG_WARN("%s: expert pool clamped to %u slots/layer below the cache-tier floor %u - pool disabled\n",
            __func__, slots, floor_slots);
        return;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf) + buf_total - cache - want;
    base = (char *) ((uintptr_t) base & ~(uintptr_t) 255);
    want = (size_t) ((char *) ggml_backend_buffer_get_base(buf) + buf_total - cache - base);

    if (!pool->set_region(buf, base, want, slots)) {
        return;
    }
    pool->backend_router = backends[pshard_layout.compute].get();
    expert_pool_bytes = want;
    expert_pool = std::move(pool);
}

void llama_context::pshard_update_pool_mode(const llama_pshard_plan & plan) {
    if (!expert_pool) {
        return;
    }
    // legacy tiers in a mixed registry stream normally: the pool disengages
    expert_pool->set_active(plan.strategy == LLAMA_PSHARD_EXPERT_POOL, sched.get());
    const bool ab = (uint64_t) plan.batch_size * expert_pool->n_expert_used * 2 >= expert_pool->n_expert;
    expert_pool->set_ab_mode(ab, sched.get());
}

// the pool's remapped-ids leaves anchor on the same virtual backend as the
// layer's expert tensors, so the mm split consumes them without a copy
void llama_context::pshard_assign_pool_tensors() {
    if (!expert_pool) {
        return;
    }
    for (auto & L : expert_pool->layers) {
        if (L.tensors.empty() || L.ids_gpu == nullptr) {
            continue;
        }
        const int32_t bid = pshard_layout.shard(L.il);
        if (bid >= 0 && bid < (int32_t) backends.size()) {
            ggml_backend_sched_set_tensor_backend_hint(sched.get(), L.ids_gpu, backends[bid].get());
        }
    }
}

void llama_context::pshard_setup_sched() {
    ggml_backend_sched_set_prefetch_weights(sched.get(), cparams.pshard_overlap);

    g_split_ctx = {};
    g_split_ctx.pipe_shards = memory->get_pipe_shards();

    if (!g_split_ctx.pipe_shards.empty()) {
        ggml_backend_sched_set_split_callbacks(sched.get(), pshard_pre_compute, pshard_post_compute, &g_split_ctx);
        ggml_backend_sched_set_prefetch_cb(sched.get(), pshard_prefetch);
    }

    if (expert_pool) {
        expert_pool->register_sched(sched.get());
    }

    if (model.get_dev_preload_buf()) {
        size_t preloaded_size = model.get_dev_preloaded_size();
        size_t buf_total = ggml_backend_buffer_get_size(model.get_dev_preload_buf());

        size_t pinned_cache_size = total_pinned_cache_size(memory.get()) + expert_pool_bytes;

        if (preloaded_size + pinned_cache_size > buf_total) {
            LLAMA_LOG_ERROR("%s: weights = %.2f MiB, pinned cache = %.2f MiB, buffer = %.2f MiB, overshoot = %.2f MiB\n",
                __func__,
                preloaded_size    / (1024.0 * 1024.0),
                pinned_cache_size / (1024.0 * 1024.0),
                buf_total         / (1024.0 * 1024.0),
                (preloaded_size + pinned_cache_size - buf_total) / (1024.0 * 1024.0));
        }
        if (preloaded_size + pinned_cache_size > buf_total) {
            // the canonical layout is the UNION across all tier plans plus the pinned cache;
            // per-plan probes (even with the packing margin) cannot see this union, so at
            // tight budgets it can overshoot. Degrade loudly instead of crashing: pshard is
            // disabled for this run and the caller falls back to the stock path.
            LLAMA_LOG_WARN("%s: >>> canonical layout (%.2f MiB) + pinned cache (%.2f MiB) exceed the %.2f MiB budget; pshard DISABLED for this run <<<\n",
                __func__, preloaded_size / (1024.0 * 1024.0), pinned_cache_size / (1024.0 * 1024.0),
                buf_total / (1024.0 * 1024.0));
            throw std::runtime_error("pshard: canonical layout overshoots the VRAM budget");
        }

        size_t scratch_size = buf_total - preloaded_size - pinned_cache_size;

        ggml_backend_sched_set_buffer(sched.get(), backends[pshard_layout.compute].get(),
            model.get_dev_preload_buf(), preloaded_size, scratch_size);

        LLAMA_LOG_DEBUG("%s: single buffer: weights %.2f MiB, cache %.2f MiB, scratch %.2f MiB (total %.2f MiB)\n",
            __func__, preloaded_size / (1024.0 * 1024.0),
            pinned_cache_size / (1024.0 * 1024.0),
            scratch_size / (1024.0 * 1024.0), buf_total / (1024.0 * 1024.0));
    }
}

void llama_context::pshard_apply_plan(const llama_pshard_plan & plan, bool with_upload, bool force_upload) {
    // pool engage/mode must track EVERY applied plan (warmup reserves each tier
    // before the initial plan lands; mixed registries flip per tier)
    pshard_update_pool_mode(plan);
    // per-tier transport mode: governs split_graph keepalives and the runtime prefetch scan
    ggml_backend_sched_set_prefetch_weights(sched.get(), cparams.pshard_overlap && plan.overlap);
    ggml_backend_t gpu = backends[pshard_layout.compute].get();
    size_t scratch_off = const_cast<llama_model &>(model).pshard_apply_plan(plan, with_upload ? gpu : nullptr, force_upload);

    const auto & lbids = model.get_layer_backend_ids();
    const auto & layout = pshard_layout;

    auto bid_for = [&](uint32_t il) -> int32_t {
        auto it = lbids.find(il);
        return (it != lbids.end()) ? it->second : -1;
    };

    for (auto * ps : memory->get_pipe_shards()) {
        sync_pins(ps, bid_for, layout, gpu);
    }

    if (model.get_dev_preload_buf()) {
        size_t buf_total = ggml_backend_buffer_get_size(model.get_dev_preload_buf());
        size_t pinned_cache_size = total_pinned_cache_size(memory.get()) + expert_pool_bytes;

        if (scratch_off + pinned_cache_size > buf_total) {
            LLAMA_LOG_ERROR("%s: scratch_off = %.2f MiB, pinned cache = %.2f MiB, buffer = %.2f MiB, overshoot = %.2f MiB\n",
                __func__,
                scratch_off       / (1024.0 * 1024.0),
                pinned_cache_size / (1024.0 * 1024.0),
                buf_total         / (1024.0 * 1024.0),
                (scratch_off + pinned_cache_size - buf_total) / (1024.0 * 1024.0));
        }
        GGML_ASSERT(scratch_off + pinned_cache_size <= buf_total &&
            "pshard: weights + pinned cache exceed VRAM buffer -- plan overshoots budget");

        size_t scratch_size = buf_total - scratch_off - pinned_cache_size;

        ggml_backend_sched_set_alloc_range(sched.get(), backends[pshard_layout.compute].get(),
            scratch_off, scratch_size);

        LLAMA_LOG_DEBUG("%s: scratch_off = %.2f MiB, cache = %.2f MiB, scratch = %.2f MiB\n",
            __func__, scratch_off / (1024.0 * 1024.0),
            pinned_cache_size / (1024.0 * 1024.0), scratch_size / (1024.0 * 1024.0));
    }

    if (plan.alloc_state.valid) {
        auto * galloc = ggml_backend_sched_get_galloc(sched.get());
        ggml_gallocr_restore_state(galloc,
            plan.alloc_state.node_allocs.data(), plan.alloc_state.node_allocs.size(),
            plan.alloc_state.leaf_allocs.data(), plan.alloc_state.leaf_allocs.size(),
            plan.alloc_state.n_nodes, plan.alloc_state.n_leafs);
        ggml_backend_sched_restore_backend_ids(sched.get(),
            plan.alloc_state.node_backend_ids.data(), (int)plan.alloc_state.node_backend_ids.size(),
            plan.alloc_state.leaf_backend_ids.data(), (int)plan.alloc_state.leaf_backend_ids.size());
    } else {
        pshard_reserve_and_save(plan);
    }
}

void llama_context::pshard_reserve_and_save(const llama_pshard_plan & plan) {
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory context\n", __func__);
            plan.alloc_state.valid = false;
            return;
        }
    }

    const uint32_t n_seqs   = cparams.n_seq_max;
    const uint32_t n_tokens = plan.batch_size;
    const uint32_t n_outputs = n_tokens;

    // start with unconstrained scratch packing
    ggml_backend_t gpu = backends[pshard_layout.compute].get();
    const bool external_buf = model.get_dev_preload_buf() != nullptr;
    size_t scratch_off   = 0;
    size_t scratch_avail = 0;

    if (external_buf) {
        const size_t buf_total         = ggml_backend_buffer_get_size(model.get_dev_preload_buf());
        const size_t pinned_cache_size = total_pinned_cache_size(memory.get()) + expert_pool_bytes;
        scratch_off   = plan.cached_scratch_off;
        scratch_avail = buf_total - scratch_off - pinned_cache_size;
        ggml_backend_sched_set_alloc_range(sched.get(), gpu, scratch_off, SIZE_MAX/2);
    }

    auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());

    if (gf && external_buf) {
        const int    n_chunks    = ggml_backend_sched_get_n_chunks(sched.get(), gpu);
        const size_t chunk0_max  = (n_chunks >= 1) ? ggml_backend_sched_get_chunk_max_size(sched.get(), gpu, 0) : 0;
        const size_t chunk0_used = (chunk0_max > scratch_off) ? chunk0_max - scratch_off : 0;

        if (chunk0_used <= scratch_avail) {
            ggml_backend_sched_set_alloc_range(sched.get(), gpu, scratch_off, chunk0_used);
            pshard_save_alloc_state(plan);
            return;
        }

        LLAMA_LOG_WARN("%s: unconstrained packing %.2f MiB > scratch budget %.2f MiB; retrying constrained\n",
            __func__, chunk0_used / (1024.0 * 1024.0), scratch_avail / (1024.0 * 1024.0));

        ggml_backend_sched_set_alloc_range(sched.get(), gpu, scratch_off, scratch_avail);
        gf = graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
    }

    if (!gf) {
        LLAMA_LOG_ERROR("%s: graph_reserve failed for plan %s n_pinned=%u; alloc state not saved\n",
            __func__, llama_pshard_strategy_name(plan.strategy), plan.n_pinned);
        plan.alloc_state.valid = false;
        return;
    }

    pshard_save_alloc_state(plan);
}

void llama_context::pshard_save_alloc_state(const llama_pshard_plan & plan) {
    auto * galloc = ggml_backend_sched_get_galloc(sched.get());

    size_t node_size = 0, leaf_size = 0;
    ggml_gallocr_get_state_sizes(galloc, &node_size, &leaf_size);

    plan.alloc_state.node_allocs.resize(node_size);
    plan.alloc_state.leaf_allocs.resize(leaf_size);
    ggml_gallocr_save_state(galloc,
        plan.alloc_state.node_allocs.data(),
        plan.alloc_state.leaf_allocs.data(),
        &plan.alloc_state.n_nodes,
        &plan.alloc_state.n_leafs);

    int sched_n_nodes = 0, sched_n_leafs = 0;
    ggml_backend_sched_save_backend_ids(sched.get(), nullptr, nullptr, &sched_n_nodes, &sched_n_leafs);

    plan.alloc_state.node_backend_ids.resize(sched_n_nodes);
    plan.alloc_state.leaf_backend_ids.resize(sched_n_leafs);
    ggml_backend_sched_save_backend_ids(sched.get(),
        plan.alloc_state.node_backend_ids.data(),
        plan.alloc_state.leaf_backend_ids.data(),
        nullptr, nullptr);

    plan.alloc_state.valid = (plan.alloc_state.n_nodes > 0);

    LLAMA_LOG_DEBUG("%s: saved alloc state: nodes=%d (%.1f KiB), leafs=%d (%.1f KiB), bids=%d/%d\n",
        __func__, plan.alloc_state.n_nodes, node_size / 1024.0,
        plan.alloc_state.n_leafs, leaf_size / 1024.0,
        sched_n_nodes, sched_n_leafs);
}

void llama_context::pshard_warmup_plan_reserves() {
    auto * registry = model.get_plan_registry();
    if (!registry) return;

    LLAMA_LOG_INFO("%s: pre-computing scratch offsets for %zu tier plans ...\n",
        __func__, registry->tier_sizes.size());
    const int64_t t0 = llama_time_us();

    for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
        auto & plan = registry->best_plans[t];
        if (!plan.is_viable) continue;

        // the reserve must see the same prefetch/keepalive mode this tier will run with
        ggml_backend_sched_set_prefetch_weights(sched.get(), cparams.pshard_overlap && plan.overlap);

        const_cast<llama_model &>(model).pshard_compute_scratch_off(plan); // see pshard_apply_plan

        LLAMA_LOG_DEBUG("%s: tier %zu (bs=%u, %s, n_pinned=%u) scratch_off=%.2f MiB\n",
            __func__, t, registry->tier_sizes[t],
            llama_pshard_strategy_name(plan.strategy), plan.n_pinned,
            plan.cached_scratch_off / (1024.0 * 1024.0));

        // the planner's no_alloc probe can undershoot the runtime canonical packing by
        // alignment/padding; a tier whose packed weights + pinned cache exceed the managed
        // buffer must degrade, not assert (seen: 24 MiB overshoot at a 2000 MiB budget)
        if (model.get_dev_preload_buf()) {
            const size_t buf_total = ggml_backend_buffer_get_size(model.get_dev_preload_buf());
            const size_t pc        = total_pinned_cache_size(memory.get()) + expert_pool_bytes;
            if (plan.cached_scratch_off + pc > buf_total) {
                LLAMA_LOG_WARN("%s: tier %zu (bs=%u, %s, n_pinned=%u) packing overshoots buffer by %.2f MiB; marking unviable\n",
                    __func__, t, registry->tier_sizes[t],
                    llama_pshard_strategy_name(plan.strategy), plan.n_pinned,
                    (plan.cached_scratch_off + pc - buf_total) / (1024.0 * 1024.0));
                plan.is_viable = false;
                continue;
            }
        }

        pshard_apply_plan(plan, /*with_upload=*/false);

        if (!plan.alloc_state.valid) {
            LLAMA_LOG_WARN("%s: tier %zu (bs=%u, %s, n_pinned=%u) reserve failed; marking unviable\n",
                __func__, t, registry->tier_sizes[t],
                llama_pshard_strategy_name(plan.strategy), plan.n_pinned);
            plan.is_viable = false;
        }
    }

    LLAMA_LOG_INFO("%s: tier summary:\n", __func__);
    for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
        auto & plan = registry->best_plans[t];
        if (!plan.is_viable) {
            LLAMA_LOG_INFO("%s:   tier %zu bs=%-5u — no viable plan\n", __func__, t, registry->tier_sizes[t]);
            continue;
        }

        char attn_buf[32] = "";
        if (plan.n_attn_pinned > 0) {
            snprintf(attn_buf, sizeof(attn_buf), " (attn=%u)", plan.n_attn_pinned);
        }

        if (plan.tps > 0.0f) {
            LLAMA_LOG_INFO("%s:   tier %zu bs=%-5u %s n_pinned=%u%s tps=%.1f\n",
                __func__, t, registry->tier_sizes[t],
                llama_pshard_strategy_name(plan.strategy), plan.n_pinned, attn_buf, plan.tps);
        } else {
            LLAMA_LOG_INFO("%s:   tier %zu bs=%-5u %s n_pinned=%u%s\n",
                __func__, t, registry->tier_sizes[t],
                llama_pshard_strategy_name(plan.strategy), plan.n_pinned, attn_buf);
        }
    }

    const int64_t t1 = llama_time_us();
    LLAMA_LOG_INFO("%s: pre-computed %zu tiers in %.1f ms\n",
        __func__, registry->tier_sizes.size(), (t1 - t0) / 1000.0);
}

void llama_context::pshard_apply_initial_plan() {
    auto * registry = model.get_plan_registry();
    if (!registry) return;

    size_t initial_tier = registry->tier_index(16);
    llama_pshard_plan * initial = registry->get_best(initial_tier);

    if (!initial) {
        initial = registry->get_best(0);
    }

    if (initial) {
        // force_upload: warmup applies moved tensor addresses without uploading, so an
        // extra landing at the same offset as the last warmup tier would otherwise be skipped
        pshard_apply_plan(*initial, /*with_upload=*/true, /*force_upload=*/true);
        pshard_active_plan = initial;
        pshard_update_pool_mode(*initial);
    }
}

void llama_context::pshard_switch_plan(
        const llama_pshard_plan & old_plan,
        const llama_pshard_plan & new_plan,
        size_t                    old_tier,
        size_t                    new_tier,
        uint32_t                  n_tokens) {
    ggml_backend_t gpu = backends[pshard_layout.compute].get();

    auto pipe_shards = memory->get_pipe_shards();

    // save old pin state
    std::vector<std::unordered_map<int32_t, bool>> old_pins(pipe_shards.size());
    for (size_t i = 0; i < pipe_shards.size(); i++) {
        for (const auto & l : pipe_shards[i]->get_layers()) {
            old_pins[i][l.il] = l.is_pinned;
        }
    }

    const_cast<llama_model &>(model).pshard_set_backend_maps(new_plan); // see pshard_apply_plan
    const auto & new_lbids = model.get_layer_backend_ids();
    auto will_be_pinned = [&](uint32_t il) -> bool {
        auto it = new_lbids.find(il);
        return it != new_lbids.end() && it->second == pshard_layout.compute;
    };

    // download layers moving off gpu
    int n_down = 0, n_skip_down = 0;
    for (auto * ps : pipe_shards) {
        for (const auto & l : ps->get_layers()) {
            if (!l.is_pinned) continue;
            if (!will_be_pinned(l.il)) {
                ps->download_for_switch(l.il, gpu);
                n_down++;
            } else {
                n_skip_down++;
            }
        }
    }

    // apply new plan
    pshard_apply_plan(new_plan);

    // upload newly pinned layers
    int n_up = 0, n_skip_up = 0;
    for (size_t i = 0; i < pipe_shards.size(); i++) {
        for (const auto & l : pipe_shards[i]->get_layers()) {
            if (!l.is_pinned) continue;
            auto it = old_pins[i].find(l.il);
            if (it != old_pins[i].end() && it->second) {
                n_skip_up++;
                continue;
            }
            pipe_shards[i]->upload_for_switch(l.il, gpu);
            n_up++;
        }
    }

    auto * registry = model.get_plan_registry();
    auto tier_bs = [&](size_t tier) -> uint32_t {
        return registry && tier < registry->tier_sizes.size() ? registry->tier_sizes[tier] : 0;
    };

    pshard_update_pool_mode(new_plan);

    LLAMA_LOG_DEBUG("%s: tokens=%u tier %zu(bs=%u) -> %zu(bs=%u): %s (n_pinned=%u) -> %s (n_pinned=%u) | down=%d skip=%d up=%d skip=%d\n",
        __func__,
        n_tokens, old_tier, tier_bs(old_tier), new_tier, tier_bs(new_tier),
        llama_pshard_strategy_name(old_plan.strategy), old_plan.n_pinned,
        llama_pshard_strategy_name(new_plan.strategy), new_plan.n_pinned,
        n_down, n_skip_down, n_up, n_skip_up);
}

// restore saved alloc state for the active plan
void llama_context::pshard_reapply_active_plan() {
    if (!pshard_active_plan) {
        return;
    }
    if (!pshard_active_plan->alloc_state.valid) {
        // invalidated by a scheduler rebuild (sched_reserve): the plan is landed, so the
        // reserve under its current placement is the correct one - take it now
        pshard_reserve_and_save(*pshard_active_plan);
        if (!pshard_active_plan->alloc_state.valid) {
            return;
        }
    }
    const llama_pshard_plan & plan = *pshard_active_plan;

    if (model.get_dev_preload_buf()) {
        const size_t buf_total         = ggml_backend_buffer_get_size(model.get_dev_preload_buf());
        const size_t pinned_cache_size = total_pinned_cache_size(memory.get()) + expert_pool_bytes;
        const size_t scratch_off       = plan.cached_scratch_off;
        const size_t scratch_size      = buf_total - scratch_off - pinned_cache_size;

        ggml_backend_sched_set_alloc_range(sched.get(), backends[pshard_layout.compute].get(), scratch_off, scratch_size);
    }

    auto * galloc = ggml_backend_sched_get_galloc(sched.get());
    ggml_gallocr_restore_state(galloc,
        plan.alloc_state.node_allocs.data(), plan.alloc_state.node_allocs.size(),
        plan.alloc_state.leaf_allocs.data(), plan.alloc_state.leaf_allocs.size(),
        plan.alloc_state.n_nodes,            plan.alloc_state.n_leafs);
    ggml_backend_sched_restore_backend_ids(sched.get(),
        plan.alloc_state.node_backend_ids.data(), (int)plan.alloc_state.node_backend_ids.size(),
        plan.alloc_state.leaf_backend_ids.data(), (int)plan.alloc_state.leaf_backend_ids.size());
}

bool llama_context::pshard_prepare_host_access() {
    if (!cparams.pshard || !memory) {
        return false;
    }

    auto pipe_shards = memory->get_pipe_shards();
    if (pipe_shards.empty()) {
        return false;
    }

    for (auto * ps : pipe_shards) {
        ps->prepare_for_host_access();
    }

    pshard_memory_dirty = true;
    return true;
}

void llama_context::pshard_restore_after_host_access() {
    if (!pshard_memory_dirty) {
        return;
    }

    if (pshard_active_plan) {
        pshard_apply_plan(*pshard_active_plan);
    }

    pshard_memory_dirty = false;
}

void llama_context::pshard_maybe_switch(uint32_t n_tokens) {
    if (pshard_memory_dirty) {
        pshard_restore_after_host_access();
    }

    auto * registry = model.get_plan_registry();
    if (!registry) return;

    size_t tier = registry->tier_index(n_tokens);
    llama_pshard_plan * best = registry->get_best(tier);
    if (!best) return;

    if (best != pshard_active_plan) {
        if (pshard_active_plan) {
            size_t old_tier = registry->tier_sizes.size();
            for (size_t i = 0; i < registry->best_plans.size(); i++) {
                if (&registry->best_plans[i] == pshard_active_plan) {
                    old_tier = i;
                    break;
                }
            }
            pshard_switch_plan(*pshard_active_plan, *best, old_tier, tier, n_tokens);
        } else {
            pshard_apply_plan(*best);
        }
        pshard_active_plan = best;
    } else {
        pshard_reapply_active_plan();
    }
}

void llama_context::pshard_update_write_cells(llama_memory_context_i * mctx) {
    g_kv_write_cells.clear();
    g_swa_write_cells.clear();
    g_lid_write_cells.clear();
    for (auto * ps : g_split_ctx.pipe_shards) {
        ps->set_write_cells(nullptr);
        ps->clear_prefetch();
    }

    if (!mctx || g_split_ctx.pipe_shards.empty()) return;

    // bind write_cells to the matching pipe shard
    // indices match get_pipe_shards order
    //   plain KV:       [kv_ps]
    //   iSWA:           [base_ps, swa_ps]
    //   hybrid:         [kv_ps, rs_ps]
    //   hybrid_iswa:    [base_ps, swa_ps, rs_ps]
    //   DSA:            [mla_ps, lid_ps]
    //   DSA_iswa:       [mla_ps, lid_ps, swa_ps]
    //   DSV4:           [raw_base_ps, raw_swa_ps, csa_ps, hca_ps, lid_ps] - only the raw base
    //                   half writes through a KV context; the SWA half and the compressed
    //                   caches (graph-output rows) keep no write-cell set, so their shards
    //                   fall back to whole-layer downloads on a pin change
    auto assign_wc = [&](const llama_kv_cache_context * kv_ctx,
                         std::vector<std::vector<uint32_t>> & storage, size_t ps_idx) {
        if (!kv_ctx || ps_idx >= g_split_ctx.pipe_shards.size()) return;
        storage = kv_ctx->get_write_cells();
        bool has_any = false;
        for (const auto & v : storage) { if (!v.empty()) { has_any = true; break; } }
        if (has_any) {
            g_split_ctx.pipe_shards[ps_idx]->set_write_cells(&storage);
            LLAMA_LOG_DEBUG("%s: bound write_cells to pipe_shard[%zu] (%zu streams)\n",
                __func__, ps_idx, storage.size());
        }
    };

    if (auto * kv_ctx = dynamic_cast<llama_kv_cache_context *>(mctx)) {
        assign_wc(kv_ctx, g_kv_write_cells, 0);
        return;
    }

    if (auto * iswa_ctx = dynamic_cast<llama_kv_cache_iswa_context *>(mctx)) {
        assign_wc(iswa_ctx->get_base(), g_kv_write_cells,  0);
        assign_wc(iswa_ctx->get_swa(),  g_swa_write_cells, 1);
        return;
    }

    if (auto * dsa_ctx = dynamic_cast<llama_kv_cache_dsa_context *>(mctx)) {
        assign_wc(dsa_ctx->get_mla(), g_kv_write_cells,  0);
        assign_wc(dsa_ctx->get_lid(), g_lid_write_cells, 1);
        return;
    }

    if (auto * dsa_iswa = dynamic_cast<llama_kv_cache_dsa_iswa_context *>(mctx)) {
        auto * dsa_ctx = dsa_iswa->get_dsa();
        if (dsa_ctx) {
            assign_wc(dsa_ctx->get_mla(), g_kv_write_cells,  0);
            assign_wc(dsa_ctx->get_lid(), g_lid_write_cells, 1);
        }
        assign_wc(dsa_iswa->get_swa(), g_swa_write_cells, 2);
        return;
    }

    if (auto * d4 = dynamic_cast<llama_kv_cache_dsv4_context *>(mctx)) {
        if (const auto * raw = d4->get_raw()) {
            assign_wc(raw->get_base_ctx(), g_kv_write_cells, 0);
        }
        return;
    }

    if (auto * h = dynamic_cast<llama_memory_hybrid_context *>(mctx)) {
        assign_wc(h->get_attn(), g_kv_write_cells, 0);
        return;
    }

    if (auto * h = dynamic_cast<llama_memory_hybrid_iswa_context *>(mctx)) {
        auto * iswa_ctx = h->get_attn();
        assign_wc(iswa_ctx->get_base(), g_kv_write_cells,  0);
        assign_wc(iswa_ctx->get_swa(),  g_swa_write_cells, 1);
        return;
    }
}
