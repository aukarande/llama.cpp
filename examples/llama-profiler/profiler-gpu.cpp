#define GPU_WARMUP_ITERS  3
#define GPU_TIMED_ITERS   5

#include "profiler-common.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cassert>
#include <climits>
#include <functional>
#include <stdexcept>

// operands shared by every batch of one shape, built once: a MUL_MAT / MUL_MAT_ID weight (a), or K (a) and V (v)
struct gpu_operands {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         a   = nullptr;
    ggml_tensor *         v   = nullptr;
    gpu_operands() {
        ggml_init_params params = { ggml_tensor_overhead() * 4, NULL, true };
        ctx = ggml_init(params);
    }
    gpu_operands(const gpu_operands &) = delete;
    gpu_operands & operator=(const gpu_operands &) = delete;
    ~gpu_operands() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    // allocates the tensors on the device if they fit; a held pinned arena is freed first when they fit only without
    // it (unified memory: pinned host memory takes from the device's pool)
    bool alloc(ggml_backend_t be, const std::string & what, ggml_backend_buffer_t * arena = nullptr) {
        size_t need = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->view_src == nullptr) need += ggml_nbytes(t);
        }
        size_t gpu_free = 0, gpu_total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
        if (need > gpu_free * 0.9 && arena != nullptr && *arena != nullptr) {
            ggml_backend_buffer_free(*arena);
            *arena = nullptr;
            ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
        }
        if (need > gpu_free * 0.9) {
            printf("SKIPPED: %s (need %.2f GB, free %.2f GB)\n", what.c_str(), need / 1e9, gpu_free / 1e9);
            return false;
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { fprintf(stderr, "Failed to allocate GPU buffer\n"); return false; }
        return true;
    }
    // places the tensors in the device's pinned host buffer arena, which its kernels read in place; false when the
    // device has none or its kernels cannot read it (CUDA maps pinned memory into the device; other backends only a
    // host buffer they report as their own), would pad a tensor (the runtime copies such a tensor instead) or the
    // pinned allocation fell back
    bool alloc_host(ggml_backend_t be, ggml_backend_buffer_t & arena) {
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
        if (host == nullptr) return false;
        bool readable = ggml_backend_dev_supports_buft(dev, host);
#ifdef GGML_USE_CUDA
        readable = readable || ggml_backend_is_cuda(be);
#endif
        if (!readable) return false;
        const size_t align = ggml_backend_buft_get_alignment(host);
        size_t need = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->view_src != nullptr) continue;
            if (ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(be), t) != ggml_nbytes(t)) {
                return false;
            }
            need += GGML_PAD(ggml_nbytes(t), align);
        }
        // the arena grows to the largest operands: pinning host memory is slow
        if (arena == nullptr || ggml_backend_buffer_get_size(arena) < need) {
            if (arena) ggml_backend_buffer_free(arena);
            arena = ggml_backend_buft_alloc_buffer(host, need);
            if (arena != nullptr && ggml_backend_buffer_get_type(arena) != host) {
                ggml_backend_buffer_free(arena);
                arena = nullptr;
            }
            if (arena == nullptr) return false;
        }
        char * base = (char *) ggml_backend_buffer_get_base(arena);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->view_src != nullptr) continue;
            ggml_backend_tensor_alloc(arena, t, base);
            base += GGML_PAD(ggml_nbytes(t), align);
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->view_src != nullptr) ggml_backend_view_init(t);
        }
        return true;
    }
};

// batches of the in-place rows: decode, speculative verify and server steps, prompts
static const std::vector<int32_t> host_batch_sizes = { 1, 2, 4, 8, 16, 64, 512, 2048, 4096 };

// warms graph gf up and returns its time per pass; bounded (weights read over the host link): a first pass over 1 ms
// is the warm-up, and the timed passes add up to about 20 ms
static double time_graph(ggml_backend_t be, ggml_cgraph * gf, bool bounded) {
    bench_timer w; w.start();
    ggml_backend_graph_compute(be, gf);
    const double first = w.stop();
    int n_timed = GPU_TIMED_ITERS;
    if (bounded && first > 1e-3) {
        n_timed = std::max(1, std::min(GPU_TIMED_ITERS, (int) std::ceil(0.02 / first)));
    } else {
        for (int i = 1; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    bench_timer t; t.start();
    for (int i = 0; i < n_timed; ++i) ggml_backend_graph_compute_async(be, gf);
    ggml_backend_synchronize(be);
    return t.stop() / n_timed;
}

// fills t from 0.5 floats as the rows always did (for a block quant this call writes only its first blocks)
static void fill_half(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n, 0.5f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / ggml_blck_size(t->type), 1, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
    }
}

static double benchmark_mul_mat_raw(
        ggml_backend_t be, ggml_tensor * A, int N, int K, int batch_size,
        double * out_time_s, double * out_ops, double * out_bytes, bool bounded = false) {

    if ((int64_t)N * batch_size > INT_MAX) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d (output exceeds INT_MAX)\n", N, K, batch_size);
        return 0.0;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B_tensor);
    ggml_build_forward_expand(gf, C);

    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(B_tensor) + ggml_nbytes(C);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d (need %.2f GB, free %.2f GB)\n",
            N, K, batch_size, need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<float> B_data((size_t)K * batch_size, 1.0f);
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));

    double time_per_iter = time_graph(be, gf, bounded);
    double ops_total = 2.0 * N * K * batch_size;
    double bytes_total = (double)(ggml_nbytes(A) + ggml_nbytes(B_tensor) + ggml_nbytes(C));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops)    *out_ops = ops_total;
    if (out_bytes)  *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_mul_mat_id_raw(
        ggml_backend_t be, ggml_tensor * A, int N, int K, int n_experts, int n_experts_used,
        int batch_size,
        double * out_time_s, double * out_ops, double * out_bytes, bool bounded = false) {

    if ((int64_t)N * batch_size * n_experts_used > INT_MAX) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (output exceeds INT_MAX)\n", N, K, batch_size);
        return 0.0;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 0.0; }

    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat_id(ctx, A, B, ids);
    ggml_build_forward_expand(gf, C);

    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(B) + ggml_nbytes(ids) + ggml_nbytes(C);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (need %.2f GB, free %.2f GB)\n",
            N, K, batch_size, need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<float> B_data((size_t)K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    for (int i = 0; i < n_experts_used * batch_size; i++) ids_data[i] = i % n_experts;
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

    double time_per_iter = time_graph(be, gf, bounded);
    double ops_total = 2.0 * N * K * batch_size * n_experts_used;
    double bytes_total = (double)(ggml_nbytes(A) * n_experts_used / n_experts) + ggml_nbytes(B) + ggml_nbytes(C);

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops)    *out_ops = ops_total;
    if (out_bytes)  *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_flash_attn_raw(
        ggml_backend_t be, ggml_tensor * K, ggml_tensor * V, int n_tokens, int ctx_len,
        const attn_config & cfg,
        double * out_time_s, double * out_ops, double * out_bytes) {

    const int n_q_heads  = cfg.n_q_heads;
    const int head_dim   = cfg.head_dim;
    const int head_dim_v = cfg.head_dim_v > 0 ? cfg.head_dim_v : cfg.head_dim;

    if ((int64_t)head_dim * n_tokens * n_q_heads > INT_MAX) {
        printf("SKIPPED: FLASH_ATTN (output exceeds INT_MAX)\n");
        return 0.0;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, 1);
    ggml_tensor * mask = cfg.masked ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ctx_len, n_tokens, 1, 1) : nullptr;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_build_forward_expand(gf, out);

    // a V inside K reads no bytes of its own
    const size_t v_bytes = cfg.v_in_k ? 0 : ggml_nbytes(V);
    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(Q) + ggml_nbytes(out) + (mask ? ggml_nbytes(mask) : 0);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: FLASH_ATTN (need %.2f GB, free %.2f GB)\n", need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    if (mask) {
        // all zero: every position attends, the dense work the runtime's masks leave
        std::vector<uint8_t> z(ggml_nbytes(mask), 0);
        ggml_backend_tensor_set(mask, z.data(), 0, z.size());
    }

    for (int i = 0; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);

    bench_timer t; t.start();
    for (int i = 0; i < GPU_TIMED_ITERS; ++i) ggml_backend_graph_compute_async(be, gf);
    ggml_backend_synchronize(be);
    double total_time = t.stop();

    double time_per_iter = total_time / GPU_TIMED_ITERS;
    // QK^T over the K head size, then the V product over the V head size
    double ops_total = 2.0 * n_tokens * ctx_len * n_q_heads * (double)(head_dim + head_dim_v);
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K) + v_bytes + ggml_nbytes(out));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops)    *out_ops = ops_total;
    if (out_bytes)  *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static void run_matmul_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes,
        int32_t filter_batch, bool fast,
        std::vector<bench_result> & results) {

    auto sizes  = get_matmul_sizes(fast);
    auto quants = get_matmul_quants(fast);
    if (filter_batch >= 0 && std::find(batch_sizes.begin(), batch_sizes.end(), filter_batch) == batch_sizes.end()) return;

    printf("=== MUL_MAT Operations ===\n\n");
    ggml_backend_buffer_t arena = nullptr;
    for (ggml_type qt : quants) {
        printf("--- Quantization: %s ---\n", ggml_type_name(qt));
        row_order<bench_result> order;
        row_order<bench_result> host_order;
        for (size_t si = 0; si < sizes.size(); si++) {
            const auto & sz = sizes[si];
            if (qt == GGML_TYPE_Q2_K && (sz.K % 256 != 0)) continue;

            // the weight in the device's pinned host buffer, read in place (rows MUL_MAT@host)
            for (bool host : { false, true }) {
                gpu_operands w;
                w.a = ggml_new_tensor_2d(w.ctx, qt, sz.K, sz.N);
                if (host ? !w.alloc_host(be, arena) : !w.alloc(be, "MUL_MAT N=" + std::to_string(sz.N) + " K=" + std::to_string(sz.K), &arena)) continue;
                fill_half(w.a);
                const auto & bss = host ? host_batch_sizes : batch_sizes;
                for (size_t bi = 0; bi < bss.size(); bi++) {
                    const int32_t bs = bss[bi];
                    if (filter_batch >= 0 && bs != filter_batch) continue;

                    bench_result res;
                    res.op_name = host ? "MUL_MAT@host" : "MUL_MAT";
                    res.quant_type = ggml_type_name(qt);
                    res.N = sz.N; res.K = sz.K; res.B = bs;
                    benchmark_mul_mat_raw(be, w.a, sz.N, sz.K, bs, &res.time_s, &res.ops, &res.bytes, host);
                    res.calculate_derived();
                    printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                        res.op_name.c_str(), res.quant_type.c_str(),
                        res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                    res.print_dims();
                    printf("\n");
                    (host ? host_order : order).add(bi, si, res);
                }
            }
        }
        order.flush(results);
        host_order.flush(results);
        printf("\n");
    }
    if (arena) ggml_backend_buffer_free(arena);
}

static void run_moe_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes,
        int32_t filter_batch, bool fast,
        std::vector<bench_result> & results) {

    auto configs = get_moe_configs(fast);
    auto quants  = get_matmul_quants(fast);
    if (filter_batch >= 0 && std::find(batch_sizes.begin(), batch_sizes.end(), filter_batch) == batch_sizes.end()) return;

    printf("=== MUL_MAT_ID Operations (MoE) ===\n\n");
    ggml_backend_buffer_t arena = nullptr;
    for (ggml_type qt : quants) {
        printf("--- MoE Quantization: %s ---\n", ggml_type_name(qt));
        row_order<bench_result> order;
        row_order<bench_result> host_order;
        for (size_t si = 0; si < configs.size(); si++) {
            const auto & cfg = configs[si];
            if (qt == GGML_TYPE_Q2_K && (cfg.K % 256 != 0)) continue;
            // the experts in the device's pinned host buffer, read in place (rows MUL_MAT_ID@host)
            for (bool host : { false, true }) {
                try {
                    gpu_operands w;
                    w.a = ggml_new_tensor_3d(w.ctx, qt, cfg.K, cfg.N, cfg.n_experts);
                    if (host ? !w.alloc_host(be, arena) : !w.alloc(be, "MUL_MAT_ID N=" + std::to_string(cfg.N) + " K=" + std::to_string(cfg.K), &arena)) continue;
                    if (host) {
                        ggml_backend_tensor_memset(w.a, 0, 0, ggml_nbytes(w.a));
                    } else {
                        ggml_backend_buffer_clear(w.buf, 0);
                    }
                    const auto & bss = host ? host_batch_sizes : batch_sizes;
                    for (size_t bi = 0; bi < bss.size(); bi++) {
                        const int32_t bs = bss[bi];
                        if (filter_batch >= 0 && bs != filter_batch) continue;

                        bench_result res;
                        res.op_name = host ? "MUL_MAT_ID@host" : "MUL_MAT_ID";
                        res.quant_type = ggml_type_name(qt);
                        res.N = cfg.N; res.K = cfg.K; res.B = bs;
                        res.n_tokens = cfg.n_experts_used; res.ctx_len = cfg.n_experts;
                        double gflops = benchmark_mul_mat_id_raw(be, w.a, cfg.N, cfg.K, cfg.n_experts, cfg.n_experts_used, bs,
                            &res.time_s, &res.ops, &res.bytes, host);
                        if (gflops == 0.0) continue;
                        res.calculate_derived();
                        printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                            res.op_name.c_str(), res.quant_type.c_str(),
                            res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                        res.print_dims();
                        printf("\n");
                        (host ? host_order : order).add(bi, si, res);
                    }
                } catch (const std::exception & e) {
                    printf("SKIPPED: %s N=%d K=%d %s (%s)\n", host ? "MUL_MAT_ID@host" : "MUL_MAT_ID", cfg.N, cfg.K,
                        ggml_type_name(qt), e.what());
                }
            }
        }
        order.flush(results);
        host_order.flush(results);
        printf("\n");
    }
    if (arena) ggml_backend_buffer_free(arena);
}

static void run_attention_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes,
        int32_t filter_batch, bool fast,
        std::vector<bench_result> & results) {

    // shared-KV heads (MLA: V = the first 512 columns of a 576 K; V = K at 512), masked as the runtime calls them.
    // They get every context length and their own token list (decode, MTP verify, prompt).
    static const attn_config shared_kv[] = {
        { "MLA",     64, 1, 576, 512, true, true },
        { "KVS-512", 64, 1, 512, 512, true, true },
    };
    static const std::vector<int32_t> shared_kv_tokens = { 1, 4, 16, 64, 512, 2048 };
    auto configs  = get_attn_configs(fast);
    configs.insert(configs.end(), std::begin(shared_kv), std::end(shared_kv));

    printf("=== FLASH_ATTN Operations ===\n\n");
    for (const auto & cfg : configs) {
        const bool shared = cfg.v_in_k;
        const auto ctx_lens = get_attn_ctx_lens(fast && !shared);
        const auto & tokens = shared ? shared_kv_tokens : batch_sizes;
        if (filter_batch >= 0 && std::find(tokens.begin(), tokens.end(), filter_batch) == tokens.end()) continue;
        const int head_dim_v = cfg.head_dim_v > 0 ? cfg.head_dim_v : cfg.head_dim;
        printf("--- %s (n_q=%d, n_kv=%d, head_dim=%d) ---\n", cfg.name, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim);
        row_order<bench_result> order;
        for (size_t ci = 0; ci < ctx_lens.size(); ci++) {
            const int32_t cl = ctx_lens[ci];
            gpu_operands kv;
            kv.a = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, cfg.head_dim, cl, cfg.n_kv_heads, 1);
            kv.v = cfg.v_in_k
                ? ggml_view_4d(kv.ctx, kv.a, head_dim_v, cl, cfg.n_kv_heads, 1, kv.a->nb[1], kv.a->nb[2], kv.a->nb[3], 0)
                : ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, head_dim_v, cl, cfg.n_kv_heads, 1);
            if (!kv.alloc(be, "FLASH_ATTN ctx=" + std::to_string(cl))) continue;
            fill_half(kv.a);
            if (!cfg.v_in_k) {
                fill_half(kv.v);
            }
            for (size_t ti = 0; ti < tokens.size(); ti++) {
                const int32_t n_tok = tokens[ti];
                if (filter_batch >= 0 && n_tok != filter_batch) continue;

                bench_result res;
                res.op_name = std::string("FLASH_ATTN_") + cfg.name;
                res.quant_type = ggml_type_name(GGML_TYPE_F16);
                res.n_tokens = n_tok; res.ctx_len = cl; res.n_heads = cfg.n_kv_heads; res.head_dim = cfg.head_dim;
                benchmark_flash_attn_raw(be, kv.a, kv.v, n_tok, cl, cfg, &res.time_s, &res.ops, &res.bytes);
                res.calculate_derived();
                printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                    res.op_name.c_str(), res.quant_type.c_str(),
                    res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                res.print_dims();
                printf("\n");
                order.add(ti, ci, res);
            }
        }
        order.flush(results);
        printf("\n");
    }
}

// bytes and elements of a small op as the planner's predictor counts them (llama_op_metrics_compute), so a row's
// rate prices a model node from the node's own bytes
static void small_op_size(const ggml_tensor * t, double & bytes, int64_t & n_elements) {
    auto nb = [](const ggml_tensor * x) { return x ? (double) ggml_nbytes(x) : 0.0; };
    n_elements = ggml_nelements(t);
    switch (t->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_GLU:
            bytes = nb(t->src[0]) + nb(t->src[1]) + nb(t);
            break;
        case GGML_OP_GET_ROWS:
            bytes = nb(t);
            break;
        default:
            bytes = nb(t->src[0]) + nb(t);
            break;
    }
}

// the small ops a model graph is made of (elementwise, norms, activations, routing, copies, the short convolution
// and the gated delta-net recurrence), at prompt and decode token counts. A graph holds several independent
// copies of the op, so a row's time includes a node's share of the launch cost as it runs inside a model graph
static void run_small_op_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes, int32_t filter_batch,
        std::vector<bench_result> & results) {
    const int64_t W    = 4096;   // activation width
    const int64_t NE   = 256;    // router width
    const int64_t S    = 128;    // delta-net head size
    const int64_t HK   = 16;     // delta-net key heads
    const int64_t HV   = 32;     // delta-net value heads
    const int64_t CONV = 4;      // short convolution width
    const int     REPS = 8;
    const int32_t max_tokens = 2048;

    struct op_case {
        const char * name;
        // builds REPS nodes from shared inputs; returns the inputs that take integer data (row ids)
        std::function<void(ggml_context *, int64_t, std::vector<ggml_tensor *> &, std::vector<ggml_tensor *> &)> build;
    };
    auto f32 = [](ggml_context * c, int64_t a, int64_t b) { return ggml_new_tensor_2d(c, GGML_TYPE_F32, a, b); };
    const std::vector<op_case> cases = {
        { "ADD", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T), * b = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_add(c, a, b)); } },
        { "MUL", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T), * b = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_mul(c, a, b)); } },
        { "RMS_NORM", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_rms_norm(c, a, 1e-6f)); } },
        { "L2_NORM", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, S, HV * T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_l2_norm(c, a, 1e-6f)); } },
        { "SCALE", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_scale(c, a, 0.5f)); } },
        { "UNARY", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_silu(c, a)); } },
        { "GLU", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T), * b = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_swiglu_split(c, a, b)); } },
        { "SOFT_MAX", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, NE, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_soft_max(c, a)); } },
        { "ARGSORT", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, NE, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_argsort(c, a, GGML_SORT_ORDER_DESC)); } },
        { "CONCAT", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T), * b = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_concat(c, a, b, 0)); } },
        { "CPY", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_cpy(c, a, ggml_new_tensor_2d(c, GGML_TYPE_F16, W, T))); } },
        { "CONT", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * a = f32(c, W, T);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_cont(c, ggml_transpose(c, a))); } },
        { "GET_ROWS", [&](ggml_context * c, int64_t T, auto & o, auto & ids) {
            ggml_tensor * a = f32(c, W, 1024);
            ggml_tensor * r = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
            ids.push_back(r);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_get_rows(c, a, r)); } },
        { "SSM_CONV", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * sx = ggml_new_tensor_3d(c, GGML_TYPE_F32, CONV - 1 + T, 2 * S * HK + S * HV, 1);
            ggml_tensor * k  = f32(c, CONV, 2 * S * HK + S * HV);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_ssm_conv(c, sx, k)); } },
        { "GATED_DELTA_NET", [&](ggml_context * c, int64_t T, auto & o, auto &) {
            ggml_tensor * q  = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, HK, T, 1);
            ggml_tensor * k  = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, HK, T, 1);
            ggml_tensor * v  = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, HV, T, 1);
            ggml_tensor * g  = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, HV, T, 1);
            ggml_tensor * bt = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, HV, T, 1);
            ggml_tensor * st = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, HV, 1);
            for (int i = 0; i < REPS; i++) o.push_back(ggml_gated_delta_net(c, q, k, v, g, bt, st, 1)); } },
    };

    printf("=== Small ops ===\n\n");
    for (const auto & oc : cases) {
        for (int32_t T : batch_sizes) {
            if (T > max_tokens || (filter_batch >= 0 && T != filter_batch)) continue;
            ggml_init_params params = { 64ULL * 1024 * 1024, NULL, true };
            ggml_context * ctx = ggml_init(params);
            std::vector<ggml_tensor *> outs, ids;
            try {
                oc.build(ctx, T, outs, ids);
            } catch (const std::exception & e) {
                printf("SKIPPED: %s T=%d (%s)\n", oc.name, T, e.what());
                ggml_free(ctx);
                continue;
            }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);

            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (!buffer) {
                printf("SKIPPED: %s T=%d (allocation)\n", oc.name, T);
                ggml_free(ctx);
                continue;
            }
            // inputs: small finite values, row ids inside the table
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                if (t->op != GGML_OP_NONE || t->view_src != nullptr) continue;
                if (t->type == GGML_TYPE_F32) {
                    std::vector<float> d((size_t) ggml_nelements(t), 0.01f);
                    ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
                } else if (t->type == GGML_TYPE_I32) {
                    std::vector<int32_t> d((size_t) ggml_nelements(t));
                    for (size_t i = 0; i < d.size(); i++) d[i] = (int32_t) (i % 1024);
                    ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
                }
            }

            for (int i = 0; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            bench_timer tm; tm.start();
            for (int i = 0; i < GPU_TIMED_ITERS; ++i) ggml_backend_graph_compute_async(be, gf);
            ggml_backend_synchronize(be);
            const double t_node = tm.stop() / GPU_TIMED_ITERS / (double) outs.size();

            bench_result res;
            res.op_name    = ggml_op_name(outs[0]->op);
            res.quant_type = ggml_type_name(outs[0]->type);
            small_op_size(outs[0], res.bytes, res.n_elements);
            res.time_s = t_node;
            res.calculate_derived();
            printf("%-20s quant=%-6s BW=%.2f GB/s time=%.2f us", res.op_name.c_str(), res.quant_type.c_str(),
                res.effective_bw_gb_s, t_node * 1e6);
            res.print_dims();
            printf("\n");
            results.push_back(res);

            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    printf("\n");
}

static void save_results_gpu(
        const char * path,
        const std::vector<bench_result> & results,
        const char * backend_name,
        const std::vector<int32_t> & batch_sizes,
        double peak_gpu_bw, double peak_gpu_compute, double peak_host_bw) {

    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); return; }

    fprintf(f, "# GPU Profiling (backend=%s, batch_sizes=[", backend_name);
    for (size_t i = 0; i < batch_sizes.size(); i++)
        fprintf(f, "%d%s", batch_sizes[i], i + 1 < batch_sizes.size() ? "," : "");
    fprintf(f, "], GPU_Memory_BW=%.1f GB/s, GPU_Peak_Compute=%.1f GFLOP/s)\n", peak_gpu_bw, peak_gpu_compute);
    fprintf(f, "# op_name quant AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) N K B n_tokens ctx_len n_heads head_dim n_elements\n");

    // the in-place rows' ridge sits at the in-place read rate, not the device memory rate
    std::map<std::string, double> ridge_map;
    for (const auto & rr : compute_ridge_points(results, peak_gpu_bw)) ridge_map[rr.key] = rr.ridge;
    for (const auto & rr : compute_ridge_points(results, peak_host_bw)) {
        if (rr.key.find("@host") != std::string::npos) ridge_map[rr.key] = rr.ridge;
    }

    for (const auto & r : results) {
        std::string key = r.op_name + "_" + r.quant_type;
        double ridge = ridge_map.count(key) ? ridge_map[key] : 0.0;
        fprintf(f, "%s %s %.4f %.2f %.2f %.4f %d %d %d %d %d %d %d %lld\n",
            r.op_name.c_str(), r.quant_type.c_str(),
            r.arithmetic_intensity, r.effective_bw_gb_s, r.effective_gflops, ridge,
            r.N, r.K, r.B, r.n_tokens, r.ctx_len, r.n_heads, r.head_dim, (long long)r.n_elements);
    }

    fclose(f);
    printf("Results saved to %s (%zu benchmarks)\n", path, results.size());
}

int main(int argc, char ** argv) {
    bool fast_mode = true;
    int32_t filter_batch = -1;
    const char * output_path = "gpu_profile.txt";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--fast")) {
            fast_mode = true;
        } else if (!strcmp(argv[i], "--full")) {
            fast_mode = false;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [options]\n", argv[0]);
            printf("\n");
            printf("options:\n");
            printf("  -h, --help\n");
            printf("  --fast              fast mode with fewer configs (default)\n");
            printf("  --full              full mode with all configs\n");
            printf("  --batch <n>         only run batch size N\n");
            printf("  --output <path>     output file (default: gpu_profile.txt)\n");
            return 0;
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], filter_batch)) {
                fprintf(stderr, "Invalid --batch value: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    std::vector<int32_t> batch_sizes = { 1, 64, 512, 1024, 2048, 4096, 8192, 16384 };

    printf("=== GPU Profiler ===\n");
    printf("Mode: %s\n\n", fast_mode ? "FAST" : "FULL");

    ggml_backend_t gpu_be = profiler_gpu_backend_init();
    if (!gpu_be) {
        fprintf(stderr, "No GPU backend available. Cannot run GPU profiler.\n");
        return 1;
    }
    printf("GPU: %s\n\n", ggml_backend_name(gpu_be));

    ggml_quantize_init(GGML_TYPE_Q2_K);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    ggml_quantize_init(GGML_TYPE_Q4_1);
    ggml_quantize_init(GGML_TYPE_Q5_0);
    ggml_quantize_init(GGML_TYPE_Q8_0);
    ggml_quantize_init(GGML_TYPE_MXFP4);

    std::vector<bench_result> all_results;
    bench_timer overall; overall.start();

    run_matmul_benchmarks(gpu_be, batch_sizes, filter_batch, fast_mode, all_results);
    run_moe_benchmarks(gpu_be, batch_sizes, filter_batch, fast_mode, all_results);
    run_attention_benchmarks(gpu_be, batch_sizes, filter_batch, fast_mode, all_results);
    run_small_op_benchmarks(gpu_be, batch_sizes, filter_batch, all_results);

    double peak_gpu_bw = 0.0, peak_gpu_compute = 0.0, peak_host_bw = 0.0;
    for (const auto & r : all_results) {
        const bool host = r.op_name.find("@host") != std::string::npos;
        if (r.arithmetic_intensity < 2.0 && r.ops > 0.0) {
            double & bw = host ? peak_host_bw : peak_gpu_bw;
            bw = std::max(bw, (double)r.effective_bw_gb_s);
        }
        if (!host && (r.arithmetic_intensity > 10.0 || (r.op_name == "MUL_MAT" && r.N >= 4096 && r.K >= 4096)))
            peak_gpu_compute = std::max(peak_gpu_compute, (double)r.effective_gflops);
    }
    printf("\nEstimated GPU Memory BW: %.1f GB/s\n", peak_gpu_bw);
    printf("Estimated GPU Compute:   %.1f GFLOP/s\n", peak_gpu_compute);
    printf("Estimated in-place read: %.1f GB/s\n", peak_host_bw);
    printf("Total time: %.1f s, %zu benchmarks\n", overall.stop(), all_results.size());

    save_results_gpu(output_path, all_results, ggml_backend_name(gpu_be), batch_sizes, peak_gpu_bw, peak_gpu_compute, peak_host_bw);

    ggml_backend_free(gpu_be);
    ggml_quantize_free();
    return 0;
}
