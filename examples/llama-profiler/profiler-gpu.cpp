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

static double benchmark_mul_mat_raw(
        ggml_backend_t be, int N, int K, int batch_size, ggml_type quant,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 4096ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * A = ggml_new_tensor_2d(ctx, quant, K, N);
    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B_tensor);
    ggml_build_forward_expand(gf, C);

    if ((int64_t)N * batch_size > INT_MAX) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d (output exceeds INT_MAX)\n", N, K, batch_size);
        ggml_free(ctx); return 0.0;
    }

    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(A) + ggml_nbytes(B_tensor) + ggml_nbytes(C);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d (need %.2f GB, free %.2f GB)\n",
            N, K, batch_size, need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<float> A_float((size_t)K * N, 0.5f);
    if (quant == GGML_TYPE_F32) {
        ggml_backend_tensor_set(A, A_float.data(), 0, ggml_nbytes(A));
    } else {
        std::vector<uint8_t> A_q(ggml_nbytes(A));
        ggml_quantize_chunk(quant, A_float.data(), A_q.data(), 0,
            (int64_t)K * N / ggml_blck_size(quant), 1, nullptr);
        ggml_backend_tensor_set(A, A_q.data(), 0, ggml_nbytes(A));
    }
    std::vector<float> B_data((size_t)K * batch_size, 1.0f);
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));

    for (int i = 0; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);

    bench_timer t; t.start();
    for (int i = 0; i < GPU_TIMED_ITERS; ++i) ggml_backend_graph_compute_async(be, gf);
    ggml_backend_synchronize(be);
    double total_time = t.stop();

    double time_per_iter = total_time / GPU_TIMED_ITERS;
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
        ggml_backend_t be, int N, int K, int n_experts, int n_experts_used,
        int batch_size, ggml_type quant,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 0.0; }

    ggml_tensor * A   = ggml_new_tensor_3d(ctx, quant, K, N, n_experts);
    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    if ((int64_t)N * batch_size * n_experts_used > INT_MAX) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (output exceeds INT_MAX)\n", N, K, batch_size);
        ggml_free(ctx); return 0.0;
    }

    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(A) + ggml_nbytes(B) + ggml_nbytes(ids);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (need %.2f GB, free %.2f GB)\n",
            N, K, batch_size, need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat_id(ctx, A, B, ids);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<uint8_t> A_data(ggml_nbytes(A), 0);
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));
    std::vector<float> B_data((size_t)K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    for (int i = 0; i < n_experts_used * batch_size; i++) ids_data[i] = i % n_experts;
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

    for (int i = 0; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);

    bench_timer t; t.start();
    for (int i = 0; i < GPU_TIMED_ITERS; ++i) ggml_backend_graph_compute_async(be, gf);
    ggml_backend_synchronize(be);
    double total_time = t.stop();

    double time_per_iter = total_time / GPU_TIMED_ITERS;
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
        ggml_backend_t be, int n_tokens, int ctx_len,
        int n_q_heads, int n_kv_heads, int head_dim, ggml_type kv_quant,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, 1);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_build_forward_expand(gf, out);

    if ((int64_t)head_dim * n_tokens * n_q_heads > INT_MAX) {
        printf("SKIPPED: FLASH_ATTN (output exceeds INT_MAX)\n");
        ggml_free(ctx); return 0.0;
    }

    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);
    size_t need = ggml_nbytes(Q) + ggml_nbytes(K) + ggml_nbytes(V) + ggml_nbytes(out);
    if (need > gpu_free * 0.9) {
        printf("SKIPPED: FLASH_ATTN (need %.2f GB, free %.2f GB)\n", need / 1e9, gpu_free / 1e9);
        ggml_free(ctx); return 0.0;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) { fprintf(stderr, "Failed to allocate GPU buffer\n"); ggml_free(ctx); return 0.0; }

    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    int64_t kv_elems = ggml_nelements(K);
    std::vector<float> KV_float(kv_elems, 0.5f);
    if (kv_quant == GGML_TYPE_F32) {
        ggml_backend_tensor_set(K, KV_float.data(), 0, ggml_nbytes(K));
        ggml_backend_tensor_set(V, KV_float.data(), 0, ggml_nbytes(V));
    } else {
        std::vector<uint8_t> Kq(ggml_nbytes(K)), Vq(ggml_nbytes(V));
        ggml_quantize_chunk(kv_quant, KV_float.data(), Kq.data(), 0, kv_elems / ggml_blck_size(kv_quant), 1, nullptr);
        ggml_quantize_chunk(kv_quant, KV_float.data(), Vq.data(), 0, kv_elems / ggml_blck_size(kv_quant), 1, nullptr);
        ggml_backend_tensor_set(K, Kq.data(), 0, ggml_nbytes(K));
        ggml_backend_tensor_set(V, Vq.data(), 0, ggml_nbytes(V));
    }

    for (int i = 0; i < GPU_WARMUP_ITERS; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);

    bench_timer t; t.start();
    for (int i = 0; i < GPU_TIMED_ITERS; ++i) ggml_backend_graph_compute_async(be, gf);
    ggml_backend_synchronize(be);
    double total_time = t.stop();

    double time_per_iter = total_time / GPU_TIMED_ITERS;
    double ops_total = 2.0 * n_tokens * head_dim * ctx_len * n_q_heads * 2;
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K) + ggml_nbytes(V) + ggml_nbytes(out));

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

    printf("=== MUL_MAT Operations ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            if (filter_batch >= 0 && bs != filter_batch) continue;
            printf("  [Batch=%d]\n", bs);
            for (const auto & sz : sizes) {
                if (qt == GGML_TYPE_Q2_K && (sz.K % 256 != 0)) continue;

                bench_result res;
                res.op_name = "MUL_MAT";
                res.quant_type = ggml_type_name(qt);
                res.N = sz.N; res.K = sz.K; res.B = bs;
                benchmark_mul_mat_raw(be, sz.N, sz.K, bs, qt, &res.time_s, &res.ops, &res.bytes);
                res.calculate_derived();
                printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                    res.op_name.c_str(), res.quant_type.c_str(),
                    res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                res.print_dims();
                printf("\n");
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void run_moe_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes,
        int32_t filter_batch, bool fast,
        std::vector<bench_result> & results) {

    auto configs = get_moe_configs(fast);
    auto quants  = get_matmul_quants(fast);

    printf("=== MUL_MAT_ID Operations (MoE) ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- MoE Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            if (filter_batch >= 0 && bs != filter_batch) continue;
            printf("  [Batch=%d]\n", bs);
            for (const auto & cfg : configs) {
                if (qt == GGML_TYPE_Q2_K && (cfg.K % 256 != 0)) continue;
                try {
                    bench_result res;
                    res.op_name = "MUL_MAT_ID";
                    res.quant_type = ggml_type_name(qt);
                    res.N = cfg.N; res.K = cfg.K; res.B = bs;
                    res.n_tokens = cfg.n_experts_used; res.ctx_len = cfg.n_experts;
                    double gflops = benchmark_mul_mat_id_raw(be, cfg.N, cfg.K, cfg.n_experts, cfg.n_experts_used, bs, qt,
                        &res.time_s, &res.ops, &res.bytes);
                    if (gflops == 0.0) continue;
                    res.calculate_derived();
                    printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                        res.op_name.c_str(), res.quant_type.c_str(),
                        res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                    res.print_dims();
                    printf("\n");
                    results.push_back(res);
                } catch (const std::exception & e) {
                    printf("SKIPPED: MUL_MAT_ID N=%d K=%d %s (%s)\n", cfg.N, cfg.K, ggml_type_name(qt), e.what());
                }
            }
        }
        printf("\n");
    }
}

static void run_attention_benchmarks(
        ggml_backend_t be, const std::vector<int32_t> & batch_sizes,
        int32_t filter_batch, bool fast,
        std::vector<bench_result> & results) {

    auto configs  = get_attn_configs(fast);
    auto ctx_lens = get_attn_ctx_lens(fast);

    printf("=== FLASH_ATTN Operations ===\n\n");
    for (const auto & cfg : configs) {
        printf("--- %s (n_q=%d, n_kv=%d, head_dim=%d) ---\n", cfg.name, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim);
        for (int32_t n_tok : batch_sizes) {
            if (filter_batch >= 0 && n_tok != filter_batch) continue;
            printf("  [n_tokens=%d]\n", n_tok);
            for (int32_t cl : ctx_lens) {
                bench_result res;
                res.op_name = std::string("FLASH_ATTN_") + cfg.name;
                res.quant_type = ggml_type_name(GGML_TYPE_F16);
                res.n_tokens = n_tok; res.ctx_len = cl; res.n_heads = cfg.n_kv_heads; res.head_dim = cfg.head_dim;
                benchmark_flash_attn_raw(be, n_tok, cl, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, GGML_TYPE_F16,
                    &res.time_s, &res.ops, &res.bytes);
                res.calculate_derived();
                printf("%-20s quant=%-6s AI=%.3f BW=%.2f GB/s Perf=%.2f GFLOP/s",
                    res.op_name.c_str(), res.quant_type.c_str(),
                    res.arithmetic_intensity, res.effective_bw_gb_s, res.effective_gflops);
                res.print_dims();
                printf("\n");
                results.push_back(res);
            }
        }
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
        double peak_gpu_bw, double peak_gpu_compute) {

    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); return; }

    fprintf(f, "# GPU Profiling (backend=%s, batch_sizes=[", backend_name);
    for (size_t i = 0; i < batch_sizes.size(); i++)
        fprintf(f, "%d%s", batch_sizes[i], i + 1 < batch_sizes.size() ? "," : "");
    fprintf(f, "], GPU_Memory_BW=%.1f GB/s, GPU_Peak_Compute=%.1f GFLOP/s)\n", peak_gpu_bw, peak_gpu_compute);
    fprintf(f, "# op_name quant AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) N K B n_tokens ctx_len n_heads head_dim n_elements\n");

    auto ridges = compute_ridge_points(results, peak_gpu_bw);
    std::map<std::string, double> ridge_map;
    for (const auto & rr : ridges) ridge_map[rr.key] = rr.ridge;

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

    double peak_gpu_bw = 0.0, peak_gpu_compute = 0.0;
    for (const auto & r : all_results) {
        if (r.arithmetic_intensity < 2.0 && r.ops > 0.0)
            peak_gpu_bw = std::max(peak_gpu_bw, (double)r.effective_bw_gb_s);
        if (r.arithmetic_intensity > 10.0 || (r.op_name == "MUL_MAT" && r.N >= 4096 && r.K >= 4096))
            peak_gpu_compute = std::max(peak_gpu_compute, (double)r.effective_gflops);
    }
    printf("\nEstimated GPU Memory BW: %.1f GB/s\n", peak_gpu_bw);
    printf("Estimated GPU Compute:   %.1f GFLOP/s\n", peak_gpu_compute);
    printf("Total time: %.1f s, %zu benchmarks\n", overall.stop(), all_results.size());

    save_results_gpu(output_path, all_results, ggml_backend_name(gpu_be), batch_sizes, peak_gpu_bw, peak_gpu_compute);

    ggml_backend_free(gpu_be);
    ggml_quantize_free();
    return 0;
}
