// test-pshard-spec.cpp - pshard speculative-decoding certification.
//
// The same prompt, the same greedy sampler and the same draft are run through llama.cpp's own
// speculative driver (the loop of examples/speculative-simple) under several pshard strategies.
// Bars:
//   - DETERMINISM: every strategy is run twice; token stream, accepted/drafted counts and a
//     running FNV-1a hash over EVERY verify step's full output logits must be identical between
//     the two runs (bitwise-strength: races, stale slots, uninitialized scratch all show here),
//   - PARITY across strategies: the first verify step sees the identical input in every mode
//     (the prompt), so its position-0 logits are compared value-wise and the pairwise
//     max |delta| matrix is printed with each mode's top-2 margin. Enforced only when
//     LLAMA_TEST_PSHARD_SPEC_TOL is set: on MoE targets a placement-level rounding difference
//     can flip an expert-routing near-tie and move logits by O(1) (q35: s0 vs s1 ~1.0 while
//     each is bitwise deterministic), so the tolerance is per model. Token-stream equality
//     across strategies is REPORTED, and enforced with LLAMA_TEST_PSHARD_SPEC_STRICT=1,
//   - pshard actually engaged (a silent stock fallback fails the test),
//   - the draft accelerated something (n_accept > 0).
// The spec batches are the tier exercise: the prompt runs the prefill tier, every verify batch
// (1 + n_draft tokens) the small-batch tier, so plan switches happen around every step.
// Optionally the stock side at the SAME budget (-fitb) is run as a reference: token agreement
// is reported, and enforced with LLAMA_TEST_PSHARD_SPEC_STOCK_STRICT=1 (greedy near-ties can
// legitimately diverge between placements, see qa/README.md).
//
// Env (skips when the target is unset):
//   LLAMA_TEST_PSHARD_SPEC_TARGET   target gguf (a NextN-bearing gguf when no draft is given -> MTP)
//   LLAMA_TEST_PSHARD_SPEC_DRAFT    separate draft gguf (optional)
//   LLAMA_TEST_PSHARD_SPEC_MVA      budget MiB for both sides (default 4000)
//   LLAMA_TEST_PSHARD_SPEC_CTX      n_ctx (default 2048)
//   LLAMA_TEST_PSHARD_SPEC_NPREDICT tokens to generate (default 48)
//   LLAMA_TEST_PSHARD_SPEC_NDRAFT   draft length (default 8; 2 for MTP)
//   LLAMA_TEST_PSHARD_SPEC_STRATS   comma list of strategies, "auto" or 0..4 (default "0,1,auto")
//   LLAMA_TEST_PSHARD_SPEC_TOL      max |logit delta| of the first verify step across modes (0 = report only, default)
//   LLAMA_TEST_PSHARD_SPEC_STRICT   1 -> cross-strategy token/hash mismatch fails the test
//   LLAMA_TEST_PSHARD_SPEC_STOCK    1 (default) also run the stock reference at -fitb MVA
//   LLAMA_TEST_PSHARD_SPEC_STOCK_STRICT 1 -> stock token mismatch fails the test

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::fflush(stdout);                                               \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) { setenv(k, v, 1); } else { unsetenv(k); }
#endif
}

static const char * env_or(const char * k, const char * def) {
    const char * v = std::getenv(k);
    return v && *v ? v : def;
}

static const char * SPEC_PROMPT =
    "The history of computing begins long before the first electronic machines. Mechanical "
    "calculators, punched cards, and tabulating engines each carried one idea forward: that";

struct spec_cfg {
    std::string target;
    std::string draft;      // empty -> MTP over the target's own NextN head
    bool        mtp = false;
    std::vector<enum common_speculative_type> types;
    int32_t     n_draft_max = 8;
    int32_t     mva         = 4000;
    uint32_t    n_ctx       = 2048;
    int32_t     n_predict   = 48;
};

struct spec_result {
    std::vector<llama_token> tokens;
    int      n_accept    = 0;
    int      n_drafted   = 0;
    int      n_predict   = 0;
    uint64_t logits_hash = 1469598103934665603ull;  // FNV-1a offset basis
    std::vector<float> first_logits;                 // position-0 logits of the first verify step
    double   dec_tps     = 0.0;
    bool     ok          = false;
    bool     pshard      = false;
};

// gap between the best and the second-best logit: a small gap means the greedy pick was a
// near-tie that any placement-level rounding can flip
static float top2_margin(const std::vector<float> & l) {
    float m1 = -1e30f, m2 = -1e30f;
    for (float v : l) {
        if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
    }
    return m1 - m2;
}

static float max_abs_delta(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return 1e30f;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) { m = d; }
    }
    return m;
}

static void hash_bytes(uint64_t & h, const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
}

static common_params make_params(const spec_cfg & c, bool pshard) {
    common_params p;
    p.model.path        = c.target;
    p.n_ctx             = c.n_ctx;
    p.n_predict         = c.n_predict;
    p.sampling.temp     = 0.0f;  // greedy: determinism is the bar
    p.sampling.seed     = 42;
    p.speculative.types = c.types;
    if (!c.mtp) {
        p.speculative.draft.mparams.path = c.draft;
    }
    p.speculative.draft.n_max = c.n_draft_max;
    // a hand-built params struct skips common_params_parse's post-processing: resolve the
    // thread counts (-1 = auto) exactly as it does, for the target and the draft alike
    postprocess_cpu_params(p.cpuparams,                         nullptr);
    postprocess_cpu_params(p.cpuparams_batch,                   &p.cpuparams);
    postprocess_cpu_params(p.speculative.draft.cpuparams,       &p.cpuparams);
    postprocess_cpu_params(p.speculative.draft.cpuparams_batch, &p.cpuparams_batch);
    p.pshard         = pshard;
    p.max_vram_alloc = pshard ? (size_t) c.mva : 0;
    if (!pshard) {
        // the stock side at the SAME budget for weights + KV + compute (qa perf rule)
        std::fill(p.fit_params_budget.begin(), p.fit_params_budget.end(), (size_t) c.mva * 1024 * 1024);
    }
    // pad the override buffer the fit writes into (common_params_parse does the same)
    while (p.tensor_buft_overrides.size() < llama_max_tensor_buft_overrides()) {
        p.tensor_buft_overrides.push_back({ nullptr, nullptr, -1 });
    }
    return p;
}

// mirror of tools/pshard-plan-params plan_pshard_context: fresh registry for the forced strategy
// currently in PSHARD_STRATEGY (unset = auto); the runtime only LOADS registries
static void plan_pshard(const spec_cfg & c) {
    std::remove((c.target + ".tensor_overrides.pshard_registry").c_str());

    common_params params = make_params(c, true);
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = c.n_ctx;
    params.tensor_buft_overrides.assign(4096, {});

    const uint32_t tier_max = std::min(std::max(cparams.n_batch, (uint32_t) 16384), cparams.n_ctx);
    const auto limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max          = limits.total;
    params.n_outputs_max_per_seq  = limits.per_seq;
    cparams.n_outputs_max         = limits.total;
    cparams.n_outputs_max_per_seq = limits.per_seq;
    const uint32_t n_draft_tier = limits.per_seq > 1 ? (uint32_t) limits.per_seq - 1 : 0;

    mparams.pshard_registry        = llama_pshard_registry_create(tier_max, cparams.n_seq_max, n_draft_tier);
    mparams.pshard_cache_skip_load = true;

    size_t mva_eff = params.max_vram_alloc;
    if (const size_t dres = common_pshard_draft_reserve_mb(params, cparams.n_ctx); dres > 0) {
        mva_eff = mva_eff > dres ? mva_eff - dres : 1;
    }
    llama_params_fit_pshard_plan(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_buft_overrides.data(), mva_eff, 0);
    llama_pshard_registry_free(mparams.pshard_registry);
}

// one full speculative generation: the loop of examples/speculative-simple with a hash over
// every verify step's logits
static spec_result run_spec(const spec_cfg & c, bool pshard) {
    spec_result r;

    common_params params = make_params(c, pshard);
    const auto limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    if (!model_tgt || !ctx_tgt) {
        std::printf("FAIL: target load failed\n");
        return r;
    }
    r.pshard = llama_model_pshard_active(model_tgt);
    const llama_vocab * vocab   = llama_model_get_vocab(model_tgt);
    const size_t        n_vocab = (size_t) llama_vocab_n_tokens(vocab);

    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        if (!spec_init || !spec_init->context()) {
            std::printf("FAIL: draft/MTP context init failed\n");
            return r;
        }
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = ctx_dft && common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    const llama_seq_id seq_id = 0;
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, SPEC_PROMPT, true, true);
    if (inp.size() < 2) {
        std::printf("FAIL: prompt tokenized to %zu tokens\n", inp.size());
        return r;
    }

    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (!smpl || !spec) {
        std::printf("FAIL: sampler/speculator init failed\n");
        return r;
    }

    {
        llama_batch batch_prompt = llama_batch_init((int32_t) inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_prompt, inp[i], (llama_pos) i, { seq_id }, false);
        }
        llama_decode(ctx_tgt, batch_prompt);
        const bool okp = common_speculative_process(spec, batch_prompt);
        llama_batch_free(batch_prompt);
        if (!okp) {
            std::printf("FAIL: speculative prompt processing failed\n");
            common_speculative_free(spec);
            return r;
        }
    }

    llama_token  id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch  batch_tgt = llama_batch_init((int32_t) llama_n_batch(ctx_tgt), 0, 1);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    bool has_eos = false;

    const int64_t t0 = ggml_time_us();
    while (true) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(),
                            llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            n_draft_max = std::min(n_draft_max, params.n_predict - r.n_predict - 1);
            n_draft_max = std::max(n_draft_max, 0);
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ n_draft_max,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(spec);
            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], (llama_pos) (n_past + i), { seq_id }, true);
        }
        llama_decode(ctx_tgt, batch_tgt);

        // the bar: every verify step's full logits, all outputs
        for (int32_t oi = 0; oi < batch_tgt.n_tokens; ++oi) {
            const float * l = llama_get_logits_ith(ctx_tgt, oi);
            if (l) {
                hash_bytes(r.logits_hash, l, n_vocab * sizeof(float));
                if (oi == 0 && r.first_logits.empty()) {
                    r.first_logits.assign(l, l + n_vocab);  // identical input in every mode
                }
            }
        }

        if (!common_speculative_process(spec, batch_tgt)) {
            std::printf("FAIL: speculative batch processing failed\n");
            break;
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        const size_t n_draft_sz = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        GGML_ASSERT(ids.size() > 0);

        if (use_ckpt_tgt && ids.size() - 1 < n_draft_sz) {
            draft = std::move(ids);
            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl   = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));
        n_past      += (int) ids.size() - 1;
        r.n_drafted += (int) n_draft_sz;
        r.n_accept  += (int) ids.size() - 1;
        r.n_predict += (int) ids.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            r.tokens.push_back(id_last);
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        }
        if (r.n_predict >= params.n_predict || has_eos) {
            break;
        }
    }
    const int64_t t1 = ggml_time_us();
    r.dec_tps = r.n_predict / std::max(1e-6, (t1 - t0) / 1e6);
    r.ok      = r.n_predict > 0;

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    spec_init.reset();  // draft ctx/model down before the target
    return r;
}

static void print_result(const char * tag, const spec_result & r) {
    std::printf("%-12s %s pshard=%d predicted=%d accepted=%d/%d decode=%.1f tok/s logits_hash=%016llx\n",
                tag, r.ok ? "ok  " : "FAIL", (int) r.pshard, r.n_predict, r.n_accept, r.n_drafted,
                r.dec_tps, (unsigned long long) r.logits_hash);
    std::fflush(stdout);
}

static bool tokens_equal(const char * tag, const spec_result & a, const spec_result & b) {
    const size_t n = std::min(a.tokens.size(), b.tokens.size());
    for (size_t i = 0; i < n; ++i) {
        if (a.tokens[i] != b.tokens[i]) {
            std::printf("%s: token stream diverges at %zu: %d vs %d\n", tag, i, a.tokens[i], b.tokens[i]);
            return false;
        }
    }
    if (a.tokens.size() != b.tokens.size()) {
        std::printf("%s: token stream length %zu vs %zu\n", tag, a.tokens.size(), b.tokens.size());
        return false;
    }
    return true;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const char * target = std::getenv("LLAMA_TEST_PSHARD_SPEC_TARGET");
    if (!target || !*target) {
        std::printf("note: LLAMA_TEST_PSHARD_SPEC_TARGET not set - pshard speculative test skipped\n");
        return 0;
    }

    spec_cfg c;
    c.target    = target;
    c.draft     = env_or("LLAMA_TEST_PSHARD_SPEC_DRAFT", "");
    c.mtp       = c.draft.empty();
    c.mva       = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_MVA", "4000"));
    c.n_ctx     = (uint32_t) atoi(env_or("LLAMA_TEST_PSHARD_SPEC_CTX", "2048"));
    c.n_predict = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_NPREDICT", "48"));
    if (c.mtp) {
        c.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        c.n_draft_max = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_NDRAFT", "2"));
    } else {
        c.types = common_speculative_types_from_gguf(c.draft);
        if (c.types.empty() || c.types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
            std::printf("FAIL: draft type not inferable from %s\n", c.draft.c_str());
            return 1;
        }
        c.n_draft_max = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_NDRAFT", "8"));
    }
    const std::string strats_s = env_or("LLAMA_TEST_PSHARD_SPEC_STRATS", "0,1,auto");
    std::vector<std::string> strats;
    for (size_t i = 0, j; i < strats_s.size(); i = j + 1) {
        j = strats_s.find(',', i);
        if (j == std::string::npos) { j = strats_s.size(); }
        if (j > i) { strats.push_back(strats_s.substr(i, j - i)); }
    }
    const bool  run_stock    = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_STOCK", "1")) != 0;
    const bool  stock_strict = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_STOCK_STRICT", "0")) != 0;
    const bool  strict       = atoi(env_or("LLAMA_TEST_PSHARD_SPEC_STRICT", "0")) != 0;
    // cross-strategy logit tolerance; 0 (default) = report only. On MoE targets a placement-level
    // rounding difference can flip an expert-routing near-tie and move logits by O(1), so a
    // universal tolerance does not exist - calibrate per model with the printed matrix.
    const float tol          = (float) atof(env_or("LLAMA_TEST_PSHARD_SPEC_TOL", "0"));

    std::printf("pshard spec test: target=%s draft=%s mva=%d ctx=%u n_predict=%d n_draft=%d strategies=%s\n",
                c.target.c_str(), c.mtp ? "(MTP head)" : c.draft.c_str(), c.mva, c.n_ctx, c.n_predict,
                c.n_draft_max, strats_s.c_str());

    common_init();
    llama_backend_init();

    std::vector<spec_result> results;
    for (const auto & s : strats) {
        set_env("PSHARD_STRATEGY", s == "auto" ? nullptr : s.c_str());
        std::printf("== strategy %s: planning\n", s.c_str());
        plan_pshard(c);
        std::printf("== strategy %s: run 1\n", s.c_str());
        spec_result r = run_spec(c, true);
        print_result(s.c_str(), r);
        CHECK(r.ok);
        CHECK(r.pshard);          // a silent stock fallback is a failure, not a pass
        CHECK(r.n_drafted > 0);
        CHECK(r.n_accept > 0);    // the draft must actually accelerate something

        // DETERMINISM: the same plan, twice - bitwise over the whole speculative loop
        std::printf("== strategy %s: run 2 (determinism)\n", s.c_str());
        spec_result r2 = run_spec(c, true);
        print_result((s + "/2").c_str(), r2);
        CHECK(r2.ok);
        CHECK(tokens_equal((s + " run1 vs run2").c_str(), r, r2));
        CHECK(r2.n_accept == r.n_accept);
        CHECK(r2.n_drafted == r.n_drafted);
        CHECK(r2.logits_hash == r.logits_hash);  // every verify step, bitwise

        // PARITY across strategies: the first verify step's position-0 logits see the identical
        // input in every mode; kernel placement may round differently but not disagree
        if (!results.empty() && r.ok) {
            const spec_result & ref = results[0];
            const float d = max_abs_delta(ref.first_logits, r.first_logits);
            std::printf("%s vs %s: first verify step max |logit delta| = %.5f (top-2 margin %.4f / %.4f)%s\n",
                        s.c_str(), strats[0].c_str(), d, top2_margin(ref.first_logits), top2_margin(r.first_logits),
                        tol > 0.0f ? "" : " [report only]");
            if (tol > 0.0f) { CHECK(d <= tol); }
            const bool same = tokens_equal((s + " vs " + strats[0]).c_str(), ref, r);
            if (same && !c.mtp && r.logits_hash != ref.logits_hash) {
                std::printf("%s: tokens agree with %s but verify-logits hash differs (placement rounding)\n",
                            s.c_str(), strats[0].c_str());
            }
            if (strict) {
                CHECK(same);
                if (!c.mtp) { CHECK(r.logits_hash == ref.logits_hash); }
            } else if (!same) {
                std::printf("%s: token stream differs from %s (greedy near-tie across placements; LLAMA_TEST_PSHARD_SPEC_STRICT=1 enforces)\n",
                            s.c_str(), strats[0].c_str());
            }
        }
        results.push_back(r);
    }
    set_env("PSHARD_STRATEGY", nullptr);

    spec_result stock_res;
    if (run_stock && !results.empty()) {
        std::printf("== stock reference at -fitb %d\n", c.mva);
        spec_result rs = run_spec(c, false);
        print_result("stock", rs);
        CHECK(rs.ok);
        CHECK(!rs.pshard);
        if (rs.ok) {
            const float d = max_abs_delta(results[0].first_logits, rs.first_logits);
            std::printf("stock vs %s: first verify step max |logit delta| = %.5f\n", strats[0].c_str(), d);
        }
        stock_res = rs;
        const bool same = tokens_equal("stock", results[0], rs);
        if (stock_strict) {
            CHECK(same);
        } else if (!same) {
            std::printf("stock: token stream differs from pshard (near-tie at greedy is legitimate across placements; set LLAMA_TEST_PSHARD_SPEC_STOCK_STRICT=1 to enforce)\n");
        }
    }

    // pairwise first-verify-step logit deltas over every mode (placement numerics at a glance)
    {
        std::vector<std::pair<std::string, const std::vector<float> *>> modes;
        for (size_t i = 0; i < results.size(); ++i) { modes.emplace_back(strats[i], &results[i].first_logits); }
        if (run_stock && stock_res.ok) { modes.emplace_back("stock", &stock_res.first_logits); }
        std::printf("first-verify-step max |logit delta| matrix:\n%-8s", "");
        for (const auto & m : modes) { std::printf(" %8s", m.first.c_str()); }
        std::printf("\n");
        for (const auto & a : modes) {
            std::printf("%-8s", a.first.c_str());
            for (const auto & b : modes) { std::printf(" %8.4f", max_abs_delta(*a.second, *b.second)); }
            std::printf("   top-2 margin %.4f\n", top2_margin(*a.second));
        }
    }

    llama_backend_free();

    if (g_failures == 0) {
        std::printf("pshard spec test: all checks passed\n");
        return 0;
    }
    std::printf("pshard spec test: %d check(s) FAILED\n", g_failures);
    return 1;
}
