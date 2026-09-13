#include "llama-pshard-workload.h"

#include "llama.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>

std::string llama_pshard_workload::path_for(const std::string & path_model) {
    return path_model + ".pshard_workload";
}

double llama_pshard_workload::zipf_h(double alpha, uint32_t s, uint32_t n_expert) {
    if (n_expert == 0) {
        return 0.0;
    }
    if (s >= n_expert) {
        return 1.0;
    }
    double num = 0.0, den = 0.0;
    for (uint32_t i = 1; i <= n_expert; i++) {
        const double w = std::pow((double) i, -alpha);
        den += w;
        if (i <= s) {
            num += w;
        }
    }
    return den > 0.0 ? num / den : 0.0;
}

bool llama_pshard_workload::fit(const std::vector<std::vector<uint64_t>> & counts, uint32_t n_expert, llama_pshard_workload & out) {
    if (n_expert == 0) {
        return false;
    }
    std::vector<double> h_obs(n_expert + 1, 0.0);
    uint32_t n_layers = 0, n_folds = 0;
    uint64_t samples  = 0;
    std::vector<uint64_t> a(n_expert), b(n_expert);
    std::vector<uint32_t> idx(n_expert);
    for (size_t li = 0; li < counts.size(); li++) {
        const auto & c = counts[li];
        if (c.size() != n_expert) {
            continue;
        }
        uint64_t total = 0;
        for (uint64_t v : c) {
            total += v;
        }
        if (total == 0) {
            continue;
        }
        // split-half: every route of expert i lands in fold A or B with equal probability (deterministic seed)
        std::mt19937_64 rng(0x5EEDF17ull + 1315423911ull * (uint64_t) (li + 1));
        uint64_t ta = 0, tb = 0;
        for (uint32_t i = 0; i < n_expert; i++) {
            if (c[i] > 0) {
                std::binomial_distribution<unsigned long long> bin(c[i], 0.5);
                a[i] = bin(rng);
            } else {
                a[i] = 0;
            }
            b[i] = c[i] - a[i];
            ta += a[i];
            tb += b[i];
        }
        if (ta == 0 || tb == 0) {
            continue;
        }
        // rank on one fold, measure the top-s cumulative share on the other
        auto fold = [&](const std::vector<uint64_t> & rank_on, const std::vector<uint64_t> & measure, uint64_t total_measure) {
            std::iota(idx.begin(), idx.end(), 0u);
            std::stable_sort(idx.begin(), idx.end(), [&](uint32_t x, uint32_t y) { return rank_on[x] > rank_on[y]; });
            double cum = 0.0;
            for (uint32_t s = 1; s <= n_expert; s++) {
                cum += (double) measure[idx[s - 1]];
                h_obs[s] += cum / (double) total_measure;
            }
        };
        fold(a, b, tb);
        fold(b, a, ta);
        n_folds += 2;
        n_layers++;
        samples += total;
    }
    if (n_folds == 0) {
        return false;
    }
    for (uint32_t s = 1; s <= n_expert; s++) {
        h_obs[s] /= (double) n_folds;
    }
    double best_a = 0.0, best_e = 1e300;
    std::vector<double> w(n_expert + 1, 0.0);
    for (int step = 0; step <= 600; step++) {
        const double alpha = 0.005 * step;
        double den = 0.0;
        for (uint32_t i = 1; i <= n_expert; i++) {
            w[i] = std::pow((double) i, -alpha);
            den += w[i];
        }
        double cum = 0.0, err = 0.0;
        for (uint32_t s = 1; s <= n_expert; s++) {
            cum += w[s];
            const double d = h_obs[s] - cum / den;
            err += d * d;
        }
        if (err < best_e) {
            best_e = err;
            best_a = alpha;
        }
    }
    out.zipf_alpha = best_a;
    out.fit_rms    = std::sqrt(best_e / (double) n_expert);
    out.samples    = samples;
    out.n_layers   = n_layers;
    out.n_expert   = n_expert;
    return true;
}

bool llama_pshard_workload::refit() {
    std::vector<std::vector<uint64_t>> v;
    v.reserve(counts.size());
    for (const auto & p : counts) {
        v.push_back(p.second);
    }
    llama_pshard_workload tmp;
    if (!fit(v, n_expert, tmp)) {
        return false;
    }
    zipf_alpha = tmp.zipf_alpha;
    fit_rms    = tmp.fit_rms;
    samples    = tmp.samples;
    n_layers   = tmp.n_layers;
    return true;
}

bool llama_pshard_workload::set_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c, uint32_t ne, const char * src) {
    counts   = c;
    n_expert = ne;
    source   = src ? src : "unknown";
    return refit();
}

bool llama_pshard_workload::merge_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c) {
    for (const auto & p : c) {
        if (p.second.size() != n_expert) {
            continue;
        }
        bool found = false;
        for (auto & q : counts) {
            if (q.first == p.first) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    q.second[i] += p.second[i];
                }
                found = true;
                break;
            }
        }
        if (!found) {
            counts.push_back(p);
        }
    }
    return refit();
}

bool llama_pshard_workload::save(const std::string & path) const {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return false;
    }
    fprintf(f, "# pshard routing workload: Zipf exponent of the router's expert popularity, fitted (split-half) from measured routes;\n");
    fprintf(f, "# the per-layer route histograms follow (layer <il>: one count per expert); real runs accumulate into them\n");
    fprintf(f, "zipf_alpha=%.3f fit_rms=%.4f samples=%llu layers=%u n_expert=%u source=%s\n",
        zipf_alpha, fit_rms, (unsigned long long) samples, n_layers, n_expert, source.empty() ? "unknown" : source.c_str());
    for (const auto & p : counts) {
        fprintf(f, "layer %u:", p.first);
        for (uint64_t v : p.second) {
            fprintf(f, " %llu", (unsigned long long) v);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

bool llama_pshard_workload::load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    std::vector<char> line(64 * 1024);
    bool ok = false;
    counts.clear();
    while (fgets(line.data(), (int) line.size(), f)) {
        double a = 0.0, r = 0.0;
        unsigned long long smp = 0;
        unsigned nl = 0, ne = 0;
        char src[32] = { 0 };
        if (sscanf(line.data(), "zipf_alpha=%lf fit_rms=%lf samples=%llu layers=%u n_expert=%u source=%31s", &a, &r, &smp, &nl, &ne, src) == 6) {
            zipf_alpha = a;
            fit_rms    = r;
            samples    = smp;
            n_layers   = nl;
            n_expert   = ne;
            source     = src;
            ok = true;
            continue;
        }
        unsigned il = 0;
        int consumed = 0;
        if (sscanf(line.data(), "layer %u:%n", &il, &consumed) == 1 && consumed > 0 && n_expert > 0) {
            std::vector<uint64_t> row;
            row.reserve(n_expert);
            const char * p = line.data() + consumed;
            while (row.size() < n_expert) {
                char * end = nullptr;
                const unsigned long long v = strtoull(p, &end, 10);
                if (end == p) {
                    break;
                }
                row.push_back((uint64_t) v);
                p = end;
            }
            if (row.size() == n_expert) {
                counts.push_back({ il, std::move(row) });
            }
        }
    }
    fclose(f);
    if (!ok) {
        return false;
    }
    if (!counts.empty()) {
        refit();   // the file's alpha may come from an older estimator; the counts are the record
    }
    return zipf_alpha >= 0.0;
}

namespace {
struct calib_state {
    std::vector<std::vector<uint64_t>> counts;   // by layer index
    std::vector<int32_t>               row;
    uint32_t                           n_expert = 0;
};

// eval callback: observe every router top-k node ("ffn_moe_topk-<il>", I32 [n_expert_used, n_tokens], a view
// whose rows are contiguous) and count its ids. The prediction nodes ("ffn_moe_pred_topk-<il>") do not match.
bool calib_cb(struct ggml_tensor * t, bool ask, void * ud) {
    calib_state * st = (calib_state *) ud;
    const char * name = ggml_get_name(t);
    const bool is_topk = strncmp(name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk || t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return true;
    }
    const int il = atoi(name + 13);
    if (il < 0) {
        return true;
    }
    if ((size_t) il >= st->counts.size()) {
        st->counts.resize((size_t) il + 1);
    }
    if (st->counts[il].empty()) {
        st->counts[il].assign(st->n_expert, 0);
    }
    const int64_t n0 = t->ne[0], n1 = t->ne[1];
    st->row.resize((size_t) n0);
    for (int64_t i1 = 0; i1 < n1; i1++) {
        ggml_backend_tensor_get(t, st->row.data(), (size_t) i1 * t->nb[1], (size_t) n0 * sizeof(int32_t));
        for (int32_t e : st->row) {
            if (e >= 0 && (uint32_t) e < st->n_expert) {
                st->counts[il][e]++;
            }
        }
    }
    return true;
}
} // namespace

bool llama_pshard_workload_calibrate(const std::string & path_model, const llama_model_params * base_mparams, int n_threads,
                                     uint32_t n_tokens, uint32_t seed, uint32_t n_expert,
                                     llama_pshard_workload & out, std::string & err) {
    if (n_expert == 0 || n_tokens == 0) {
        err = "nothing to calibrate";
        return false;
    }
    // the caller's load parameters (kv overrides, load mode) minus placement and pshard: CPU only, since routing
    // does not depend on the placement and the load then fits on any machine; the model's own sampled text
    // stands in for the workload until a real pool run replaces it
    llama_model_params mp = base_mparams ? *base_mparams : llama_model_default_params();
    mp.n_gpu_layers          = 0;
    mp.pshard                = false;
    mp.pshard_registry       = nullptr;
    mp.tensor_buft_overrides = nullptr;
    mp.progress_callback     = nullptr;
    llama_model * model = llama_model_load_from_file(path_model.c_str(), mp);
    if (model == nullptr) {
        err = "model load failed";
        return false;
    }
    calib_state st;
    st.n_expert = n_expert;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_tokens + 8;
    cp.n_batch           = 8;
    cp.n_ubatch          = 8;
    cp.n_threads         = n_threads > 0 ? n_threads : cp.n_threads;
    cp.n_threads_batch   = cp.n_threads;
    cp.cb_eval           = calib_cb;
    cp.cb_eval_user_data = &st;
    cp.pshard            = false;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (lctx == nullptr) {
        llama_model_free(model);
        err = "context init failed";
        return false;
    }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) {
        bos = 0;   // no BOS token: start from the first vocabulary entry
    }
    // segments of 32 tokens, the first from BOS, the others from a random vocabulary token: the continuation
    // of one start is one topic in one register and routes to few experts (q35: 4 experts took every token
    // in the first layers), diverse starts spread the routing the way a real prompt mix does
    std::mt19937   seg_rng(seed);
    const int32_t  n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t seg_len = 32;
    llama_token tok = bos;
    bool ok = true;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (i > 0 && i % seg_len == 0 && n_vocab > 0) {
            tok = (llama_token) (seg_rng() % (uint32_t) n_vocab);
        }
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(lctx, batch) != 0) {
            err = "decode failed";
            ok = false;
            break;
        }
        tok = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            tok = bos;   // the text ended: start another rather than routing a degenerate continuation
        }
    }
    llama_sampler_free(smpl);
    llama_free(lctx);
    llama_model_free(model);
    if (!ok) {
        return false;
    }
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> counts;
    for (size_t il = 0; il < st.counts.size(); il++) {
        if (!st.counts[il].empty()) {
            counts.push_back({ (uint32_t) il, st.counts[il] });
        }
    }
    if (counts.empty() || !out.set_counts(counts, n_expert, "calibration")) {
        err = "no routing observed (no ffn_moe_topk nodes)";
        return false;
    }
    return true;
}
