#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct llama_model_params;

// The model's routing workload: the Zipf exponent alpha of the router's expert choices, the input of the
// planner's hit-rate model h(s). The expert pool histograms every cache-mode route and folds each run into
// <model>.pshard_workload at exit (runs accumulate); a model without a file is calibrated at plan time
// (llama_pshard_workload_calibrate below), and the first real run replaces that calibration.
struct llama_pshard_workload {
    double      zipf_alpha = -1.0;   // -1 = unknown
    double      fit_rms    = 0.0;    // RMS of h_obs(s) - h_zipf(s; alpha) over s = 1..n_expert
    uint64_t    samples    = 0;      // routes (token x expert) in `counts`
    uint32_t    n_layers   = 0;      // layers with counts
    uint32_t    n_expert   = 0;
    std::string source;              // "runtime" | "calibration"
    // the histogram store: per MoE layer (its index il) the routes seen per expert
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> counts;

    static std::string path_for(const std::string & path_model);
    bool load(const std::string & path);        // header + counts; refits alpha from the counts
    bool save(const std::string & path) const;  // header + counts

    // replace the store with these counts (a calibration, or the first real run) and refit
    bool set_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c, uint32_t n_expert, const char * src);
    // add these counts to the store (layers matched by il, new layers appended) and refit
    bool merge_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c);
    bool refit();

    // fit alpha from per-layer expert use counts. h_obs(s) is a split-half top-s mass: each expert's routes are
    // split into two folds (deterministic binomial split), the experts are ranked on one fold and the cumulative
    // share of the s top-ranked is taken on the other, both ways, averaged over layers - ranking and scoring on
    // the same sample inflates the top-s mass when routes per expert are few, which reads as a too-skewed alpha.
    // alpha minimises sum_s (h_obs(s) - h_zipf(s; alpha))^2 over s = 1..n_expert on a 0.005 grid in [0, 3].
    static bool fit(const std::vector<std::vector<uint64_t>> & counts, uint32_t n_expert, llama_pshard_workload & out);

    // Zipf mass of the s most popular of n_expert experts at exponent alpha (the planner's h(s))
    static double zipf_h(double alpha, uint32_t s, uint32_t n_expert);
};

// plan-time bootstrap: load the model CPU-only (the caller's load parameters minus placement and pshard),
// sample n_tokens tokens from BOS at temperature 1 with a fixed seed (an end-of-generation token restarts
// from BOS), histogram the routers' top-k ids ("ffn_moe_topk-<il>" nodes) through the eval callback, fit.
// The caller saves. false + err on failure.
bool llama_pshard_workload_calibrate(const std::string & path_model, const llama_model_params * base_mparams, int n_threads,
                                     uint32_t n_tokens, uint32_t seed, uint32_t n_expert,
                                     llama_pshard_workload & out, std::string & err);
