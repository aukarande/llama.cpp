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
// Every pshard run also adds the distinct experts of its multi-row MoE passes whose expert copies the scheduler
// sliced by used ids to <model>.pshard_workload_rows at exit; the predictor prices those copies from them.
struct llama_pshard_workload {
    double      zipf_alpha = -1.0;   // -1 = unknown
    double      fit_rms    = 0.0;    // RMS of h_obs(s) - h_zipf(s; alpha) over s = 1..n_expert
    uint64_t    samples    = 0;      // routes (token x expert) in `counts`
    uint32_t    n_layers   = 0;      // layers with counts
    uint32_t    n_expert   = 0;
    std::string source;              // "runtime" | "calibration"
    // the histogram store: per MoE layer (its index il) the routes seen per expert
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> counts;

    // multi-row MoE layer passes, bucketed by floor(log2(rows)): passes, token rows and distinct experts summed
    struct distinct_bucket {
        uint64_t passes   = 0;
        uint64_t rows     = 0;
        uint64_t distinct = 0;
    };
    std::vector<distinct_bucket> rows_distinct;

    // distinct share of the experts per pass at share_steps points per octave of rows from 1 row (build_share_table)
    static constexpr int share_steps = 4;
    std::vector<double>  share_table;

    static std::string path_for(const std::string & path_model);
    bool load(const std::string & path);        // header + counts; refits alpha from the counts
    bool save(const std::string & path) const;  // header + counts

    // replace the store with these counts (a calibration, or the first real run) and refit
    bool set_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c, uint32_t n_expert, const char * src);
    // add these counts to the store (layers matched by il, new layers appended) and refit
    bool merge_counts(const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> & c);
    bool refit();

    // rows_distinct lives in its own file: builds that predate it rewrite <model>.pshard_workload without it
    static std::string distinct_path_for(const std::string & path_model);
    bool load_distinct(const std::string & path, uint32_t n_expert);         // false: missing, unreadable or another expert count
    bool save_distinct(const std::string & path, uint32_t n_expert) const;

    void     add_distinct(uint64_t rows, uint64_t n_distinct);   // one pass; rows < 2 are not recorded
    void     merge_distinct(const std::vector<distinct_bucket> & d);
    uint64_t distinct_passes() const;

    // the share table from 1 to max_rows rows. The model is the independent-draw expectation over the route
    // histogram, sum_i 1 - (1 - q_i)^rows (q_i = routes of expert i per counted token, averaged over layers), or
    // uniform routing (q_i = n_expert_used/n_expert) without a histogram or when the histogram cannot reach a
    // measured share. The model runs at scaled rows: exact at one row (the top-k share), through every measured
    // bucket, the scale linear in log2(rows) between them and held past the last. No table with neither
    void   build_share_table(uint32_t n_expert, uint32_t n_expert_used, uint32_t max_rows);
    double distinct_share(double rows) const;   // -1 = no table

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
