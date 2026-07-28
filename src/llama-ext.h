#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

//
// pipeline sharding (pshard)
//

struct llama_pshard_plan_registry;

// Baseline params-fitting engine injected into the pshard planner.
//
// The engine itself lives in common/fit.cpp (common_fit_params), and common
// links against llama - so libllama cannot call it directly. The caller passes
// it in instead. Implementations must throw on failure.
typedef void (*llama_pshard_fit_fn)(
                                   const char   * path_model,
                    struct llama_model_params   * mparams,
                    struct llama_context_params * cparams,
                                          float * tensor_split,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
                                         size_t * margins_s,
                                       uint32_t   n_ctx_min,
                            enum ggml_log_level   log_level,
                                           void * user_data);

// RUNTIME path. Loads the cached pshard plan registry written by the planner
// (llama-fit-params --pshard) and populates tensor_buft_overrides for model
// loading. Never plans, so startup stays fast. If no usable plan is found,
// clears mparams->pshard and returns, leaving baseline loading in place.
// Prefer common_fit_params_pshard() in common/fit.h.
LLAMA_API void llama_params_fit_pshard_inference(
                                   const char   * path_model,
                    struct llama_model_params   * mparams,
                    struct llama_context_params * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
                                         size_t   max_vram_mb,    // 0 = use actual free VRAM minus fit_target_mb
                                         size_t   fit_target_mb); // ignored when max_vram_mb > 0

// PLANNER path. Probes VRAM usage for each strategy/tier combination, selects the best
// plan (by TPS if benchmark data is available, else by VRAM fit), writes/updates
// <model_path>.tensor_overrides.pshard_registry and populates tensor_buft_overrides.
// If all layers fit in VRAM, sets mparams->pshard=false and returns (baseline loading).
// Prefer common_pshard_plan() in common/fit.h, which supplies fit_fn for you.
LLAMA_API void llama_params_fit_pshard_planning(
                                   const char   * path_model,
                    struct llama_model_params   * mparams,
                    struct llama_context_params * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
                                         size_t   max_vram_mb,   // 0 = use actual free VRAM minus fit_target_mb
                                         size_t   fit_target_mb, // ignored when max_vram_mb > 0
                            llama_pshard_fit_fn   fit_fn,
                                           void * fit_fn_ud);

// Create/free a tier plan registry. Caller owns the pointer and passes it via
// mparams->pshard_registry before calling either entry point above.
// n_tier_max: largest batch size to probe (determines tier range, typically max(n_batch, 16384)).
// n_seq_max: for speculative decoding tiers.
LLAMA_API struct llama_pshard_plan_registry * llama_pshard_registry_create(uint32_t n_tier_max, uint32_t n_seq_max);
LLAMA_API void                                llama_pshard_registry_free  (struct llama_pshard_plan_registry * registry);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
