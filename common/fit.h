#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// Pipelined-sharding (pshard) variant of common_fit_params.
//
// Loads the plan registry cached next to the model as
// <model>.tensor_overrides.pshard_registry (written by llama-fit-params --pshard)
// and fills tensor_buft_overrides for model loading. If no usable plan applies,
// clears mparams->pshard and leaves baseline loading in place. This is the
// RUNTIME path - it never plans, so startup stays fast.
//
// mparams->pshard_registry must be set by the caller (common_pshard_registry_create).
//
// Create/free the tier plan registry. The caller owns the pointer and assigns it
// to mparams->pshard_registry before calling common_fit_params_pshard.
struct llama_pshard_plan_registry * common_pshard_registry_create(uint32_t n_tier_max, uint32_t n_seq_max);
void                                common_pshard_registry_free  (struct llama_pshard_plan_registry * registry);

void common_fit_params_pshard(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
   llama_model_tensor_buft_override * tensor_buft_overrides,
                             size_t   max_vram_mb,    // 0 = use actual free VRAM minus fit_target_mb
                             size_t   fit_target_mb); // ignored when max_vram_mb > 0

// PLANNER path (llama-fit-params --pshard).
//
// Probes VRAM for each strategy/tier combination, picks the best plan per tier and
// writes/updates <model>.tensor_overrides.pshard_registry, then fills
// tensor_buft_overrides. Loads any already-cached tiers first and only plans what is
// missing. If everything already fits in VRAM it clears mparams->pshard and falls back
// to baseline loading.
//
// Supplies the baseline params-fitting engine (common_fit_params) to the planner, which
// lives in libllama and cannot call into common/ directly.
void common_pshard_plan(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
   llama_model_tensor_buft_override * tensor_buft_overrides,
                             size_t   max_vram_mb,
                             size_t   fit_target_mb);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
