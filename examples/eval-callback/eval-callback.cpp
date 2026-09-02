#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    // decode in batch-sized chunks: a prompt longer than n_batch is otherwise rejected, and a
    // second chunk is the only way to inspect a KV-warm graph (behavior-preserving otherwise)
    // LLAMA_EVAL_CB_CHUNK=N decodes N tokens per call regardless of the context batch size
    // (e.g. to inspect a 16-token batch reading a warm KV cache under a larger runtime batch)
    size_t n_batch = std::max<size_t>(1, llama_n_batch(ctx));
    if (const char * e = std::getenv("LLAMA_EVAL_CB_CHUNK"); e && *e && std::atoi(e) > 0) {
        n_batch = std::min<size_t>(n_batch, (size_t) std::atoi(e));
    }
    // LLAMA_EVAL_CB_ALL_LOGITS=1 requests logits for every token of a chunk (multi-output batch,
    // the shape perplexity and speculative verify use) instead of the last token only
    const bool all_logits = std::getenv("LLAMA_EVAL_CB_ALL_LOGITS") && std::atoi(std::getenv("LLAMA_EVAL_CB_ALL_LOGITS")) > 0;
    llama_batch batch = llama_batch_init((int32_t) n_batch, 0, 1);
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const size_t n = std::min(n_batch, tokens.size() - i);
        common_batch_clear(batch);
        for (size_t k = 0; k < n; ++k) {
            common_batch_add(batch, tokens[i + k], (llama_pos) (i + k), { 0 }, all_logits || (k + 1 == n));
        }
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval (chunk at %zu)\n", __func__, i);
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
