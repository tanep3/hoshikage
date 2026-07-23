#define HOSHIKAGE_ADAPTER_BUILD

#include "hoshikage_llama_adapter.h"

#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

static thread_local std::string hoshikage_last_error;

static void set_last_error(const std::string & message) {
    hoshikage_last_error = message;
}

static void clear_last_error() {
    hoshikage_last_error.clear();
}

static void trace(const char * message) {
    std::fprintf(stderr, "hoshikage-adapter: %s\n", message);
    std::fflush(stderr);
}

struct hoshikage_speculation_context {
    const llama_model * target_model;
    llama_context * target_context;
    llama_model * draft_model = nullptr;
    llama_context * draft_context = nullptr;
    hoshikage_speculation_config config;
    common_params common;
    common_speculative * speculator = nullptr;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
};

uint32_t hoshikage_adapter_abi_version(void) {
    return 1;
}

const char * hoshikage_adapter_last_error(void) {
    return hoshikage_last_error.c_str();
}

bool hoshikage_speculation_supports(int32_t mode) {
    switch (mode) {
    case HOSHIKAGE_SPECULATION_MODE_MTP:
        return true;
    case HOSHIKAGE_SPECULATION_MODE_DRAFT_MODEL:
    default:
        return false;
    }
}

hoshikage_speculation_context * hoshikage_speculation_init(
    const llama_model * target_model,
    llama_context * target_context,
    const hoshikage_speculation_config * config) {
    if (target_model == nullptr || target_context == nullptr || config == nullptr) {
        set_last_error("target model, target context, or config is null");
        return nullptr;
    }

    if (config->mode != HOSHIKAGE_SPECULATION_MODE_MTP) {
        set_last_error("only MTP mode is implemented in this adapter");
        return nullptr;
    }

    try {
        trace("init begin");
        clear_last_error();
        auto context = std::make_unique<hoshikage_speculation_context>();
        context->target_model = target_model;
        context->target_context = target_context;
        context->config = *config;

        context->common.n_ctx = static_cast<int32_t>(config->n_ctx);
        context->common.n_batch = static_cast<int32_t>(config->n_ctx);
        context->common.n_ubatch = static_cast<int32_t>(config->n_ctx);
        context->common.n_gpu_layers = config->n_gpu_layers_draft;
        context->common.model.path = "";
        context->common.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        context->common.speculative.draft.n_max = std::max<int32_t>(1, config->n_draft_max);
        context->common.speculative.draft.n_min = 0;
        context->common.speculative.draft.p_min = 0.0f;
        context->common.speculative.draft.n_gpu_layers = config->n_gpu_layers_draft;
        context->common.speculative.draft.ctx_tgt = target_context;

        const llama_model * draft_model_for_context = target_model;

        if (config->draft_model_path != nullptr && config->draft_model_path[0] != '\0') {
            trace("init load draft model begin");
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = config->n_gpu_layers_draft;
            context->draft_model = llama_model_load_from_file(config->draft_model_path, mparams);
            if (context->draft_model == nullptr) {
                set_last_error("draft model load returned null");
                return nullptr;
            }
            trace("init load draft model ok");
            draft_model_for_context = context->draft_model;
        }

        trace("init draft context begin");
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = config->n_ctx;
        cparams.n_batch = config->n_ctx;
        cparams.n_ubatch = config->n_ctx;
        cparams.n_rs_seq = static_cast<uint32_t>(std::max<int32_t>(1, config->n_draft_max));
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.ctx_other = target_context;

        context->draft_context = llama_init_from_model(
            const_cast<llama_model *>(draft_model_for_context),
            cparams);
        if (context->draft_context == nullptr) {
            set_last_error("MTP draft context init returned null");
            return nullptr;
        }
        trace("init draft context ok");

        context->common.speculative.draft.ctx_dft = context->draft_context;
        trace("init common speculative begin");
        context->speculator = common_speculative_init(context->common.speculative, 1);
        if (context->speculator == nullptr) {
            set_last_error("common speculative init returned null");
            return nullptr;
        }
        trace("init common speculative ok");

        return context.release();
    } catch (const std::exception &) {
        set_last_error("adapter init threw std::exception");
        return nullptr;
    } catch (...) {
        set_last_error("adapter init threw unknown exception");
        return nullptr;
    }
}

void hoshikage_speculation_free(hoshikage_speculation_context * context) {
    if (context != nullptr && context->speculator != nullptr) {
        common_speculative_free(context->speculator);
        context->speculator = nullptr;
    }
    if (context != nullptr && context->draft_context != nullptr) {
        llama_free(context->draft_context);
        context->draft_context = nullptr;
    }
    if (context != nullptr && context->draft_model != nullptr) {
        llama_model_free(context->draft_model);
        context->draft_model = nullptr;
    }
    delete context;
}

int32_t hoshikage_speculation_draft(
    hoshikage_speculation_context * context,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t n_prompt_tokens,
    llama_token * out_tokens,
    size_t out_capacity,
    size_t * out_n_tokens) {
    (void) seq_id;

    if (out_n_tokens != nullptr) {
        *out_n_tokens = 0;
    }

    if (context == nullptr || context->speculator == nullptr || out_n_tokens == nullptr) {
        set_last_error("draft called without initialized context");
        return -1;
    }
    if (out_capacity > 0 && out_tokens == nullptr) {
        set_last_error("draft output buffer is null");
        return -1;
    }

    try {
        trace("draft begin");
        clear_last_error();
        context->prompt.assign(prompt_tokens, prompt_tokens + n_prompt_tokens);
        context->draft.clear();

        common_speculative_begin(context->speculator, seq_id, context->prompt);
        common_speculative_get_draft_params(context->speculator, seq_id) = {
            /* .drafting   = */ true,
            /* .n_max      = */ -1,
            /* .n_past     = */ n_past,
            /* .id_last    = */ id_last,
            /* .prompt     = */ &context->prompt,
            /* .result     = */ &context->draft,
        };
        common_speculative_draft(context->speculator);
        trace("draft common ok");

        const size_t n = std::min(context->draft.size(), out_capacity);
        if (n > 0) {
            std::copy_n(context->draft.data(), n, out_tokens);
        }
        *out_n_tokens = n;
        return 0;
    } catch (const std::exception &) {
        set_last_error("draft threw std::exception");
        return -1;
    } catch (...) {
        set_last_error("draft threw unknown exception");
        return -1;
    }
}

bool hoshikage_speculation_process(
    hoshikage_speculation_context * context,
    const llama_batch * batch) {
    if (context == nullptr || context->speculator == nullptr || batch == nullptr) {
        set_last_error("process called without initialized context or batch");
        return false;
    }

    try {
        trace("process begin");
        clear_last_error();
        const bool result = common_speculative_process(context->speculator, *batch);
        trace(result ? "process ok" : "process failed");
        return result;
    } catch (const std::exception &) {
        set_last_error("process threw std::exception");
        return false;
    } catch (...) {
        set_last_error("process threw unknown exception");
        return false;
    }
}

void hoshikage_speculation_accept(
    hoshikage_speculation_context * context,
    llama_seq_id seq_id,
    uint16_t n_accepted) {
    if (context == nullptr || context->speculator == nullptr) {
        return;
    }

    trace("accept begin");
    common_speculative_accept(context->speculator, seq_id, n_accepted);
    trace("accept ok");
}
