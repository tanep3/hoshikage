#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "llama.h"

#ifdef _WIN32
#    ifdef HOSHIKAGE_ADAPTER_BUILD
#        define HOSHIKAGE_ADAPTER_API __declspec(dllexport)
#    else
#        define HOSHIKAGE_ADAPTER_API __declspec(dllimport)
#    endif
#else
#    define HOSHIKAGE_ADAPTER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum hoshikage_speculation_mode {
    HOSHIKAGE_SPECULATION_MODE_MTP = 1,
    HOSHIKAGE_SPECULATION_MODE_DRAFT_MODEL = 2,
};

struct hoshikage_speculation_context;

struct hoshikage_speculation_config {
    int32_t mode;
    int32_t n_draft_max;
    int32_t n_seq;
    uint32_t n_ctx;
    int32_t n_gpu_layers_draft;
    const char * draft_model_path;
};

HOSHIKAGE_ADAPTER_API uint32_t hoshikage_adapter_abi_version(void);
HOSHIKAGE_ADAPTER_API const char * hoshikage_adapter_last_error(void);
HOSHIKAGE_ADAPTER_API bool hoshikage_speculation_supports(int32_t mode);

HOSHIKAGE_ADAPTER_API hoshikage_speculation_context * hoshikage_speculation_init(
    const llama_model * target_model,
    llama_context * target_context,
    const hoshikage_speculation_config * config);

HOSHIKAGE_ADAPTER_API void hoshikage_speculation_free(hoshikage_speculation_context * context);

HOSHIKAGE_ADAPTER_API int32_t hoshikage_speculation_draft(
    hoshikage_speculation_context * context,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t n_prompt_tokens,
    llama_token * out_tokens,
    size_t out_capacity,
    size_t * out_n_tokens);

HOSHIKAGE_ADAPTER_API bool hoshikage_speculation_process(
    hoshikage_speculation_context * context,
    const llama_batch * batch);

HOSHIKAGE_ADAPTER_API void hoshikage_speculation_accept(
    hoshikage_speculation_context * context,
    llama_seq_id seq_id,
    uint16_t n_accepted);

#ifdef __cplusplus
}
#endif
