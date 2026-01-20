use crate::config::Config;
use crate::error::Result;
use crate::ffi::*;
use crate::inference::loader::DynamicLibraryLoader;
use libloading::Symbol;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Once;

pub type LlamaLoadModelFromFile = unsafe extern "C" fn(
    path_model: *const ::std::os::raw::c_char,
    params: llama_model_params,
) -> *mut llama_model;

pub type LlamaInitFromModel = unsafe extern "C" fn(
    model: *mut llama_model,
    params: llama_context_params,
) -> *mut llama_context;

pub type LlamaModelDefaultParams = unsafe extern "C" fn() -> llama_model_params;
pub type LlamaContextDefaultParams = unsafe extern "C" fn() -> llama_context_params;

pub type LlamaDecode = unsafe extern "C" fn(ctx: *mut llama_context, batch: llama_batch) -> i32;
pub type LlamaEncode = unsafe extern "C" fn(ctx: *mut llama_context, batch: llama_batch) -> i32;

pub type LlamaTokenize = unsafe extern "C" fn(
    vocab: *const llama_vocab,
    text: *const ::std::os::raw::c_char,
    text_len: i32,
    tokens: *mut llama_token,
    n_max_tokens: i32,
    add_special: bool,
    parse_special: bool,
) -> i32;

pub type LlamaDetokenize = unsafe extern "C" fn(
    vocab: *const llama_vocab,
    tokens: *const llama_token,
    n_tokens: i32,
    buf: *mut ::std::os::raw::c_char,
    length: i32,
    remove_special: bool,
    unparse_special: bool,
) -> i32;

pub type LlamaTokenBos = unsafe extern "C" fn(model: *const llama_model) -> llama_token;
pub type LlamaTokenEos = unsafe extern "C" fn(model: *const llama_model) -> llama_token;

pub type LlamaFreeModel = unsafe extern "C" fn(model: *mut llama_model);
pub type LlamaFree = unsafe extern "C" fn(ctx: *mut llama_context);
pub type LlamaBackendInit = unsafe extern "C" fn();
pub type LlamaBackendFree = unsafe extern "C" fn();
pub type LlamaNumaInit = unsafe extern "C" fn(numa: ggml_numa_strategy);
pub type LlamaModelGetVocab = unsafe extern "C" fn(model: *const llama_model) -> *const llama_vocab;
pub type LlamaVocabNumTokens = unsafe extern "C" fn(vocab: *const llama_vocab) -> i32;
pub type LlamaVocabGetText = unsafe extern "C" fn(
    vocab: *const llama_vocab,
    token: llama_token,
) -> *const ::std::os::raw::c_char;
pub type LlamaTokenToPiece = unsafe extern "C" fn(
    vocab: *const llama_vocab,
    token: llama_token,
    buf: *mut ::std::os::raw::c_char,
    length: i32,
    lstrip: i32,
    special: bool,
) -> i32;
pub type LlamaGetLogitsIth = unsafe extern "C" fn(ctx: *mut llama_context, i: i32) -> *mut f32;
pub type LlamaGetLogits = unsafe extern "C" fn(ctx: *mut llama_context) -> *mut f32;
pub type LlamaKvCacheClear = unsafe extern "C" fn(ctx: *mut llama_context);
pub type LlamaModelIsDiffusion = unsafe extern "C" fn(model: *const llama_model) -> bool;
pub type LlamaModelHasEncoder = unsafe extern "C" fn(model: *const llama_model) -> bool;
pub type LlamaModelHasDecoder = unsafe extern "C" fn(model: *const llama_model) -> bool;
pub type LlamaModelMetaValStr = unsafe extern "C" fn(
    model: *const llama_model,
    name: *const c_char,
    buf: *mut c_char,
    length: usize,
) -> i32;
pub type LlamaChatApplyTemplate = unsafe extern "C" fn(
    tmpl: *const ::std::os::raw::c_char,
    chat: *const llama_chat_message,
    n_msg: usize,
    add_ass: bool,
    buf: *mut ::std::os::raw::c_char,
    length: i32,
) -> i32;
pub type LlamaBatchInit =
    unsafe extern "C" fn(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
pub type LlamaBatchFree = unsafe extern "C" fn(batch: llama_batch);
pub type LlamaVocabMask = unsafe extern "C" fn(vocab: *const llama_vocab) -> llama_token;
pub type LlamaSetCausalAttn = unsafe extern "C" fn(ctx: *mut llama_context, causal_attn: bool);
pub type LlamaSamplerChainDefaultParams = unsafe extern "C" fn() -> llama_sampler_chain_params;
pub type LlamaSamplerChainInit =
    unsafe extern "C" fn(params: llama_sampler_chain_params) -> *mut llama_sampler;
pub type LlamaSamplerChainAdd =
    unsafe extern "C" fn(chain: *mut llama_sampler, smpl: *mut llama_sampler);
pub type LlamaSamplerInitTopK = unsafe extern "C" fn(k: i32) -> *mut llama_sampler;
pub type LlamaSamplerInitTopP = unsafe extern "C" fn(p: f32, min_keep: usize) -> *mut llama_sampler;
pub type LlamaSamplerInitTemp = unsafe extern "C" fn(t: f32) -> *mut llama_sampler;
pub type LlamaSamplerInitDist = unsafe extern "C" fn(seed: u32) -> *mut llama_sampler;
pub type LlamaSamplerApply =
    unsafe extern "C" fn(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array);
pub type LlamaSamplerFree = unsafe extern "C" fn(smpl: *mut llama_sampler);

#[derive(Debug, Clone)]
pub struct InferenceParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: i32,
    pub stop_sequences: Vec<String>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub diffusion_steps: Option<i32>,
    pub diffusion_algorithm: Option<i32>,
    pub diffusion_schedule: Option<i32>,
    pub diffusion_cfg_scale: Option<f32>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 1024,
            stop_sequences: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            diffusion_steps: None,
            diffusion_algorithm: None,
            diffusion_schedule: None,
            diffusion_cfg_scale: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffusionAlgorithm {
    Origin = 0,
    Entropy = 1,
    Margin = 2,
    Random = 3,
    Confidence = 4,
}

impl DiffusionAlgorithm {
    pub fn from_i32(value: i32) -> Self {
        match value {
            0 => DiffusionAlgorithm::Origin,
            1 => DiffusionAlgorithm::Entropy,
            2 => DiffusionAlgorithm::Margin,
            3 => DiffusionAlgorithm::Random,
            4 => DiffusionAlgorithm::Confidence,
            _ => DiffusionAlgorithm::Confidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransferSchedule {
    TimestepBased = 0,
    BlockBased = 1,
}

impl TransferSchedule {
    pub fn from_i32(value: i32) -> Self {
        match value {
            0 => TransferSchedule::TimestepBased,
            1 => TransferSchedule::BlockBased,
            _ => TransferSchedule::TimestepBased,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionParams {
    pub steps: i32,
    pub algorithm: DiffusionAlgorithm,
    pub schedule: TransferSchedule,
    pub cfg_scale: f32,
    pub max_length: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
    pub mask_token_id: llama_token,
    pub shift_logits: bool,
    pub eps: f32,
    pub block_length: i32,
    pub alg_temp: f32,
    pub add_gumbel_noise: bool,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            steps: 50,
            algorithm: DiffusionAlgorithm::Confidence,
            schedule: TransferSchedule::TimestepBased,
            cfg_scale: 0.0,
            max_length: 1024,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            mask_token_id: LLAMA_TOKEN_NULL,
            shift_logits: true,
            eps: 0.0,
            block_length: 0,
            alg_temp: 0.0,
            add_gumbel_noise: false,
        }
    }
}

pub struct LlamaWrapper {
    _loader: DynamicLibraryLoader,
    model: Option<*mut llama_model>,
    context: Option<*mut llama_context>,
}

unsafe impl Send for LlamaWrapper {}

fn calculate_confidence(
    cur_p: &llama_token_data_array,
    algorithm: DiffusionAlgorithm,
    rng: &mut StdRng,
) -> f32 {
    if cur_p.data.is_null() || cur_p.size == 0 {
        return 0.0;
    }

    let data = unsafe { std::slice::from_raw_parts(cur_p.data, cur_p.size) };
    let selected = cur_p.selected;
    if selected < 0 || selected as usize >= data.len() {
        return 0.0;
    }

    match algorithm {
        DiffusionAlgorithm::Confidence | DiffusionAlgorithm::Origin => data[selected as usize].p,
        DiffusionAlgorithm::Entropy => {
            let mut entropy = 0.0f32;
            let epsilon = 1e-10f32;
            for item in data {
                entropy += item.p * (item.p + epsilon).ln();
            }
            -entropy
        }
        DiffusionAlgorithm::Margin => {
            if data.len() > 1 {
                data[0].p - data[1].p
            } else {
                data[0].p
            }
        }
        DiffusionAlgorithm::Random => rng.gen::<f32>(),
    }
}

fn calculate_transfer_count(
    step: i32,
    total_steps: i32,
    remaining_masked: i32,
    schedule: TransferSchedule,
    eps: f32,
    num_transfer_tokens: &[i32],
) -> i32 {
    match schedule {
        TransferSchedule::TimestepBased => {
            let t = 1.0f32 - (step as f32 / total_steps as f32) * (1.0f32 - eps);
            let s = 1.0f32 - ((step + 1) as f32 / total_steps as f32) * (1.0f32 - eps);
            let p_transfer = if step < total_steps - 1 {
                1.0f32 - s / t
            } else {
                1.0f32
            };
            (remaining_masked as f32 * p_transfer) as i32
        }
        TransferSchedule::BlockBased => {
            if !num_transfer_tokens.is_empty()
                && step >= 0
                && (step as usize) < num_transfer_tokens.len()
            {
                num_transfer_tokens[step as usize]
            } else {
                remaining_masked / (total_steps - step)
            }
        }
    }
}

fn get_num_transfer_tokens(mask_count: i32, steps: i32) -> Vec<i32> {
    if steps <= 0 {
        return Vec::new();
    }
    let base = mask_count / steps;
    let remainder = mask_count % steps;
    let mut result = Vec::with_capacity(steps as usize);
    for i in 0..steps {
        result.push(base + if i < remainder { 1 } else { 0 });
    }
    result
}

fn sample_token(logits: &[f32], temperature: f32, top_p: f32) -> Result<i32> {
    if logits.is_empty() {
        return Err(crate::error::HoshikageError::InferenceError(
            "Empty logits".to_string(),
        ));
    }
    if temperature <= 0.0 {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        return Ok(max_idx as i32);
    }

    let temp = temperature.max(1e-5);
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| ((i), ((logit - max_logit) / temp).exp()))
        .collect();

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let threshold = total * top_p.clamp(0.0, 1.0);
    let mut cumulative = 0.0f32;
    let mut candidates = Vec::new();
    for (idx, prob) in probs {
        cumulative += prob;
        candidates.push((idx, prob));
        if cumulative >= threshold {
            break;
        }
    }

    let norm: f32 = candidates.iter().map(|(_, p)| p).sum();
    if norm <= 0.0 {
        return Ok(candidates.first().map(|(i, _)| *i).unwrap_or(0) as i32);
    }

    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>() * norm;
    for (idx, prob) in candidates {
        if r <= prob {
            return Ok(idx as i32);
        }
        r -= prob;
    }

    Ok(0)
}

fn apply_penalties(
    logits: &mut [f32],
    counts: &HashMap<i32, u32>,
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    if (presence_penalty == 0.0 && frequency_penalty == 0.0) || logits.is_empty() {
        return;
    }

    for (token, count) in counts {
        if *count == 0 {
            continue;
        }
        let idx = *token as usize;
        if idx >= logits.len() {
            continue;
        }
        if presence_penalty != 0.0 {
            logits[idx] -= presence_penalty;
        }
        if frequency_penalty != 0.0 {
            logits[idx] -= frequency_penalty * (*count as f32);
        }
    }
}

fn apply_repeat_penalty(logits: &mut [f32], recent_tokens: &[i32], repeat_penalty: f32) {
    if repeat_penalty == 1.0 || recent_tokens.is_empty() || logits.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &token in recent_tokens {
        if token < 0 {
            continue;
        }
        if !seen.insert(token) {
            continue;
        }
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        let logit = logits[idx];
        logits[idx] = if logit < 0.0 {
            logit * repeat_penalty
        } else {
            logit / repeat_penalty
        };
    }
}

impl LlamaWrapper {
    pub fn new(lib_path: PathBuf) -> Result<Self> {
        let mut loader = DynamicLibraryLoader::new(lib_path);
        loader
            .load()
            .map_err(|e| crate::error::HoshikageError::LibraryLoadError(e.to_string()))?;

        static BACKEND_INIT: Once = Once::new();
        BACKEND_INIT.call_once(|| unsafe {
            if let Ok(llama_backend_init) =
                loader.get_symbol::<LlamaBackendInit>("llama_backend_init")
            {
                llama_backend_init();
            }
            #[cfg(target_os = "linux")]
            {
                if let Ok(llama_numa_init) = loader.get_symbol::<LlamaNumaInit>("llama_numa_init") {
                    llama_numa_init(0);
                }
            }
        });

        Ok(Self {
            _loader: loader,
            model: None,
            context: None,
        })
    }

    pub fn load_model(&mut self, model_path: &str, config: &Config) -> Result<()> {
        let model_path_c = CString::new(model_path)?;

        unsafe {
            let llama_load_model_from_file: Symbol<LlamaLoadModelFromFile> = self
                ._loader
                .get_symbol("llama_load_model_from_file")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_load_model_from_file: {}",
                        e
                    ))
                })?;

            let llama_model_default_params: Symbol<LlamaModelDefaultParams> = self
                ._loader
                .get_symbol("llama_model_default_params")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_default_params: {}",
                        e
                    ))
                })?;

            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = config.n_gpu_layers;

            let model = llama_load_model_from_file(model_path_c.as_ptr(), model_params);

            if model.is_null() {
                return Err(crate::error::HoshikageError::ModelLoadFailed(
                    "Model load returned null".to_string(),
                ));
            }

            self.model = Some(model);
        }

        let context = self.init_context(config)?;
        self.context = Some(context);

        Ok(())
    }

    pub fn format_chat_prompt(&self, messages: &[crate::api::ChatMessage]) -> Result<String> {
        let mut role_cstrings = Vec::with_capacity(messages.len());
        let mut content_cstrings = Vec::with_capacity(messages.len());
        let mut chat_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role = CString::new(msg.role.as_str())?;
            let content = CString::new(msg.content.as_str())?;
            role_cstrings.push(role);
            content_cstrings.push(content);
        }

        for i in 0..messages.len() {
            chat_messages.push(llama_chat_message {
                role: role_cstrings[i].as_ptr(),
                content: content_cstrings[i].as_ptr(),
            });
        }

        unsafe {
            let llama_chat_apply_template: Symbol<LlamaChatApplyTemplate> = self
                ._loader
                .get_symbol("llama_chat_apply_template")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_chat_apply_template: {}",
                        e
                    ))
                })?;

            let mut buf_len = 4096;
            loop {
                let mut buf = vec![0i8; buf_len];
                let n = llama_chat_apply_template(
                    std::ptr::null(),
                    chat_messages.as_ptr(),
                    chat_messages.len(),
                    true,
                    buf.as_mut_ptr(),
                    buf_len as i32,
                );

                if n < 0 {
                    return Err(crate::error::HoshikageError::InferenceError(
                        "Failed to apply chat template".to_string(),
                    ));
                }

                if (n as usize) >= buf_len {
                    buf_len = (n as usize) + 1;
                    continue;
                }

                let bytes = std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize);
                let prompt = String::from_utf8_lossy(bytes).to_string();
                return Ok(prompt);
            }
        }
    }

    pub fn prepare_for_inference(&mut self, config: &Config) -> Result<()> {
        let context = self.context.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Context not loaded".to_string())
        })?;

        unsafe {
            if let Ok(kv_clear) = self
                ._loader
                .get_symbol::<LlamaKvCacheClear>("llama_kv_cache_clear")
            {
                kv_clear(context);
                return Ok(());
            }
        }

        if let Some(context) = self.context {
            unsafe {
                if let Ok(llama_free) = self._loader.get_symbol::<LlamaFree>("llama_free") {
                    llama_free(context);
                }
            }
            self.context = None;
        }

        let context = self.init_context(config)?;
        self.context = Some(context);

        Ok(())
    }

    fn init_context(&self, config: &Config) -> Result<*mut llama_context> {
        let model = self.model.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        unsafe {
            let llama_init_from_model: Symbol<LlamaInitFromModel> = self
                ._loader
                .get_symbol("llama_init_from_model")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_init_from_model: {}",
                        e
                    ))
                })?;

            let llama_context_default_params: Symbol<LlamaContextDefaultParams> = self
                ._loader
                .get_symbol("llama_context_default_params")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_context_default_params: {}",
                        e
                    ))
                })?;

            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = config.n_ctx;
            ctx_params.n_batch = config.n_ctx;
            ctx_params.n_ubatch = config.n_ctx;

            let context = llama_init_from_model(model, ctx_params);
            if context.is_null() {
                return Err(crate::error::HoshikageError::ModelLoadFailed(
                    "Context init returned null".to_string(),
                ));
            }

            Ok(context)
        }
    }

    pub fn generate(&self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let mut result = String::new();
        self.generate_with_callback(prompt, params, |chunk| {
            result.push_str(&chunk);
            Ok(())
        })?;

        Ok(result)
    }

    pub fn generate_with_diffusion(
        &self,
        prompt: &str,
        params: &InferenceParams,
        config: &Config,
    ) -> Result<(String, u32, u32)> {
        let prompt_tokens = self.count_tokens(prompt)?;
        let ctx_limit = config.n_ctx as i32;
        let diffusion_limit = if config.diffusion_max_tokens > 0 {
            config.diffusion_max_tokens.min(ctx_limit)
        } else {
            ctx_limit
        };
        let max_tokens = params.max_tokens.min(diffusion_limit);
        let max_length = (prompt_tokens + max_tokens).min(ctx_limit);

        let schedule = TransferSchedule::from_i32(
            params
                .diffusion_schedule
                .unwrap_or(config.diffusion_schedule),
        );

        let diff_params = DiffusionParams {
            steps: params.diffusion_steps.unwrap_or(config.diffusion_steps),
            algorithm: DiffusionAlgorithm::from_i32(
                params
                    .diffusion_algorithm
                    .unwrap_or(config.diffusion_algorithm),
            ),
            schedule,
            cfg_scale: params
                .diffusion_cfg_scale
                .unwrap_or(config.diffusion_cfg_scale),
            max_length,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: 0,
            seed: 0,
            mask_token_id: LLAMA_TOKEN_NULL,
            shift_logits: true,
            eps: 0.0,
            block_length: if schedule == TransferSchedule::BlockBased {
                max_length
            } else {
                0
            },
            alg_temp: 0.0,
            add_gumbel_noise: false,
        };

        let result = self.generate_diffusion(prompt, &diff_params, config, params)?;

        let prompt_tokens = prompt_tokens as u32;
        let completion_tokens = self.count_tokens(&result)? as u32;

        Ok((result, prompt_tokens, completion_tokens))
    }

    pub fn generate_with_callback<F>(
        &self,
        prompt: &str,
        params: &InferenceParams,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(String) -> Result<()>,
    {
        let context = self.context.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Context not loaded".to_string())
        })?;
        let model = self.model.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let prompt_c = CString::new(prompt)?;

        unsafe {
            let llama_model_get_vocab: Symbol<LlamaModelGetVocab> = self
                ._loader
                .get_symbol("llama_model_get_vocab")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_get_vocab: {}",
                        e
                    ))
                })?;
            let vocab = llama_model_get_vocab(model);
            if vocab.is_null() {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Vocab not initialized".to_string(),
                ));
            }

            let llama_tokenize: Symbol<LlamaTokenize> =
                self._loader.get_symbol("llama_tokenize").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_tokenize: {}",
                        e
                    ))
                })?;

            let mut tokens = [0i32; 8192];
            let n_tokens = llama_tokenize(
                vocab,
                prompt_c.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                8192,
                true,
                true,
            );

            if n_tokens < 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Tokenization failed".to_string(),
                ));
            }

            let llama_vocab_n_tokens: Symbol<LlamaVocabNumTokens> = self
                ._loader
                .get_symbol("llama_vocab_n_tokens")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_vocab_n_tokens: {}",
                        e
                    ))
                })?;
            let llama_vocab_get_text: Symbol<LlamaVocabGetText> = self
                ._loader
                .get_symbol("llama_vocab_get_text")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_vocab_get_text: {}",
                        e
                    ))
                })?;
            let llama_token_to_piece: Symbol<LlamaTokenToPiece> = self
                ._loader
                .get_symbol("llama_token_to_piece")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_token_to_piece: {}",
                        e
                    ))
                })?;
            let llama_decode: Symbol<LlamaDecode> =
                self._loader.get_symbol("llama_decode").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_decode: {}",
                        e
                    ))
                })?;
            let llama_get_logits_ith: Symbol<LlamaGetLogitsIth> = self
                ._loader
                .get_symbol("llama_get_logits_ith")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_get_logits_ith: {}",
                        e
                    ))
                })?;

            let n_tokens = n_tokens as usize;
            let mut result = String::new();
            let mut position: i32 = 0;
            let mut token_counts: HashMap<i32, u32> = HashMap::new();
            let mut recent_tokens: VecDeque<i32> = VecDeque::new();
            let repeat_last_n = params.repeat_last_n.min(8192);

            for (i, &token) in tokens.iter().enumerate().take(n_tokens) {
                let mut token = token;
                *token_counts.entry(token).or_insert(0) += 1;
                if repeat_last_n > 0 {
                    recent_tokens.push_back(token);
                    if recent_tokens.len() > repeat_last_n {
                        recent_tokens.pop_front();
                    }
                }
                let mut pos = position;
                let mut seq_id: llama_seq_id = 0;
                let mut seq_id_ptr: *mut llama_seq_id = &mut seq_id;
                let mut n_seq_id: i32 = 1;
                let mut logits: i8 = if i + 1 == n_tokens { 1 } else { 0 };

                let batch = llama_batch {
                    n_tokens: 1,
                    token: &mut token,
                    embd: std::ptr::null_mut(),
                    pos: &mut pos,
                    n_seq_id: &mut n_seq_id,
                    seq_id: &mut seq_id_ptr,
                    logits: &mut logits,
                };

                if llama_decode(context, batch) < 0 {
                    return Err(crate::error::HoshikageError::InferenceError(
                        "Decode failed".to_string(),
                    ));
                }
                position += 1;
            }

            let vocab_size = llama_vocab_n_tokens(vocab);
            if vocab_size <= 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Invalid vocab size".to_string(),
                ));
            }

            let mut generated_tokens = 0;
            let mut pending_bytes: Vec<u8> = Vec::new();
            for _ in 0..params.max_tokens.min(4096_i32) {
                let logits_ptr = llama_get_logits_ith(context, 0);
                if logits_ptr.is_null() {
                    break;
                }
                let logits_slice = std::slice::from_raw_parts(logits_ptr, vocab_size as usize);
                let mut adjusted_logits = logits_slice.to_vec();
                apply_penalties(
                    &mut adjusted_logits,
                    &token_counts,
                    params.presence_penalty,
                    params.frequency_penalty,
                );
                if repeat_last_n > 0 {
                    apply_repeat_penalty(
                        &mut adjusted_logits,
                        recent_tokens.as_slices().0,
                        params.repeat_penalty,
                    );
                    apply_repeat_penalty(
                        &mut adjusted_logits,
                        recent_tokens.as_slices().1,
                        params.repeat_penalty,
                    );
                }
                let token = sample_token(&adjusted_logits, params.temperature, params.top_p)?;

                // Stop token check (string-based, compatible with all model types)
                if let Ok(stop_hit) =
                    Self::is_stop_token(vocab, token, &params.stop_sequences, &llama_vocab_get_text)
                {
                    if stop_hit {
                        if generated_tokens == 0 {
                            // Avoid empty output on first stop token.
                        } else {
                            return Ok(result);
                        }
                    }
                }

                let mut token = token;
                let mut pos = position;
                let mut seq_id: llama_seq_id = 0;
                let mut seq_id_ptr: *mut llama_seq_id = &mut seq_id;
                let mut n_seq_id: i32 = 1;
                let mut logits: i8 = 1;

                let batch = llama_batch {
                    n_tokens: 1,
                    token: &mut token,
                    embd: std::ptr::null_mut(),
                    pos: &mut pos,
                    n_seq_id: &mut n_seq_id,
                    seq_id: &mut seq_id_ptr,
                    logits: &mut logits,
                };

                if llama_decode(context, batch) < 0 {
                    return Err(crate::error::HoshikageError::InferenceError(
                        "Decode failed".to_string(),
                    ));
                }
                position += 1;
                *token_counts.entry(token).or_insert(0) += 1;
                if repeat_last_n > 0 {
                    recent_tokens.push_back(token);
                    if recent_tokens.len() > repeat_last_n {
                        recent_tokens.pop_front();
                    }
                }

                let piece_ptr = llama_vocab_get_text(vocab, token);
                if piece_ptr.is_null() {
                    continue;
                }

                let mut buf = vec![0i8; 256];
                let mut n = llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0,
                    false,
                );
                if n < 0 {
                    let needed = (-n) as usize;
                    buf.resize(needed, 0);
                    n = llama_token_to_piece(
                        vocab,
                        token,
                        buf.as_mut_ptr(),
                        buf.len() as i32,
                        0,
                        false,
                    );
                }

                if n > 0 {
                    let bytes = std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize);
                    pending_bytes.extend_from_slice(bytes);

                    loop {
                        match std::str::from_utf8(&pending_bytes) {
                            Ok(valid) => {
                                if !valid.is_empty() {
                                    on_token(valid.to_string())?;
                                    result.push_str(valid);
                                    generated_tokens += 1;
                                }
                                pending_bytes.clear();
                                break;
                            }
                            Err(err) => {
                                let valid_up_to = err.valid_up_to();
                                if valid_up_to == 0 {
                                    break;
                                }
                                let valid =
                                    std::str::from_utf8_unchecked(&pending_bytes[..valid_up_to]);
                                on_token(valid.to_string())?;
                                result.push_str(valid);
                                generated_tokens += 1;
                                pending_bytes.drain(..valid_up_to);
                            }
                        }
                    }

                    for stop_seq in &params.stop_sequences {
                        if result.ends_with(stop_seq) {
                            if generated_tokens == 1 && result == stop_seq.as_str() {
                                // First token is a stop sequence; ignore once to avoid empty output.
                                break;
                            }
                            result = result[..result.len() - stop_seq.len()].to_string();
                            return Ok(result);
                        }
                    }
                }
            }

            Ok(result)
        }
    }

    fn is_stop_token(
        vocab: *const llama_vocab,
        token: llama_token,
        stop_sequences: &[String],
        llama_vocab_get_text: &Symbol<LlamaVocabGetText>,
    ) -> Result<bool> {
        if stop_sequences.is_empty() {
            return Ok(false);
        }
        let text_ptr = unsafe { llama_vocab_get_text(vocab, token) };
        if text_ptr.is_null() {
            return Ok(false);
        }
        let text = unsafe { CStr::from_ptr(text_ptr) }
            .to_string_lossy()
            .to_string();
        Ok(stop_sequences.iter().any(|s| s == &text))
    }

    pub fn generate_diffusion(
        &self,
        prompt: &str,
        params: &DiffusionParams,
        config: &Config,
        inference_params: &InferenceParams,
    ) -> Result<String> {
        let context = self.context.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Context not loaded".to_string())
        })?;
        let model = self.model.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let prompt_c = CString::new(prompt)?;

        unsafe {
            let llama_model_get_vocab: Symbol<LlamaModelGetVocab> = self
                ._loader
                .get_symbol("llama_model_get_vocab")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_get_vocab: {}",
                        e
                    ))
                })?;
            let vocab = llama_model_get_vocab(model);
            if vocab.is_null() {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Vocab not initialized".to_string(),
                ));
            }

            let llama_tokenize: Symbol<LlamaTokenize> =
                self._loader.get_symbol("llama_tokenize").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_tokenize: {}",
                        e
                    ))
                })?;

            let llama_vocab_n_tokens: Symbol<LlamaVocabNumTokens> = self
                ._loader
                .get_symbol("llama_vocab_n_tokens")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_vocab_n_tokens: {}",
                        e
                    ))
                })?;

            let llama_token_to_piece: Symbol<LlamaTokenToPiece> = self
                ._loader
                .get_symbol("llama_token_to_piece")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_token_to_piece: {}",
                        e
                    ))
                })?;

            let llama_decode: Symbol<LlamaDecode> =
                self._loader.get_symbol("llama_decode").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_decode: {}",
                        e
                    ))
                })?;
            let llama_encode: Symbol<LlamaEncode> =
                self._loader.get_symbol("llama_encode").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_encode: {}",
                        e
                    ))
                })?;
            let llama_model_has_encoder: Symbol<LlamaModelHasEncoder> = self
                ._loader
                .get_symbol("llama_model_has_encoder")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_has_encoder: {}",
                        e
                    ))
                })?;
            let llama_model_has_decoder: Symbol<LlamaModelHasDecoder> = self
                ._loader
                .get_symbol("llama_model_has_decoder")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_has_decoder: {}",
                        e
                    ))
                })?;
            let llama_model_is_diffusion: Symbol<LlamaModelIsDiffusion> = self
                ._loader
                .get_symbol("llama_model_is_diffusion")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_is_diffusion: {}",
                        e
                    ))
                })?;

            let llama_get_logits: Symbol<LlamaGetLogits> =
                self._loader.get_symbol("llama_get_logits").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_get_logits: {}",
                        e
                    ))
                })?;

            let llama_batch_init: Symbol<LlamaBatchInit> =
                self._loader.get_symbol("llama_batch_init").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_batch_init: {}",
                        e
                    ))
                })?;

            let llama_batch_free: Symbol<LlamaBatchFree> =
                self._loader.get_symbol("llama_batch_free").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_batch_free: {}",
                        e
                    ))
                })?;

            let llama_vocab_mask: Symbol<LlamaVocabMask> =
                self._loader.get_symbol("llama_vocab_mask").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_vocab_mask: {}",
                        e
                    ))
                })?;

            let llama_set_causal_attn: Symbol<LlamaSetCausalAttn> = self
                ._loader
                .get_symbol("llama_set_causal_attn")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_set_causal_attn: {}",
                        e
                    ))
                })?;

            let llama_model_meta_val_str: Symbol<LlamaModelMetaValStr> = self
                ._loader
                .get_symbol("llama_model_meta_val_str")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_meta_val_str: {}",
                        e
                    ))
                })?;

            let llama_sampler_chain_default_params: Symbol<LlamaSamplerChainDefaultParams> = self
                ._loader
                .get_symbol("llama_sampler_chain_default_params")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_chain_default_params: {}",
                        e
                    ))
                })?;

            let llama_sampler_chain_init: Symbol<LlamaSamplerChainInit> = self
                ._loader
                .get_symbol("llama_sampler_chain_init")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_chain_init: {}",
                        e
                    ))
                })?;

            let llama_sampler_chain_add: Symbol<LlamaSamplerChainAdd> = self
                ._loader
                .get_symbol("llama_sampler_chain_add")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_chain_add: {}",
                        e
                    ))
                })?;

            let llama_sampler_init_top_k: Symbol<LlamaSamplerInitTopK> = self
                ._loader
                .get_symbol("llama_sampler_init_top_k")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_init_top_k: {}",
                        e
                    ))
                })?;

            let llama_sampler_init_top_p: Symbol<LlamaSamplerInitTopP> = self
                ._loader
                .get_symbol("llama_sampler_init_top_p")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_init_top_p: {}",
                        e
                    ))
                })?;

            let llama_sampler_init_temp: Symbol<LlamaSamplerInitTemp> = self
                ._loader
                .get_symbol("llama_sampler_init_temp")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_init_temp: {}",
                        e
                    ))
                })?;

            let llama_sampler_init_dist: Symbol<LlamaSamplerInitDist> = self
                ._loader
                .get_symbol("llama_sampler_init_dist")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_init_dist: {}",
                        e
                    ))
                })?;

            let llama_sampler_apply: Symbol<LlamaSamplerApply> = self
                ._loader
                .get_symbol("llama_sampler_apply")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_apply: {}",
                        e
                    ))
                })?;

            let llama_sampler_free: Symbol<LlamaSamplerFree> =
                self._loader.get_symbol("llama_sampler_free").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_sampler_free: {}",
                        e
                    ))
                })?;

            let n_ctx = config.n_ctx as usize;
            let max_length = params.max_length.min(n_ctx as i32);
            if max_length <= 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Invalid max_length for diffusion".to_string(),
                ));
            }
            let max_length = max_length as usize;

            let vocab_size = llama_vocab_n_tokens(vocab) as usize;
            if vocab_size == 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Invalid vocab size".to_string(),
                ));
            }

            let mut tokens = vec![0i32; n_ctx];
            let n_prompt_tokens = llama_tokenize(
                vocab,
                prompt_c.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            );

            if n_prompt_tokens < 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Tokenization failed".to_string(),
                ));
            }

            let n_prompt_tokens = n_prompt_tokens as usize;
            if n_prompt_tokens >= max_length {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Prompt too long for diffusion".to_string(),
                ));
            }

            let mask_token = llama_vocab_mask(vocab);
            if mask_token == LLAMA_TOKEN_NULL {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Mask token not found".to_string(),
                ));
            }

            let mut shift_logits = params.shift_logits;
            let key = CString::new("diffusion.shift_logits")?;
            let mut shift_buf = [0i8; 8];
            if llama_model_meta_val_str(
                model,
                key.as_ptr(),
                shift_buf.as_mut_ptr(),
                shift_buf.len(),
            ) >= 0
            {
                if let Ok(value) = CStr::from_ptr(shift_buf.as_ptr()).to_str() {
                    shift_logits = value == "true";
                }
            } else {
                shift_logits = true;
            }

            let mut output_tokens = vec![0i32; max_length];
            output_tokens[..n_prompt_tokens].copy_from_slice(&tokens[..n_prompt_tokens]);
            output_tokens[n_prompt_tokens..max_length].fill(mask_token);

            llama_set_causal_attn(context, false);

            let sampler_chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if sampler_chain.is_null() {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Failed to init sampler chain".to_string(),
                ));
            }
            if params.top_k > 0 {
                let top_k_sampler = llama_sampler_init_top_k(params.top_k);
                if !top_k_sampler.is_null() {
                    llama_sampler_chain_add(sampler_chain, top_k_sampler);
                }
            }
            if params.top_p < 1.0 {
                let top_p_sampler = llama_sampler_init_top_p(params.top_p, 1);
                if !top_p_sampler.is_null() {
                    llama_sampler_chain_add(sampler_chain, top_p_sampler);
                }
            }
            if params.temperature > 0.0 {
                let temp_sampler = llama_sampler_init_temp(params.temperature);
                if !temp_sampler.is_null() {
                    llama_sampler_chain_add(sampler_chain, temp_sampler);
                }
            }
            let dist_in_chain = llama_sampler_init_dist(params.seed);
            if !dist_in_chain.is_null() {
                llama_sampler_chain_add(sampler_chain, dist_in_chain);
            }

            let dist_sampler = llama_sampler_init_dist(params.seed);
            if dist_sampler.is_null() {
                llama_sampler_free(sampler_chain);
                return Err(crate::error::HoshikageError::InferenceError(
                    "Failed to init dist sampler".to_string(),
                ));
            }

            let mut batch = llama_batch_init(max_length as i32, 0, 1);
            if batch.token.is_null() {
                llama_sampler_free(sampler_chain);
                llama_sampler_free(dist_sampler);
                return Err(crate::error::HoshikageError::InferenceError(
                    "Failed to init llama_batch".to_string(),
                ));
            }
            batch.n_tokens = max_length as i32;

            let tokens_slice = std::slice::from_raw_parts_mut(batch.token, max_length);
            let pos_slice = std::slice::from_raw_parts_mut(batch.pos, max_length);
            let n_seq_id_slice = std::slice::from_raw_parts_mut(batch.n_seq_id, max_length);
            let seq_id_slice = std::slice::from_raw_parts_mut(batch.seq_id, max_length);
            let logits_slice = std::slice::from_raw_parts_mut(batch.logits, max_length);

            let logits_size = vocab_size * max_length;
            let mut cond_logits_buffer = Vec::new();
            let mut un_x_buffer = Vec::new();
            if params.cfg_scale > 0.0 {
                cond_logits_buffer.resize(logits_size, 0.0f32);
                un_x_buffer.resize(max_length, mask_token);
            }

            let mut rng = StdRng::seed_from_u64(params.seed as u64);

            let mut num_transfer_tokens: Vec<i32> = Vec::new();
            let (num_blocks, steps_per_block) = match params.schedule {
                TransferSchedule::BlockBased => {
                    let block_length = if params.block_length > 0 {
                        params.block_length as usize
                    } else {
                        max_length
                    };
                    #[allow(clippy::manual_is_multiple_of)]
                    if max_length % block_length != 0 {
                        llama_batch_free(batch);
                        llama_sampler_free(sampler_chain);
                        llama_sampler_free(dist_sampler);
                        return Err(crate::error::HoshikageError::InferenceError(
                            "Invalid block length for diffusion".to_string(),
                        ));
                    }
                    let num_blocks = max_length / block_length;
                    if params.steps % num_blocks as i32 != 0 {
                        llama_batch_free(batch);
                        llama_sampler_free(sampler_chain);
                        llama_sampler_free(dist_sampler);
                        return Err(crate::error::HoshikageError::InferenceError(
                            "Steps not divisible by num_blocks".to_string(),
                        ));
                    }
                    (num_blocks, params.steps / num_blocks as i32)
                }
                TransferSchedule::TimestepBased => (1, params.steps),
            };

            let mut candidates = vec![
                llama_token_data {
                    id: 0,
                    logit: 0.0,
                    p: 0.0,
                };
                vocab_size
            ];
            let mut conf_candidates: Vec<llama_token_data> = Vec::new();
            let mut mask_positions: Vec<usize> = Vec::new();

            let has_encoder = llama_model_has_encoder(model);
            let has_decoder = llama_model_has_decoder(model);
            let is_diffusion = llama_model_is_diffusion(model);
            let use_encode = if is_diffusion {
                has_encoder
            } else {
                has_encoder && !has_decoder
            };

            for block_num in 0..num_blocks {
                let block_start = if params.schedule == TransferSchedule::BlockBased {
                    n_prompt_tokens + block_num * (params.block_length as usize)
                } else {
                    0
                };
                let block_end = if params.schedule == TransferSchedule::BlockBased {
                    (n_prompt_tokens + (block_num + 1) * (params.block_length as usize))
                        .min(max_length)
                } else {
                    max_length
                };

                if params.schedule == TransferSchedule::BlockBased {
                    let mut block_mask_count = 0;
                    for &token in output_tokens.iter().take(block_end).skip(block_start) {
                        if token == mask_token {
                            block_mask_count += 1;
                        }
                    }
                    num_transfer_tokens =
                        get_num_transfer_tokens(block_mask_count, steps_per_block);
                }

                for step in 0..steps_per_block {
                    for i in 0..max_length {
                        tokens_slice[i] = output_tokens[i];
                        pos_slice[i] = i as llama_pos;
                        n_seq_id_slice[i] = 1;
                        logits_slice[i] = 1;
                        let seq_ptr = seq_id_slice[i];
                        if !seq_ptr.is_null() {
                            *seq_ptr = 0;
                        }
                    }

                    let logits_ptr = if params.cfg_scale > 0.0 {
                        let ret = if use_encode {
                            llama_encode(context, batch)
                        } else {
                            llama_decode(context, batch)
                        };
                        if ret != 0 {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Failed to decode conditional".to_string(),
                            ));
                        }
                        let cond_logits_ptr = llama_get_logits(context);
                        if cond_logits_ptr.is_null() {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Failed to get conditional logits".to_string(),
                            ));
                        }
                        std::ptr::copy_nonoverlapping(
                            cond_logits_ptr,
                            cond_logits_buffer.as_mut_ptr(),
                            logits_size,
                        );

                        un_x_buffer.copy_from_slice(&output_tokens);
                        un_x_buffer[..n_prompt_tokens].fill(mask_token);
                        tokens_slice[..max_length].copy_from_slice(&un_x_buffer[..max_length]);
                        let ret = if use_encode {
                            llama_encode(context, batch)
                        } else {
                            llama_decode(context, batch)
                        };
                        if ret != 0 {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Failed to decode unconditional".to_string(),
                            ));
                        }
                        let uncond_logits_ptr = llama_get_logits(context);
                        if uncond_logits_ptr.is_null() {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Failed to get unconditional logits".to_string(),
                            ));
                        }

                        for (i, val) in cond_logits_buffer.iter_mut().enumerate().take(logits_size)
                        {
                            let uncond = *uncond_logits_ptr.add(i);
                            let cond = *val;
                            *val = uncond + (params.cfg_scale + 1.0) * (cond - uncond);
                        }
                        cond_logits_buffer.as_ptr()
                    } else {
                        let ret = if use_encode {
                            llama_encode(context, batch)
                        } else {
                            llama_decode(context, batch)
                        };
                        if ret != 0 {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(format!(
                                "Failed to decode at step {}",
                                step
                            )));
                        }
                        let ptr = llama_get_logits(context);
                        if ptr.is_null() {
                            llama_batch_free(batch);
                            llama_sampler_free(sampler_chain);
                            llama_sampler_free(dist_sampler);
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Failed to get logits".to_string(),
                            ));
                        }
                        ptr
                    };

                    let get_logits_for_pos = |pos: usize| -> *const f32 {
                        if shift_logits {
                            if pos == 0 {
                                logits_ptr
                            } else {
                                logits_ptr.add((pos - 1) * vocab_size)
                            }
                        } else {
                            logits_ptr.add(pos * vocab_size)
                        }
                    };

                    mask_positions.clear();
                    for (i, &token) in output_tokens.iter().enumerate().take(max_length) {
                        if token == mask_token
                            && (params.schedule != TransferSchedule::BlockBased
                                || (i >= block_start && i < block_end))
                        {
                            mask_positions.push(i);
                        }
                    }

                    if mask_positions.is_empty() {
                        break;
                    }

                    let transfer_count = calculate_transfer_count(
                        step,
                        steps_per_block,
                        mask_positions.len() as i32,
                        params.schedule,
                        params.eps,
                        &num_transfer_tokens,
                    );

                    if params.algorithm == DiffusionAlgorithm::Origin {
                        let transfer_count = transfer_count.max(0);
                        let p_transfer = transfer_count as f32 / mask_positions.len().max(1) as f32;
                        for pos in &mask_positions {
                            if rng.gen::<f32>() < p_transfer {
                                let pos_logits = get_logits_for_pos(*pos);
                                for (token_id, cand) in
                                    candidates.iter_mut().enumerate().take(vocab_size)
                                {
                                    cand.id = token_id as i32;
                                    cand.logit = *pos_logits.add(token_id);
                                    cand.p = 0.0;
                                }
                                let mut cur_p = llama_token_data_array {
                                    data: candidates.as_mut_ptr(),
                                    size: vocab_size,
                                    selected: -1,
                                    sorted: false,
                                };
                                llama_sampler_apply(sampler_chain, &mut cur_p);
                                if cur_p.selected >= 0 {
                                    output_tokens[*pos] = candidates[cur_p.selected as usize].id;
                                }
                            }
                        }
                    } else {
                        let mut confidences: Vec<(f32, usize)> =
                            Vec::with_capacity(mask_positions.len());
                        let mut sampled_tokens: Vec<llama_token> =
                            Vec::with_capacity(mask_positions.len());

                        for (i, pos) in mask_positions.iter().enumerate() {
                            let pos_logits = get_logits_for_pos(*pos);
                            for (token_id, cand) in
                                candidates.iter_mut().enumerate().take(vocab_size)
                            {
                                cand.id = token_id as i32;
                                cand.logit = *pos_logits.add(token_id);
                                cand.p = 0.0;
                            }

                            let mut cur_p = llama_token_data_array {
                                data: candidates.as_mut_ptr(),
                                size: vocab_size,
                                selected: -1,
                                sorted: false,
                            };
                            llama_sampler_apply(sampler_chain, &mut cur_p);

                            let sampled = if cur_p.selected >= 0 {
                                candidates[cur_p.selected as usize].id
                            } else {
                                0
                            };
                            let conf = calculate_confidence(&cur_p, params.algorithm, &mut rng);
                            sampled_tokens.push(sampled);
                            confidences.push((conf, i));
                        }

                        if transfer_count > 0 {
                            if params.alg_temp == 0.0 {
                                confidences.sort_by(|a, b| {
                                    b.0.partial_cmp(&a.0)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then_with(|| a.1.cmp(&b.1))
                                });
                                for i in 0..transfer_count.min(confidences.len() as i32) {
                                    let mask_idx = confidences[i as usize].1;
                                    let pos = mask_positions[mask_idx];
                                    output_tokens[pos] = sampled_tokens[mask_idx];
                                }
                            } else {
                                conf_candidates.clear();
                                for (idx, (conf, _)) in confidences.iter().enumerate() {
                                    conf_candidates.push(llama_token_data {
                                        id: idx as i32,
                                        logit: conf / params.alg_temp,
                                        p: 0.0,
                                    });
                                }
                                let mut conf_array = llama_token_data_array {
                                    data: conf_candidates.as_mut_ptr(),
                                    size: conf_candidates.len(),
                                    selected: -1,
                                    sorted: false,
                                };
                                for _ in 0..transfer_count.min(confidences.len() as i32) {
                                    llama_sampler_apply(dist_sampler, &mut conf_array);
                                    if conf_array.selected < 0 {
                                        break;
                                    }
                                    let selected_idx = conf_array.selected as usize;
                                    let mask_idx = selected_idx;
                                    let pos = mask_positions[mask_idx];
                                    output_tokens[pos] = sampled_tokens[mask_idx];
                                    conf_candidates[selected_idx].p = 0.0;
                                    conf_array.selected = -1;
                                }
                            }
                        }
                    }
                }
            }

            llama_batch_free(batch);
            llama_sampler_free(sampler_chain);
            llama_sampler_free(dist_sampler);

            let mut result = String::new();
            for &token in output_tokens.iter().take(max_length).skip(n_prompt_tokens) {
                if token == mask_token {
                    continue;
                }

                let mut buf = vec![0i8; 256];
                let n = llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0,
                    false,
                );
                if n > 0 {
                    let bytes = std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize);
                    match std::str::from_utf8(bytes) {
                        Ok(text) => {
                            result.push_str(text);
                            if inference_params
                                .stop_sequences
                                .iter()
                                .any(|stop| result.ends_with(stop))
                            {
                                for stop in &inference_params.stop_sequences {
                                    if result.ends_with(stop) {
                                        let new_len = result.len().saturating_sub(stop.len());
                                        result.truncate(new_len);
                                        return Ok(result);
                                    }
                                }
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }

            Ok(result)
        }
    }

    pub fn count_tokens(&self, text: &str) -> Result<i32> {
        let model = self.model.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let text_c = CString::new(text)?;
        unsafe {
            let llama_model_get_vocab: Symbol<LlamaModelGetVocab> = self
                ._loader
                .get_symbol("llama_model_get_vocab")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_model_get_vocab: {}",
                        e
                    ))
                })?;
            let vocab = llama_model_get_vocab(model);
            if vocab.is_null() {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Vocab not initialized".to_string(),
                ));
            }

            let llama_tokenize: Symbol<LlamaTokenize> =
                self._loader.get_symbol("llama_tokenize").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol llama_tokenize: {}",
                        e
                    ))
                })?;

            let mut tokens = vec![0i32; 8192];
            let n_tokens = llama_tokenize(
                vocab,
                text_c.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            );

            if n_tokens < 0 {
                return Err(crate::error::HoshikageError::InferenceError(
                    "Tokenization failed".to_string(),
                ));
            }

            Ok(n_tokens)
        }
    }

    pub fn unload(&mut self) {
        if let Some(context) = self.context {
            unsafe {
                if let Ok(llama_free) = self._loader.get_symbol::<LlamaFree>("llama_free") {
                    llama_free(context);
                }
            }
            self.context = None;
        }

        if let Some(model) = self.model {
            unsafe {
                if let Ok(llama_free_model) = self
                    ._loader
                    .get_symbol::<LlamaFreeModel>("llama_free_model")
                {
                    llama_free_model(model);
                }
            }
            self.model = None;
        }

        unsafe {
            if let Ok(llama_backend_free) = self
                ._loader
                .get_symbol::<LlamaBackendFree>("llama_backend_free")
            {
                llama_backend_free();
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some() && self.context.is_some()
    }

    pub fn is_diffusion_model(&self) -> Result<bool> {
        let model = self.model.ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        unsafe {
            // シンボルがなければ false を返す（後方互換性: 古い libllama.so 対応）
            let func = match self
                ._loader
                .get_symbol::<LlamaModelIsDiffusion>("llama_model_is_diffusion")
            {
                Ok(f) => f,
                Err(_) => {
                    tracing::debug!(
                        "llama_model_is_diffusion symbol not found, assuming non-diffusion model"
                    );
                    return Ok(false);
                }
            };

            Ok(func(model))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_apply_penalties() {
        let mut logits = vec![0.0, 1.0, 2.0, 3.0];
        let mut counts = HashMap::new();
        counts.insert(1, 2);
        counts.insert(3, 1);

        apply_penalties(&mut logits, &counts, 0.5, 0.25);

        assert!((logits[0] - 0.0).abs() < f32::EPSILON);
        assert!((logits[1] - (1.0 - 0.5 - 0.5)).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < f32::EPSILON);
        assert!((logits[3] - (3.0 - 0.5 - 0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_new_wrapper_failure() {
        let wrapper = LlamaWrapper::new(PathBuf::from("/fake/path/libllama.so"));
        assert!(wrapper.is_err());
    }

    #[test]
    fn test_wrapper_not_loaded() {
        if let Ok(wrapper) = LlamaWrapper::new(PathBuf::from("/fake/path/libllama.so")) {
            assert!(!wrapper.is_loaded());
        }
    }
}
