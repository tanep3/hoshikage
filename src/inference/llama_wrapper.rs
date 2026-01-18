use crate::config::Config;
use crate::error::Result;
use crate::ffi::*;
use crate::inference::loader::DynamicLibraryLoader;
use libloading::Symbol;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
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
pub type LlamaVocabGetText = unsafe extern "C" fn(vocab: *const llama_vocab, token: llama_token) -> *const ::std::os::raw::c_char;
pub type LlamaTokenToPiece = unsafe extern "C" fn(
    vocab: *const llama_vocab,
    token: llama_token,
    buf: *mut ::std::os::raw::c_char,
    length: i32,
    lstrip: i32,
    special: bool,
) -> i32;
pub type LlamaGetLogitsIth = unsafe extern "C" fn(ctx: *mut llama_context, i: i32) -> *mut f32;
pub type LlamaKvCacheClear = unsafe extern "C" fn(ctx: *mut llama_context);
pub type LlamaChatApplyTemplate = unsafe extern "C" fn(
    tmpl: *const ::std::os::raw::c_char,
    chat: *const llama_chat_message,
    n_msg: usize,
    add_ass: bool,
    buf: *mut ::std::os::raw::c_char,
    length: i32,
) -> i32;

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
        }
    }
}

pub struct LlamaWrapper {
    _loader: DynamicLibraryLoader,
    model: Option<*mut llama_model>,
    context: Option<*mut llama_context>,
}

unsafe impl Send for LlamaWrapper {}

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
    let max_logit = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| ((i), ((logit - max_logit) / temp).exp()))
        .collect();

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let threshold = total * top_p.max(0.0).min(1.0);
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
        BACKEND_INIT.call_once(|| {
            unsafe {
                if let Ok(llama_backend_init) =
                    loader.get_symbol::<LlamaBackendInit>("llama_backend_init")
                {
                    llama_backend_init();
                }
                #[cfg(target_os = "linux")]
                {
                    if let Ok(llama_numa_init) =
                        loader.get_symbol::<LlamaNumaInit>("llama_numa_init")
                    {
                        llama_numa_init(0);
                    }
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
            if let Ok(kv_clear) = self._loader.get_symbol::<LlamaKvCacheClear>("llama_kv_cache_clear")
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

            for i in 0..n_tokens {
                let mut token = tokens[i];
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
            for _ in 0..params.max_tokens.min(4096 as i32) {
                let logits_ptr = llama_get_logits_ith(context, 0);
                if logits_ptr.is_null() {
                    break;
                }
                let logits_slice =
                    std::slice::from_raw_parts(logits_ptr, vocab_size as usize);
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
            if let Ok(llama_backend_free) =
                self._loader.get_symbol::<LlamaBackendFree>("llama_backend_free")
            {
                llama_backend_free();
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some() && self.context.is_some()
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
