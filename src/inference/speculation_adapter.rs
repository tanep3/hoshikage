use crate::error::{HoshikageError, Result};
use crate::ffi::{llama_context, llama_model, llama_pos, llama_seq_id, llama_token};
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

const EXPECTED_ABI_VERSION: u32 = 1;
const SPECULATION_MODE_MTP: i32 = 1;
const SPECULATION_MODE_DRAFT_MODEL: i32 = 2;

type HoshikageAdapterAbiVersion = unsafe extern "C" fn() -> u32;
type HoshikageAdapterLastError = unsafe extern "C" fn() -> *const c_char;
type HoshikageSpeculationSupports = unsafe extern "C" fn(mode: i32) -> bool;
type HoshikageSpeculationInit = unsafe extern "C" fn(
    target_model: *const llama_model,
    target_context: *mut llama_context,
    config: *const HoshikageSpeculationConfig,
) -> *mut c_void;
type HoshikageSpeculationFree = unsafe extern "C" fn(context: *mut c_void);
type HoshikageSpeculationDraft = unsafe extern "C" fn(
    context: *mut c_void,
    seq_id: llama_seq_id,
    n_past: llama_pos,
    id_last: llama_token,
    prompt_tokens: *const llama_token,
    n_prompt_tokens: usize,
    out_tokens: *mut llama_token,
    out_capacity: usize,
    out_n_tokens: *mut usize,
) -> i32;
type HoshikageSpeculationProcess =
    unsafe extern "C" fn(context: *mut c_void, batch: *const crate::ffi::llama_batch) -> bool;
type HoshikageSpeculationAccept =
    unsafe extern "C" fn(context: *mut c_void, seq_id: llama_seq_id, n_accepted: u16);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HoshikageSpeculationConfig {
    mode: i32,
    n_draft_max: i32,
    n_seq: i32,
    n_ctx: u32,
    n_gpu_layers_draft: i32,
    draft_model_path: *const c_char,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculationAdapterMode {
    Mtp,
    DraftModel,
}

impl SpeculationAdapterMode {
    fn as_ffi_mode(&self) -> i32 {
        match self {
            Self::Mtp => SPECULATION_MODE_MTP,
            Self::DraftModel => SPECULATION_MODE_DRAFT_MODEL,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeculationSessionConfig {
    pub mode: SpeculationAdapterMode,
    pub n_draft_max: i32,
    pub n_seq: i32,
    pub n_ctx: u32,
    pub n_gpu_layers_draft: i32,
    pub draft_model_path: Option<PathBuf>,
}

pub struct SpeculationAdapter {
    library: Arc<Library>,
    path: PathBuf,
    abi_version: u32,
}

unsafe impl Send for SpeculationAdapter {}

impl SpeculationAdapter {
    pub fn load(libllama_path: &Path) -> Result<Self> {
        let path = adapter_path_for(libllama_path);
        let library = unsafe {
            Library::new(&path).map_err(|e| {
                crate::error::HoshikageError::LibraryLoadError(format!(
                    "Failed to load {}: {}",
                    path.display(),
                    e
                ))
            })?
        };

        let abi_version = unsafe {
            let abi_version: Symbol<HoshikageAdapterAbiVersion> =
                library.get(b"hoshikage_adapter_abi_version").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_adapter_abi_version: {}",
                        e
                    ))
                })?;
            abi_version()
        };

        if abi_version != EXPECTED_ABI_VERSION {
            return Err(crate::error::HoshikageError::LibraryLoadError(format!(
                "Unsupported Hoshikage adapter ABI version: expected {}, got {}",
                EXPECTED_ABI_VERSION, abi_version
            )));
        }

        Ok(Self {
            library: Arc::new(library),
            path,
            abi_version,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    pub fn supports_mtp(&self) -> Result<bool> {
        self.supports_mode(SPECULATION_MODE_MTP)
    }

    pub fn supports_draft_model(&self) -> Result<bool> {
        self.supports_mode(SPECULATION_MODE_DRAFT_MODEL)
    }

    pub fn init_session(
        &self,
        target_model: *const llama_model,
        target_context: *mut llama_context,
        config: &SpeculationSessionConfig,
    ) -> Result<SpeculationSession> {
        if target_model.is_null() || target_context.is_null() {
            return Err(HoshikageError::InferenceError(
                "Cannot initialize speculation session without loaded target runtime".to_string(),
            ));
        }

        let draft_model_path = config
            .draft_model_path
            .as_ref()
            .map(|path| CString::new(path.to_string_lossy().as_bytes()))
            .transpose()?;

        let ffi_config = HoshikageSpeculationConfig {
            mode: config.mode.as_ffi_mode(),
            n_draft_max: config.n_draft_max,
            n_seq: config.n_seq,
            n_ctx: config.n_ctx,
            n_gpu_layers_draft: config.n_gpu_layers_draft,
            draft_model_path: draft_model_path
                .as_ref()
                .map(|path| path.as_ptr())
                .unwrap_or(std::ptr::null()),
        };

        let init = unsafe {
            self.library
                .get::<HoshikageSpeculationInit>(b"hoshikage_speculation_init")
                .map_err(|e| {
                    HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_init: {}",
                        e
                    ))
                })?
        };
        let free = unsafe {
            self.library
                .get::<HoshikageSpeculationFree>(b"hoshikage_speculation_free")
                .map_err(|e| {
                    HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_free: {}",
                        e
                    ))
                })?
        };
        let draft = unsafe {
            self.library
                .get::<HoshikageSpeculationDraft>(b"hoshikage_speculation_draft")
                .map_err(|e| {
                    HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_draft: {}",
                        e
                    ))
                })?
        };
        let accept = unsafe {
            self.library
                .get::<HoshikageSpeculationAccept>(b"hoshikage_speculation_accept")
                .map_err(|e| {
                    HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_accept: {}",
                        e
                    ))
                })?
        };
        let process = unsafe {
            self.library
                .get::<HoshikageSpeculationProcess>(b"hoshikage_speculation_process")
                .map_err(|e| {
                    HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_process: {}",
                        e
                    ))
                })?
        };

        let context = unsafe { init(target_model, target_context, &ffi_config) };
        let context = NonNull::new(context).ok_or_else(|| {
            HoshikageError::InferenceError(format!(
                "Speculation adapter session init returned null: {}",
                self.last_error()
                    .unwrap_or_else(|| "no adapter error detail".to_string())
            ))
        })?;

        Ok(SpeculationSession {
            _library: Arc::clone(&self.library),
            context,
            free: *free,
            draft: *draft,
            process: *process,
            accept: *accept,
        })
    }

    fn supports_mode(&self, mode: i32) -> Result<bool> {
        unsafe {
            let supports: Symbol<HoshikageSpeculationSupports> = self
                .library
                .get(b"hoshikage_speculation_supports")
                .map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol hoshikage_speculation_supports: {}",
                        e
                    ))
                })?;
            Ok(supports(mode))
        }
    }

    fn last_error(&self) -> Option<String> {
        let last_error = unsafe {
            self.library
                .get::<HoshikageAdapterLastError>(b"hoshikage_adapter_last_error")
                .ok()?
        };
        let ptr = unsafe { last_error() };
        if ptr.is_null() {
            return None;
        }

        let message = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .to_string();
        if message.is_empty() {
            None
        } else {
            Some(message)
        }
    }
}

pub struct SpeculationSession {
    _library: Arc<Library>,
    context: NonNull<c_void>,
    free: HoshikageSpeculationFree,
    draft: HoshikageSpeculationDraft,
    process: HoshikageSpeculationProcess,
    accept: HoshikageSpeculationAccept,
}

unsafe impl Send for SpeculationSession {}

impl SpeculationSession {
    pub fn draft(
        &self,
        seq_id: llama_seq_id,
        n_past: llama_pos,
        id_last: llama_token,
        prompt_tokens: &[llama_token],
        out_tokens: &mut [llama_token],
    ) -> Result<usize> {
        let mut out_n_tokens = 0usize;
        let rc = unsafe {
            (self.draft)(
                self.context.as_ptr(),
                seq_id,
                n_past,
                id_last,
                prompt_tokens.as_ptr(),
                prompt_tokens.len(),
                out_tokens.as_mut_ptr(),
                out_tokens.len(),
                &mut out_n_tokens,
            )
        };

        if rc < 0 {
            return Err(HoshikageError::InferenceError(format!(
                "Speculation adapter draft failed: {}",
                rc
            )));
        }

        Ok(out_n_tokens.min(out_tokens.len()))
    }

    pub fn accept(&self, seq_id: llama_seq_id, n_accepted: u16) {
        unsafe {
            (self.accept)(self.context.as_ptr(), seq_id, n_accepted);
        }
    }

    pub(crate) fn process(&self, batch: &crate::ffi::llama_batch) -> Result<()> {
        let ok = unsafe { (self.process)(self.context.as_ptr(), batch) };
        if ok {
            Ok(())
        } else {
            Err(HoshikageError::InferenceError(
                "Speculation adapter process failed".to_string(),
            ))
        }
    }
}

impl Drop for SpeculationSession {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.context.as_ptr());
        }
    }
}

pub fn adapter_path_for(libllama_path: &Path) -> PathBuf {
    libllama_path
        .parent()
        .map(|dir| dir.join(platform_library_name("hoshikage-llama-adapter")))
        .unwrap_or_else(|| PathBuf::from(platform_library_name("hoshikage-llama-adapter")))
}

fn platform_library_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_path_uses_libllama_directory() {
        let path = adapter_path_for(Path::new("/opt/hoshikage/lib/libllama.so"));

        if cfg!(target_os = "windows") {
            assert_eq!(
                path,
                PathBuf::from("/opt/hoshikage/lib/hoshikage-llama-adapter.dll")
            );
        } else if cfg!(target_os = "macos") {
            assert_eq!(
                path,
                PathBuf::from("/opt/hoshikage/lib/libhoshikage-llama-adapter.dylib")
            );
        } else {
            assert_eq!(
                path,
                PathBuf::from("/opt/hoshikage/lib/libhoshikage-llama-adapter.so")
            );
        }
    }
}
