use crate::error::Result;
use crate::ffi::{ggml_backend_sched_eval_callback, llama_flash_attn_type, llama_model};
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};

#[repr(C)]
pub struct mtmd_context {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mtmd_context_params {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: llama_flash_attn_type,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: ggml_backend_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,
    pub batch_max_tokens: i32,
    pub progress_callback:
        Option<unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
}

type MtmdContextParamsDefault = unsafe extern "C" fn() -> mtmd_context_params;
type MtmdInitFromFile = unsafe extern "C" fn(
    mmproj_fname: *const c_char,
    text_model: *const llama_model,
    ctx_params: mtmd_context_params,
) -> *mut mtmd_context;
type MtmdFree = unsafe extern "C" fn(ctx: *mut mtmd_context);
type MtmdSupportVision = unsafe extern "C" fn(ctx: *const mtmd_context) -> bool;
type MtmdGetMarker = unsafe extern "C" fn(ctx: *const mtmd_context) -> *const c_char;

pub struct VisionRuntime {
    library: Library,
    context: *mut mtmd_context,
    mmproj_path: PathBuf,
    supports_vision: bool,
    marker: String,
}

unsafe impl Send for VisionRuntime {}

impl VisionRuntime {
    pub fn load(
        libllama_path: &Path,
        mmproj_path: &Path,
        text_model: *const llama_model,
    ) -> Result<Self> {
        let libmtmd_path = libllama_path
            .parent()
            .map(|dir| dir.join(platform_library_name("mtmd")))
            .unwrap_or_else(|| PathBuf::from(platform_library_name("mtmd")));
        let library = unsafe {
            Library::new(&libmtmd_path).map_err(|e| {
                crate::error::HoshikageError::LibraryLoadError(format!(
                    "Failed to load {}: {}",
                    libmtmd_path.display(),
                    e
                ))
            })?
        };

        let mmproj = CString::new(mmproj_path.to_string_lossy().as_bytes())?;

        let context = unsafe {
            let params_default: Symbol<MtmdContextParamsDefault> =
                library.get(b"mtmd_context_params_default").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol mtmd_context_params_default: {}",
                        e
                    ))
                })?;
            let init_from_file: Symbol<MtmdInitFromFile> =
                library.get(b"mtmd_init_from_file").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol mtmd_init_from_file: {}",
                        e
                    ))
                })?;

            let mut params = params_default();
            params.use_gpu = true;
            let context = init_from_file(mmproj.as_ptr(), text_model, params);
            if context.is_null() {
                return Err(crate::error::HoshikageError::ModelLoadFailed(format!(
                    "mmproj load returned null: {}",
                    mmproj_path.display()
                )));
            }
            context
        };

        let supports_vision = unsafe {
            let support_vision: Symbol<MtmdSupportVision> =
                library.get(b"mtmd_support_vision").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol mtmd_support_vision: {}",
                        e
                    ))
                })?;
            support_vision(context)
        };

        let marker = unsafe {
            let get_marker: Symbol<MtmdGetMarker> =
                library.get(b"mtmd_get_marker").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol mtmd_get_marker: {}",
                        e
                    ))
                })?;
            let marker = get_marker(context);
            if marker.is_null() {
                String::new()
            } else {
                CStr::from_ptr(marker).to_string_lossy().to_string()
            }
        };

        if !supports_vision {
            unsafe {
                let free: Symbol<MtmdFree> = library.get(b"mtmd_free").map_err(|e| {
                    crate::error::HoshikageError::LibraryLoadError(format!(
                        "Failed to get symbol mtmd_free: {}",
                        e
                    ))
                })?;
                free(context);
            }
            return Err(crate::error::HoshikageError::ModelLoadFailed(format!(
                "mmproj does not support vision input: {}",
                mmproj_path.display()
            )));
        }

        Ok(Self {
            library,
            context,
            mmproj_path: mmproj_path.to_path_buf(),
            supports_vision,
            marker,
        })
    }

    pub fn mmproj_path(&self) -> &Path {
        &self.mmproj_path
    }

    pub fn supports_vision(&self) -> bool {
        self.supports_vision
    }

    pub fn marker(&self) -> &str {
        &self.marker
    }
}

impl Drop for VisionRuntime {
    fn drop(&mut self) {
        if self.context.is_null() {
            return;
        }

        unsafe {
            if let Ok(free) = self.library.get::<MtmdFree>(b"mtmd_free") {
                free(self.context);
            }
        }
        self.context = std::ptr::null_mut();
    }
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
    fn mtmd_context_params_layout_matches_current_header() {
        assert_eq!(std::mem::size_of::<mtmd_context_params>(), 80);
        assert_eq!(std::mem::align_of::<mtmd_context_params>(), 8);
    }

    #[test]
    fn platform_library_name_uses_current_platform() {
        let lib_name = platform_library_name("mtmd");
        if cfg!(target_os = "windows") {
            assert_eq!(lib_name, "mtmd.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(lib_name, "libmtmd.dylib");
        } else {
            assert_eq!(lib_name, "libmtmd.so");
        }
    }
}
