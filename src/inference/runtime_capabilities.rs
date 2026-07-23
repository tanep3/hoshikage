use crate::error::{HoshikageError, Result};
use crate::inference::speculation_adapter::adapter_path_for;
use libloading::Library;
use serde::Serialize;
use std::path::{Path, PathBuf};

type HoshikageSpeculationSupports = unsafe extern "C" fn(mode: i32) -> bool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    Available,
    Missing,
    AdapterRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SymbolProbe {
    pub library: String,
    pub symbol: String,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeCapability {
    pub status: CapabilityStatus,
    pub required_symbols: Vec<SymbolProbe>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeCapabilityReport {
    pub libllama_path: PathBuf,
    pub libmtmd_path: Option<PathBuf>,
    pub libllama_common_path: Option<PathBuf>,
    pub speculation_adapter_path: Option<PathBuf>,
    pub core: RuntimeCapability,
    pub vision: RuntimeCapability,
    pub speculation: RuntimeCapability,
    pub thinking_control: RuntimeCapability,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LlamaServerRuntimeReport {
    pub runtime_dir: PathBuf,
    pub llama_server_path: PathBuf,
    pub llama_cli_path: PathBuf,
    pub libllama_path: PathBuf,
    pub llama_server_available: bool,
    pub llama_cli_available: bool,
    pub libllama_available: bool,
}

impl LlamaServerRuntimeReport {
    pub fn probe(runtime_dir: impl Into<PathBuf>) -> Self {
        let runtime_dir = runtime_dir.into();
        let llama_server_path = runtime_dir.join(platform_executable_name("llama-server"));
        let llama_cli_path = runtime_dir.join(platform_executable_name("llama-cli"));
        let libllama_path = runtime_dir.join("lib").join(platform_library_name("llama"));

        Self {
            runtime_dir,
            llama_server_available: llama_server_path.is_file(),
            llama_cli_available: llama_cli_path.is_file(),
            libllama_available: libllama_path.is_file(),
            llama_server_path,
            llama_cli_path,
            libllama_path,
        }
    }
}

impl RuntimeCapabilityReport {
    pub fn probe(libllama_path: impl Into<PathBuf>) -> Result<Self> {
        let libllama_path = libllama_path.into();
        let libllama = load_library(&libllama_path)?;
        let lib_dir = libllama_path.parent().map(Path::to_path_buf);

        let libmtmd_path = lib_dir
            .as_ref()
            .map(|dir| dir.join(platform_library_name("mtmd")));
        let libllama_common_path = lib_dir
            .as_ref()
            .map(|dir| dir.join(platform_library_name("llama-common")));
        let speculation_adapter_path = Some(adapter_path_for(&libllama_path));

        let libmtmd = load_optional_library(libmtmd_path.as_deref());
        let libllama_common = load_optional_library(libllama_common_path.as_deref());
        let speculation_adapter = load_optional_library(speculation_adapter_path.as_deref());

        let core = probe_core(&libllama);
        let vision = probe_vision(libmtmd.as_ref(), libmtmd_path.as_deref());
        let speculation = probe_speculation(
            libllama_common.as_ref(),
            libllama_common_path.as_deref(),
            speculation_adapter.as_ref(),
            speculation_adapter_path.as_deref(),
        );
        let thinking_control = probe_thinking_control(
            &libllama,
            libllama_common.as_ref(),
            libllama_common_path.as_deref(),
        );

        Ok(Self {
            libllama_path,
            libmtmd_path,
            libllama_common_path,
            speculation_adapter_path,
            core,
            vision,
            speculation,
            thinking_control,
        })
    }
}

fn probe_core(libllama: &Library) -> RuntimeCapability {
    capability_from_symbols(
        libllama,
        "libllama",
        &[
            "llama_backend_init",
            "llama_load_model_from_file",
            "llama_model_default_params",
            "llama_context_default_params",
            "llama_init_from_model",
            "llama_decode",
            "llama_chat_apply_template",
            "llama_model_meta_val_str",
        ],
        Vec::new(),
    )
}

fn probe_vision(libmtmd: Option<&Library>, libmtmd_path: Option<&Path>) -> RuntimeCapability {
    let Some(libmtmd) = libmtmd else {
        return RuntimeCapability {
            status: CapabilityStatus::Missing,
            required_symbols: Vec::new(),
            notes: vec![format!(
                "{} could not be loaded",
                libmtmd_path
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "libmtmd".to_string())
            )],
        };
    };

    capability_from_symbols(
        libmtmd,
        "libmtmd",
        &[
            "mtmd_context_params_default",
            "mtmd_init_from_file",
            "mtmd_support_vision",
            "mtmd_bitmap_init",
            "mtmd_tokenize",
            "mtmd_encode_chunk",
            "mtmd_get_output_embd",
            "mtmd_free",
        ],
        vec!["Vision can be implemented through the libmtmd C API.".to_string()],
    )
}

fn probe_speculation(
    libllama_common: Option<&Library>,
    libllama_common_path: Option<&Path>,
    speculation_adapter: Option<&Library>,
    speculation_adapter_path: Option<&Path>,
) -> RuntimeCapability {
    let Some(libllama_common) = libllama_common else {
        return RuntimeCapability {
            status: CapabilityStatus::Missing,
            required_symbols: Vec::new(),
            notes: vec![format!(
                "{} could not be loaded",
                libllama_common_path
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "libllama-common".to_string())
            )],
        };
    };

    let common_capability = capability_from_symbols(
        libllama_common,
        "libllama-common",
        &[
            "_Z32common_speculative_all_types_strv",
            "_Z23common_speculative_initR25common_params_speculativej",
            "_Z24common_speculative_draftP18common_speculative",
            "_Z25common_speculative_acceptP18common_speculativeit",
            "_Z26common_speculative_processP18common_speculativeRK11llama_batch",
            "_Z23common_speculative_freeP18common_speculative",
        ],
        vec![
            "Speculative decoding exists in llama.cpp common code, including MTP support."
                .to_string(),
            "These symbols are C++ ABI, so Rust should not call them directly without a small C-compatible adapter."
                .to_string(),
        ],
    );

    if common_capability.status == CapabilityStatus::Missing {
        return common_capability;
    }

    let Some(speculation_adapter) = speculation_adapter else {
        let mut capability = common_capability;
        capability.status = CapabilityStatus::AdapterRequired;
        capability.notes.push(format!(
            "{} could not be loaded",
            speculation_adapter_path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "Hoshikage speculation adapter".to_string())
        ));
        return capability;
    };

    let adapter_capability = capability_from_symbols(
        speculation_adapter,
        "hoshikage-llama-adapter",
        &[
            "hoshikage_adapter_abi_version",
            "hoshikage_speculation_supports",
            "hoshikage_speculation_init",
            "hoshikage_speculation_free",
            "hoshikage_speculation_draft",
            "hoshikage_speculation_process",
            "hoshikage_speculation_accept",
        ],
        vec![
            "Hoshikage speculation adapter exposes a C-compatible ABI.".to_string(),
            "MTP / Draft generation can be connected through the adapter boundary.".to_string(),
        ],
    );

    let mut symbols = common_capability.required_symbols;
    symbols.extend(adapter_capability.required_symbols);
    let mut notes = common_capability.notes;
    notes.extend(adapter_capability.notes);

    let adapter_support = probe_adapter_support(speculation_adapter);
    notes.extend(adapter_support.notes);

    let status = if adapter_capability.status == CapabilityStatus::Missing {
        CapabilityStatus::AdapterRequired
    } else if adapter_support.mtp || adapter_support.draft_model {
        CapabilityStatus::Available
    } else {
        CapabilityStatus::AdapterRequired
    };

    RuntimeCapability {
        status,
        required_symbols: symbols,
        notes,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdapterSupportProbe {
    mtp: bool,
    draft_model: bool,
    notes: Vec<String>,
}

fn probe_adapter_support(speculation_adapter: &Library) -> AdapterSupportProbe {
    let supports = unsafe {
        speculation_adapter
            .get::<HoshikageSpeculationSupports>(b"hoshikage_speculation_supports")
            .ok()
    };

    let Some(supports) = supports else {
        return AdapterSupportProbe {
            mtp: false,
            draft_model: false,
            notes: vec![
                "Hoshikage speculation adapter support probe is not available.".to_string(),
            ],
        };
    };

    let mtp = unsafe { supports(1) };
    let draft_model = unsafe { supports(2) };

    let mut notes = Vec::new();
    if mtp {
        notes.push("Hoshikage speculation adapter reports MTP support.".to_string());
    }
    if draft_model {
        notes.push("Hoshikage speculation adapter reports Draft model support.".to_string());
    }
    if !mtp && !draft_model {
        notes.push(
            "Hoshikage speculation adapter is loadable, but no speculation mode is enabled yet."
                .to_string(),
        );
    }

    AdapterSupportProbe {
        mtp,
        draft_model,
        notes,
    }
}

fn probe_thinking_control(
    libllama: &Library,
    libllama_common: Option<&Library>,
    libllama_common_path: Option<&Path>,
) -> RuntimeCapability {
    let mut symbols = vec![probe_symbol(
        libllama,
        "libllama",
        "llama_chat_apply_template",
    )];
    let mut notes = vec![
        "Chat template application is available through libllama.".to_string(),
        "Output filtering can be implemented in Hoshikage regardless of llama.cpp common helpers."
            .to_string(),
    ];

    if let Some(libllama_common) = libllama_common {
        symbols.push(probe_symbol(
            libllama_common,
            "libllama-common",
            "_Z37common_sampler_reasoning_budget_forceP14common_sampler",
        ));
        symbols.push(probe_symbol(
            libllama_common,
            "libllama-common",
            "_Z33common_reasoning_format_from_nameRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
        ));
        notes.push(
            "llama.cpp common provides reasoning control helpers, but they are C++ ABI."
                .to_string(),
        );
    } else {
        notes.push(format!(
            "{} could not be loaded",
            libllama_common_path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "libllama-common".to_string())
        ));
    }

    let status = if symbols
        .first()
        .map(|symbol| symbol.available)
        .unwrap_or(false)
    {
        CapabilityStatus::AdapterRequired
    } else {
        CapabilityStatus::Missing
    };

    RuntimeCapability {
        status,
        required_symbols: symbols,
        notes,
    }
}

fn capability_from_symbols(
    library: &Library,
    library_name: &str,
    symbols: &[&str],
    notes: Vec<String>,
) -> RuntimeCapability {
    let required_symbols: Vec<SymbolProbe> = symbols
        .iter()
        .map(|symbol| probe_symbol(library, library_name, symbol))
        .collect();
    let status = if required_symbols.iter().all(|symbol| symbol.available) {
        CapabilityStatus::Available
    } else {
        CapabilityStatus::Missing
    };

    RuntimeCapability {
        status,
        required_symbols,
        notes,
    }
}

fn probe_symbol(library: &Library, library_name: &str, symbol: &str) -> SymbolProbe {
    let available = unsafe { library.get::<*const ()>(symbol.as_bytes()).is_ok() };
    SymbolProbe {
        library: library_name.to_string(),
        symbol: symbol.to_string(),
        available,
    }
}

fn load_library(path: &Path) -> Result<Library> {
    unsafe {
        Library::new(path).map_err(|err| {
            HoshikageError::LibraryLoadError(format!("Failed to load {}: {}", path.display(), err))
        })
    }
}

fn load_optional_library(path: Option<&Path>) -> Option<Library> {
    path.and_then(|path| unsafe { Library::new(path).ok() })
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

fn platform_executable_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_library_name_uses_current_platform() {
        let lib_name = platform_library_name("llama");
        if cfg!(target_os = "windows") {
            assert_eq!(lib_name, "llama.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(lib_name, "libllama.dylib");
        } else {
            assert_eq!(lib_name, "libllama.so");
        }
    }

    #[test]
    fn llama_server_runtime_report_uses_standard_layout() {
        let report = LlamaServerRuntimeReport::probe(PathBuf::from("/runtime"));

        assert_eq!(report.runtime_dir, PathBuf::from("/runtime"));
        if cfg!(target_os = "windows") {
            assert_eq!(
                report.llama_server_path,
                PathBuf::from("/runtime/llama-server.exe")
            );
            assert_eq!(
                report.llama_cli_path,
                PathBuf::from("/runtime/llama-cli.exe")
            );
        } else {
            assert_eq!(
                report.llama_server_path,
                PathBuf::from("/runtime/llama-server")
            );
            assert_eq!(report.llama_cli_path, PathBuf::from("/runtime/llama-cli"));
        }
    }

    #[test]
    fn missing_required_symbol_marks_capability_missing() {
        let lib_path = if cfg!(target_os = "linux") {
            PathBuf::from("llama_cpp_local/lib/libllama.so")
        } else {
            return;
        };

        if !lib_path.exists() {
            return;
        }

        let lib = match load_library(&lib_path) {
            Ok(lib) => lib,
            Err(_) => return,
        };
        let capability = capability_from_symbols(
            &lib,
            "libllama",
            &["llama_backend_init", "symbol_that_should_not_exist"],
            Vec::new(),
        );

        assert_eq!(capability.status, CapabilityStatus::Missing);
        assert!(capability
            .required_symbols
            .iter()
            .any(|symbol| symbol.symbol == "symbol_that_should_not_exist" && !symbol.available));
    }

    #[test]
    #[ignore]
    fn probe_local_llama_cpp_bundle() {
        let lib_path = PathBuf::from("llama_cpp_local/lib/libllama.so");
        let report = RuntimeCapabilityReport::probe(lib_path).expect("probe local llama.cpp");

        assert_eq!(report.core.status, CapabilityStatus::Available);
        assert_eq!(report.vision.status, CapabilityStatus::Available);
        assert_eq!(report.speculation.status, CapabilityStatus::AdapterRequired);
        assert_eq!(
            report.thinking_control.status,
            CapabilityStatus::AdapterRequired
        );
    }
}
