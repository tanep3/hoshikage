use crate::config::Config;
use crate::error::Result;
use crate::inference::{CapabilityStatus, LlamaServerRuntimeReport, RuntimeCapabilityReport};
use crate::model::{FallbackMode, ModelConfig};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DiagnosticStatus {
    Ok,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticCheck {
    id: String,
    status: DiagnosticStatus,
    message: String,
    remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticSummary {
    status: DiagnosticStatus,
    ok: usize,
    warn: usize,
    error: usize,
}

#[derive(Debug, Clone, Serialize)]
struct DoctorReport {
    summary: DiagnosticSummary,
    llama_cpp_runtime_dir: String,
    llama_server_path: String,
    libllama_path: String,
    model: Option<String>,
    checks: Vec<DiagnosticCheck>,
}

pub async fn doctor(model: Option<String>, json: bool) -> Result<()> {
    let config = Config::load()?;
    let server_report = LlamaServerRuntimeReport::probe(config.llama_cpp_runtime_dir()?);
    let libllama_path = resolve_runtime_libllama_path(&config, &server_report)?;
    let mut checks = Vec::new();

    checks.extend(llama_server_runtime_checks(&server_report));

    let runtime_report = match RuntimeCapabilityReport::probe(&libllama_path) {
        Ok(report) => {
            checks.extend(runtime_checks(&report));
            Some(report)
        }
        Err(e) => {
            checks.push(DiagnosticCheck {
                id: "runtime.libllama.load".to_string(),
                status: DiagnosticStatus::Error,
                message: format!("libllama を読み込めません: {}", e),
                remediation: Some(
                    "HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR または HOSHIKAGE_LIB_PATH の配置を確認してください"
                        .to_string(),
                ),
            });
            None
        }
    };

    if let Some(model_name) = &model {
        checks.extend(model_checks(&config, model_name, runtime_report.as_ref())?);
    }

    let report = DoctorReport {
        summary: summarize(&checks),
        llama_cpp_runtime_dir: server_report.runtime_dir.display().to_string(),
        llama_server_path: server_report.llama_server_path.display().to_string(),
        libllama_path: libllama_path.display().to_string(),
        model,
        checks,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }

    Ok(())
}

pub fn check_candidate_model(model_name: &str, model_config: &ModelConfig) -> Result<bool> {
    let config = Config::load()?;
    let server_report = LlamaServerRuntimeReport::probe(config.llama_cpp_runtime_dir()?);
    let libllama_path = resolve_runtime_libllama_path(&config, &server_report)?;
    let runtime_report = RuntimeCapabilityReport::probe(&libllama_path).ok();
    let mut checks = Vec::new();

    checks.extend(llama_server_runtime_checks(&server_report));

    if let Some(report) = &runtime_report {
        checks.extend(runtime_checks(report));
    } else {
        checks.push(DiagnosticCheck {
            id: "runtime.libllama.load".to_string(),
            status: DiagnosticStatus::Error,
            message: format!("libllama を読み込めません: {}", libllama_path.display()),
            remediation: Some(
                "HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR または HOSHIKAGE_LIB_PATH の配置を確認してください"
                    .to_string(),
            ),
        });
    }
    checks.extend(model_config_checks(model_config, runtime_report.as_ref()));

    let report = DoctorReport {
        summary: summarize(&checks),
        llama_cpp_runtime_dir: server_report.runtime_dir.display().to_string(),
        llama_server_path: server_report.llama_server_path.display().to_string(),
        libllama_path: libllama_path.display().to_string(),
        model: Some(model_name.to_string()),
        checks,
    };
    print_report(&report);

    Ok(report.summary.error == 0)
}

fn resolve_runtime_libllama_path(
    config: &Config,
    server_report: &LlamaServerRuntimeReport,
) -> Result<PathBuf> {
    if config.lib_path.is_some() {
        return config.resolve_lib_path();
    }

    Ok(server_report.libllama_path.clone())
}

fn llama_server_runtime_checks(report: &LlamaServerRuntimeReport) -> Vec<DiagnosticCheck> {
    vec![
        directory_check(
            "runtime.llama_cpp.dir",
            &report.runtime_dir,
            "llama.cpp runtime directory が見つかります",
            "llama.cpp runtime directory が見つかりません",
        ),
        file_check(
            "runtime.llama_server.path",
            &report.llama_server_path,
            "llama-server が見つかります",
            "llama-server が見つかりません",
        ),
        file_check(
            "runtime.llama_cli.path",
            &report.llama_cli_path,
            "llama-cli が見つかります",
            "llama-cli が見つかりません",
        ),
        file_check(
            "runtime.managed_libllama.path",
            &report.libllama_path,
            "managed runtime の libllama が見つかります",
            "managed runtime の libllama が見つかりません",
        ),
    ]
}

fn runtime_checks(report: &RuntimeCapabilityReport) -> Vec<DiagnosticCheck> {
    let mut checks = vec![
        capability_check(
            "runtime.core",
            "libllama core C API",
            report.core.status,
            "libllama を最新の Hoshikage 用ビルドに差し替えてください",
        ),
        capability_check(
            "runtime.vision",
            "Vision C API",
            report.vision.status,
            "libmtmd を含む llama.cpp runtime bundle を配置してください",
        ),
        capability_check(
            "runtime.speculation",
            "MTP / Draft model runtime",
            report.speculation.status,
            "C-compatible adapter 実装フェーズで接続します",
        ),
        capability_check(
            "runtime.thinking",
            "Thinking mode control",
            report.thinking_control.status,
            "C-compatible adapter 実装フェーズで runtime budget 制御を接続します",
        ),
    ];

    checks.push(file_check(
        "runtime.libllama.path",
        &report.libllama_path,
        "libllama が見つかります",
        "libllama が見つかりません",
    ));

    if let Some(lib_dir) = report.libllama_path.parent() {
        checks.push(optional_backend_file_check(
            "runtime.cuda_backend.path",
            &lib_dir.join(platform_library_name("ggml-cuda")),
            "CUDA backend shared library が見つかります",
            "CUDA backend shared library が見つかりません",
        ));
    }

    if let Some(path) = &report.libmtmd_path {
        checks.push(file_check(
            "runtime.libmtmd.path",
            path,
            "libmtmd が見つかります",
            "libmtmd が見つかりません",
        ));
    }
    if let Some(path) = &report.libllama_common_path {
        checks.push(file_check(
            "runtime.libllama_common.path",
            path,
            "libllama-common が見つかります",
            "libllama-common が見つかりません",
        ));
    }
    if let Some(path) = &report.speculation_adapter_path {
        checks.push(optional_adapter_file_check(path));
    }

    checks
}

fn optional_adapter_file_check(path: &Path) -> DiagnosticCheck {
    if path.is_file() {
        DiagnosticCheck {
            id: "runtime.speculation_adapter.path".to_string(),
            status: DiagnosticStatus::Ok,
            message: format!(
                "Hoshikage speculation adapter が見つかります: {}",
                path.display()
            ),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: "runtime.speculation_adapter.path".to_string(),
            status: DiagnosticStatus::Warn,
            message: format!(
                "Hoshikage speculation adapter が見つかりません: {}",
                path.display()
            ),
            remediation: Some(
                "MTP / Draft の実速度化には Hoshikage 用 adapter を配置してください".to_string(),
            ),
        }
    }
}

fn model_checks(
    config: &Config,
    model_name: &str,
    runtime_report: Option<&RuntimeCapabilityReport>,
) -> Result<Vec<DiagnosticCheck>> {
    let mut checks = Vec::new();
    let model_map_path = config.model_map_path()?;

    if !model_map_path.exists() {
        checks.push(DiagnosticCheck {
            id: "model.map.exists".to_string(),
            status: DiagnosticStatus::Error,
            message: format!(
                "model_map.json が見つかりません: {}",
                model_map_path.display()
            ),
            remediation: Some("hoshikage add でモデルを登録してください".to_string()),
        });
        return Ok(checks);
    }

    let content = std::fs::read_to_string(&model_map_path)?;
    let models: HashMap<String, ModelConfig> = serde_json::from_str(&content)?;
    let Some(model_config) = models.get(model_name) else {
        checks.push(DiagnosticCheck {
            id: "model.registered".to_string(),
            status: DiagnosticStatus::Error,
            message: format!("モデル '{}' は登録されていません", model_name),
            remediation: Some(
                "hoshikage list --details で登録済みモデルを確認してください".to_string(),
            ),
        });
        return Ok(checks);
    };

    checks.extend(model_config_checks(model_config, runtime_report));

    Ok(checks)
}

fn model_config_checks(
    model_config: &ModelConfig,
    runtime_report: Option<&RuntimeCapabilityReport>,
) -> Vec<DiagnosticCheck> {
    let mut checks = vec![file_check(
        "model.main.exists",
        &model_config.main_model_path(),
        "main model が見つかります",
        "main model が見つかりません",
    )];

    if let Some(mmproj) = &model_config.mmproj {
        checks.push(file_check(
            "model.mmproj.exists",
            &resolve_bundle_path(&model_config.path, mmproj),
            "mmproj が見つかります",
            "mmproj が見つかりません",
        ));
        checks.push(vision_compat_check(runtime_report));
    }

    if let Some(drafter) = &model_config.drafter {
        checks.push(file_check(
            "model.drafter.exists",
            &resolve_bundle_path(&model_config.path, drafter),
            "drafter が見つかります",
            "drafter が見つかりません",
        ));
    }

    checks.push(speculation_compat_check(model_config, runtime_report));
    checks.push(thinking_compat_check(model_config, runtime_report));

    checks
}

fn capability_check(
    id: &str,
    label: &str,
    status: CapabilityStatus,
    remediation: &str,
) -> DiagnosticCheck {
    let diagnostic_status = match status {
        CapabilityStatus::Available => DiagnosticStatus::Ok,
        CapabilityStatus::Missing => DiagnosticStatus::Error,
        CapabilityStatus::AdapterRequired => DiagnosticStatus::Warn,
    };

    let message = match status {
        CapabilityStatus::Available => format!("{} は利用できます", label),
        CapabilityStatus::Missing => format!("{} は利用できません", label),
        CapabilityStatus::AdapterRequired => {
            format!(
                "{} は runtime 側に存在しますが adapter 接続が必要です",
                label
            )
        }
    };

    DiagnosticCheck {
        id: id.to_string(),
        status: diagnostic_status,
        message,
        remediation: (diagnostic_status != DiagnosticStatus::Ok).then(|| remediation.to_string()),
    }
}

fn file_check(id: &str, path: &Path, ok_message: &str, err_message: &str) -> DiagnosticCheck {
    if path.is_file() {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Ok,
            message: format!("{}: {}", ok_message, path.display()),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Error,
            message: format!("{}: {}", err_message, path.display()),
            remediation: Some("登録内容または runtime 配置先のパスを確認してください".to_string()),
        }
    }
}

fn directory_check(id: &str, path: &Path, ok_message: &str, err_message: &str) -> DiagnosticCheck {
    if path.is_dir() {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Ok,
            message: format!("{}: {}", ok_message, path.display()),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Error,
            message: format!("{}: {}", err_message, path.display()),
            remediation: Some(
                "llama.cpp runtime bundle を ~/.config/hoshikage/llama.cpp に配置してください"
                    .to_string(),
            ),
        }
    }
}

fn optional_backend_file_check(
    id: &str,
    path: &Path,
    ok_message: &str,
    warn_message: &str,
) -> DiagnosticCheck {
    if path.is_file() {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Ok,
            message: format!("{}: {}", ok_message, path.display()),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Warn,
            message: format!("{}: {}", warn_message, path.display()),
            remediation: Some(
                "CUDA 版 llama.cpp runtime bundle の配置を確認してください".to_string(),
            ),
        }
    }
}

fn vision_compat_check(runtime_report: Option<&RuntimeCapabilityReport>) -> DiagnosticCheck {
    match runtime_report.map(|report| report.vision.status) {
        Some(CapabilityStatus::Available) => DiagnosticCheck {
            id: "model.vision.compat".to_string(),
            status: DiagnosticStatus::Ok,
            message: "mmproj 設定と Vision runtime は整合しています".to_string(),
            remediation: None,
        },
        _ => DiagnosticCheck {
            id: "model.vision.compat".to_string(),
            status: DiagnosticStatus::Error,
            message: "mmproj が設定されていますが Vision runtime が利用できません".to_string(),
            remediation: Some(
                "libmtmd を含む llama.cpp runtime bundle を配置してください".to_string(),
            ),
        },
    }
}

fn speculation_compat_check(
    model_config: &ModelConfig,
    runtime_report: Option<&RuntimeCapabilityReport>,
) -> DiagnosticCheck {
    if model_config.speculation.is_off() {
        return DiagnosticCheck {
            id: "model.speculation.compat".to_string(),
            status: DiagnosticStatus::Ok,
            message: "speculation は off です".to_string(),
            remediation: None,
        };
    }

    let available = runtime_report
        .map(|report| report.speculation.status == CapabilityStatus::Available)
        .unwrap_or(false);
    if available {
        return DiagnosticCheck {
            id: "model.speculation.compat".to_string(),
            status: DiagnosticStatus::Ok,
            message: "speculation 設定と runtime は整合しています".to_string(),
            remediation: None,
        };
    }

    let strict = model_config.speculation.fallback == FallbackMode::Strict;
    DiagnosticCheck {
        id: "model.speculation.compat".to_string(),
        status: if strict {
            DiagnosticStatus::Error
        } else {
            DiagnosticStatus::Warn
        },
        message: format!(
            "{:?} が設定されていますが runtime adapter が未接続です",
            model_config.speculation.modes
        ),
        remediation: Some(if strict {
            "fallback を warn にするか、adapter 実装後に有効化してください".to_string()
        } else {
            "fallback 方針に従い通常推論へ戻します".to_string()
        }),
    }
}

fn thinking_compat_check(
    model_config: &ModelConfig,
    runtime_report: Option<&RuntimeCapabilityReport>,
) -> DiagnosticCheck {
    if model_config.thinking.mode != crate::model::ThinkingMode::Off {
        return DiagnosticCheck {
            id: "model.thinking.compat".to_string(),
            status: DiagnosticStatus::Ok,
            message: "Thinking mode は auto です".to_string(),
            remediation: None,
        };
    }

    let has_prompt_policy = runtime_report
        .map(|report| report.thinking_control.status != CapabilityStatus::Missing)
        .unwrap_or(false);
    if has_prompt_policy {
        DiagnosticCheck {
            id: "model.thinking.compat".to_string(),
            status: DiagnosticStatus::Warn,
            message:
                "Thinking off は prompt policy と safety filter で適用されます。runtime budget adapter は未接続です"
                    .to_string(),
            remediation: Some("adapter 実装フェーズで runtime budget 制御を接続します".to_string()),
        }
    } else {
        DiagnosticCheck {
            id: "model.thinking.compat".to_string(),
            status: DiagnosticStatus::Error,
            message: "Thinking off の prompt policy を適用できません".to_string(),
            remediation: Some("libllama の chat template API を確認してください".to_string()),
        }
    }
}

fn resolve_bundle_path(base_path: &str, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(base_path).join(path)
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

fn summarize(checks: &[DiagnosticCheck]) -> DiagnosticSummary {
    let ok = checks
        .iter()
        .filter(|check| check.status == DiagnosticStatus::Ok)
        .count();
    let warn = checks
        .iter()
        .filter(|check| check.status == DiagnosticStatus::Warn)
        .count();
    let error = checks
        .iter()
        .filter(|check| check.status == DiagnosticStatus::Error)
        .count();
    let status = if error > 0 {
        DiagnosticStatus::Error
    } else if warn > 0 {
        DiagnosticStatus::Warn
    } else {
        DiagnosticStatus::Ok
    };

    DiagnosticSummary {
        status,
        ok,
        warn,
        error,
    }
}

fn print_report(report: &DoctorReport) {
    println!("Hoshikage doctor");
    println!("================");
    println!("status: {:?}", report.summary.status);
    println!("llama.cpp runtime: {}", report.llama_cpp_runtime_dir);
    println!("llama-server: {}", report.llama_server_path);
    println!("libllama: {}", report.libllama_path);
    if let Some(model) = &report.model {
        println!("model: {}", model);
    }
    println!(
        "checks: ok={}, warn={}, error={}",
        report.summary.ok, report.summary.warn, report.summary.error
    );
    println!();

    for check in &report.checks {
        println!("[{:?}] {} - {}", check.status, check.id, check.message);
        if let Some(remediation) = &check.remediation {
            println!("      next: {}", remediation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SpeculationConfig, SpeculationMode, ThinkingConfig, ThinkingMode};

    #[test]
    fn summary_prefers_error_over_warn() {
        let checks = vec![
            DiagnosticCheck {
                id: "ok".to_string(),
                status: DiagnosticStatus::Ok,
                message: "ok".to_string(),
                remediation: None,
            },
            DiagnosticCheck {
                id: "warn".to_string(),
                status: DiagnosticStatus::Warn,
                message: "warn".to_string(),
                remediation: None,
            },
            DiagnosticCheck {
                id: "error".to_string(),
                status: DiagnosticStatus::Error,
                message: "error".to_string(),
                remediation: None,
            },
        ];

        let summary = summarize(&checks);

        assert_eq!(summary.status, DiagnosticStatus::Error);
        assert_eq!(summary.ok, 1);
        assert_eq!(summary.warn, 1);
        assert_eq!(summary.error, 1);
    }

    #[test]
    fn speculation_strict_missing_runtime_is_error() {
        let config = ModelConfig {
            speculation: SpeculationConfig {
                modes: vec![SpeculationMode::Mtp],
                fallback: FallbackMode::Strict,
            },
            ..ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new())
        };

        let check = speculation_compat_check(&config, None);

        assert_eq!(check.status, DiagnosticStatus::Error);
    }

    #[test]
    fn speculation_warn_missing_runtime_is_warn() {
        let config = ModelConfig {
            speculation: SpeculationConfig {
                modes: vec![SpeculationMode::DraftModel],
                fallback: FallbackMode::Warn,
            },
            ..ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new())
        };

        let check = speculation_compat_check(&config, None);

        assert_eq!(check.status, DiagnosticStatus::Warn);
    }

    #[test]
    fn thinking_auto_is_ok_without_runtime() {
        let config = ModelConfig {
            thinking: ThinkingConfig {
                mode: ThinkingMode::Auto,
            },
            ..ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new())
        };

        let check = thinking_compat_check(&config, None);

        assert_eq!(check.status, DiagnosticStatus::Ok);
    }
}
