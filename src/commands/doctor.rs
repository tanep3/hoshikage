use crate::config::Config;
use crate::error::Result;
use crate::i18n::{Language, LocalizedText};
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

#[derive(Debug, Clone)]
struct DiagnosticCheck {
    id: String,
    status: DiagnosticStatus,
    message: LocalizedText,
    remediation: Option<LocalizedText>,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticSummary {
    status: DiagnosticStatus,
    ok: usize,
    warn: usize,
    error: usize,
}

#[derive(Debug, Clone)]
struct DoctorReport {
    summary: DiagnosticSummary,
    llama_cpp_runtime_dir: String,
    llama_server_path: String,
    libllama_path: String,
    model: Option<String>,
    checks: Vec<DiagnosticCheck>,
}

#[derive(Serialize)]
struct DiagnosticCheckWire<'a> {
    id: &'a str,
    status: DiagnosticStatus,
    message_key: String,
    remediation_key: Option<String>,
}

#[derive(Serialize)]
struct DoctorReportWire<'a> {
    summary: &'a DiagnosticSummary,
    llama_cpp_runtime_dir: &'a str,
    llama_server_path: &'a str,
    libllama_path: &'a str,
    model: Option<&'a str>,
    checks: Vec<DiagnosticCheckWire<'a>>,
}

impl DoctorReport {
    fn wire(&self) -> DoctorReportWire<'_> {
        DoctorReportWire {
            summary: &self.summary,
            llama_cpp_runtime_dir: &self.llama_cpp_runtime_dir,
            llama_server_path: &self.llama_server_path,
            libllama_path: &self.libllama_path,
            model: self.model.as_deref(),
            checks: self
                .checks
                .iter()
                .map(|check| DiagnosticCheckWire {
                    id: &check.id,
                    status: check.status,
                    message_key: format!("doctor.{}.message", check.id),
                    remediation_key: check
                        .remediation
                        .as_ref()
                        .map(|_| format!("doctor.{}.remediation", check.id)),
                })
                .collect(),
        }
    }
}

fn text(en: impl Into<String>, ja: impl Into<String>) -> LocalizedText {
    LocalizedText::new(en, ja)
}

#[derive(Debug, Clone)]
struct CodexEndpointSnapshot {
    model_found: bool,
    responses: bool,
    streaming: bool,
    tools: bool,
    codex_compatible: bool,
}

fn codex_capability_checks(snapshot: &CodexEndpointSnapshot) -> Vec<DiagnosticCheck> {
    let dependent_status = |available: bool, optional: bool| {
        if !snapshot.model_found {
            DiagnosticStatus::Error
        } else if available {
            DiagnosticStatus::Ok
        } else if optional {
            DiagnosticStatus::Warn
        } else {
            DiagnosticStatus::Error
        }
    };
    vec![
        DiagnosticCheck {
            id: "codex.connection".to_string(),
            status: DiagnosticStatus::Ok,
            message: text(
                "The Hoshikage API is reachable",
                "Hoshikage APIへ接続できます",
            ),
            remediation: None,
        },
        DiagnosticCheck {
            id: "codex.model".to_string(),
            status: if snapshot.model_found {
                DiagnosticStatus::Ok
            } else {
                DiagnosticStatus::Error
            },
            message: if snapshot.model_found {
                text(
                    "The requested model is registered",
                    "指定モデルが登録されています",
                )
            } else {
                text(
                    "The requested model is not registered",
                    "指定モデルが登録されていません",
                )
            },
            remediation: (!snapshot.model_found).then(|| {
                text(
                    "Run codex-model-catalog to list available models",
                    "codex-model-catalogで利用可能なモデルを確認してください",
                )
            }),
        },
        DiagnosticCheck {
            id: "codex.responses".to_string(),
            status: dependent_status(snapshot.responses, false),
            message: text(
                "Responses API capability was checked",
                "Responses API capabilityを確認しました",
            ),
            remediation: (!snapshot.responses).then(|| {
                text(
                    "Use a Responses-capable bundle with managed llama-server",
                    "Responses対応Bundleとmanaged llama-serverを使用してください",
                )
            }),
        },
        DiagnosticCheck {
            id: "codex.streaming".to_string(),
            status: dependent_status(snapshot.streaming, false),
            message: text(
                "Responses streaming capability was checked",
                "Responses streaming capabilityを確認しました",
            ),
            remediation: (!snapshot.streaming).then(|| {
                text(
                    "Use a Hoshikage version that supports streaming",
                    "streaming対応のHoshikage versionを使用してください",
                )
            }),
        },
        DiagnosticCheck {
            id: "codex.tools".to_string(),
            status: dependent_status(snapshot.tools, true),
            message: if snapshot.tools {
                text("Tool Calling is enabled", "Tool Callingが有効です")
            } else {
                text(
                    "Tool Calling is disabled; only text responses are available",
                    "Tool Callingが無効なためテキスト応答だけ利用できます",
                )
            },
            remediation: (!snapshot.tools).then(|| {
                text(
                    "Check the bundle tool_calling configuration",
                    "Bundleのtool_calling設定を診断してください",
                )
            }),
        },
        DiagnosticCheck {
            id: "codex.context".to_string(),
            status: dependent_status(snapshot.codex_compatible, true),
            message: if snapshot.codex_compatible {
                text(
                    "The model meets the minimum Codex context size",
                    "Codex用context下限を満たしています",
                )
            } else {
                text(
                    "The model does not meet the 16K minimum Codex context size",
                    "Codex用context下限16Kを満たしていません",
                )
            },
            remediation: (!snapshot.codex_compatible).then(|| {
                text(
                    "Set the bundle n_ctx to at least 16384",
                    "Bundleのn_ctxを16384以上に設定してください",
                )
            }),
        },
    ]
}

pub async fn doctor(
    model: Option<String>,
    json: bool,
    codex_base_url: Option<String>,
    language: Language,
) -> Result<()> {
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
                message: text(
                    format!("Could not load libllama: {e}"),
                    format!("libllama を読み込めません: {e}"),
                ),
                remediation: Some(text(
                    "Check HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR or HOSHIKAGE_LIB_PATH",
                    "HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR または HOSHIKAGE_LIB_PATH の配置を確認してください",
                )),
            });
            None
        }
    };

    if let Some(model_name) = &model {
        checks.extend(model_checks(&config, model_name, runtime_report.as_ref())?);
        if let Some(base_url) = &codex_base_url {
            checks.extend(codex_endpoint_checks(base_url, model_name).await);
        }
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
        println!("{}", serde_json::to_string_pretty(&report.wire())?);
    } else {
        print_report(&report, language);
    }

    Ok(())
}

async fn codex_endpoint_checks(base_url: &str, model: &str) -> Vec<DiagnosticCheck> {
    let Ok(base_url) = reqwest::Url::parse(base_url) else {
        return vec![codex_connection_error(
            text(
                "The Codex provider base URL is invalid",
                "Codex provider base URLが正しくありません",
            ),
            text(
                "Specify an absolute URL such as http://127.0.0.1:3030/v1",
                "http://127.0.0.1:3030/v1のような絶対URLを指定してください",
            ),
        )];
    };
    if !matches!(base_url.scheme(), "http" | "https") || base_url.host_str().is_none() {
        return vec![codex_connection_error(
            text(
                "The Codex provider base URL is invalid",
                "Codex provider base URLが正しくありません",
            ),
            text(
                "Specify an absolute URL such as http://127.0.0.1:3030/v1",
                "http://127.0.0.1:3030/v1のような絶対URLを指定してください",
            ),
        )];
    }
    let client = reqwest::Client::new();
    let mut health_url = base_url.clone();
    health_url.set_path("/health");
    health_url.set_query(None);
    health_url.set_fragment(None);
    match client.get(health_url).send().await {
        Ok(response) if response.status().is_success() => {}
        Ok(response) => {
            return vec![codex_connection_error(
                text(
                    format!(
                        "The Hoshikage health check returned HTTP {}",
                        response.status()
                    ),
                    format!(
                        "Hoshikage health checkがHTTP {}を返しました",
                        response.status()
                    ),
                ),
                text(
                    "Check the Hoshikage bind address, port, and process state",
                    "Hoshikageのbind address、port、process状態を確認してください",
                ),
            )];
        }
        Err(_) => {
            return vec![codex_connection_error(
                text(
                    "Could not connect to the Hoshikage API",
                    "Hoshikage APIへ接続できません",
                ),
                text(
                    "Check that Hoshikage is running and verify its bind address, port, and firewall",
                    "Hoshikageの起動、bind address、port、firewallを確認してください",
                ),
            )];
        }
    }

    let mut model_url = base_url;
    if !model_url.path().ends_with('/') {
        let path = format!("{}/", model_url.path());
        model_url.set_path(&path);
    }
    let Ok(mut segments) = model_url.path_segments_mut() else {
        return vec![codex_connection_error(
            text(
                "Could not append the API path to the Codex provider base URL",
                "Codex provider base URLへAPI pathを追加できません",
            ),
            text(
                "Use a base URL in the form http://HOST:PORT/v1",
                "base URLをhttp://HOST:PORT/v1形式で指定してください",
            ),
        )];
    };
    segments.pop_if_empty();
    segments.push("hoshikage");
    segments.push("models");
    segments.push(model);
    drop(segments);

    let mut request = client.get(model_url);
    if let Ok(token) = std::env::var("HOSHIKAGE_API_KEY") {
        if !token.is_empty() {
            request = request.bearer_auth(token);
        }
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(_) => {
            return vec![codex_connection_error(
                text(
                    "Could not connect to the model capability API",
                    "モデル能力APIへ接続できません",
                ),
                text(
                    "Check the base URL and network",
                    "base URLとnetworkを確認してください",
                ),
            )];
        }
    };
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return vec![codex_connection_error(
            text(
                "Authentication to the model capability API failed",
                "モデル能力APIの認証に失敗しました",
            ),
            text(
                "Set a named token in HOSHIKAGE_API_KEY",
                "HOSHIKAGE_API_KEYへ用途名付きTokenを設定してください",
            ),
        )];
    }
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return codex_capability_checks(&CodexEndpointSnapshot {
            model_found: false,
            responses: false,
            streaming: false,
            tools: false,
            codex_compatible: false,
        });
    }
    if !response.status().is_success() {
        return vec![codex_connection_error(
            text(
                format!(
                    "The model capability API returned HTTP {}",
                    response.status()
                ),
                format!("モデル能力APIがHTTP {}を返しました", response.status()),
            ),
            text(
                "Check the Hoshikage logs and API version",
                "HoshikageのログとAPI versionを確認してください",
            ),
        )];
    }
    let model = match response.json::<crate::model::HoshikageModelInfo>().await {
        Ok(model) => model,
        Err(_) => {
            return vec![codex_connection_error(
                text(
                    "Could not decode the model capability API response",
                    "モデル能力APIの応答を解釈できません",
                ),
                text(
                    "Use matching Hoshikage server and CLI versions",
                    "HoshikageとCLIのversionを揃えてください",
                ),
            )];
        }
    };
    codex_capability_checks(&CodexEndpointSnapshot {
        model_found: true,
        responses: model.responses,
        streaming: model.streaming,
        tools: model.tools,
        codex_compatible: model.codex_compatible,
    })
}

fn codex_connection_error(message: LocalizedText, remediation: LocalizedText) -> DiagnosticCheck {
    DiagnosticCheck {
        id: "codex.connection".to_string(),
        status: DiagnosticStatus::Error,
        message,
        remediation: Some(remediation),
    }
}

pub fn check_candidate_model(
    model_name: &str,
    model_config: &ModelConfig,
    language: Language,
) -> Result<bool> {
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
            message: text(
                format!("Could not load libllama: {}", libllama_path.display()),
                format!("libllama を読み込めません: {}", libllama_path.display()),
            ),
            remediation: Some(text(
                "Check HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR or HOSHIKAGE_LIB_PATH",
                "HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR または HOSHIKAGE_LIB_PATH の配置を確認してください",
            )),
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
    print_report(&report, language);

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
            message: text(
                format!(
                    "The Hoshikage speculation adapter is available: {}",
                    path.display()
                ),
                format!(
                    "Hoshikage speculation adapter が見つかります: {}",
                    path.display()
                ),
            ),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: "runtime.speculation_adapter.path".to_string(),
            status: DiagnosticStatus::Warn,
            message: text(
                format!(
                    "The Hoshikage speculation adapter is missing: {}",
                    path.display()
                ),
                format!(
                    "Hoshikage speculation adapter が見つかりません: {}",
                    path.display()
                ),
            ),
            remediation: Some(text(
                "Install the Hoshikage adapter to accelerate MTP / Draft inference",
                "MTP / Draft の実速度化には Hoshikage 用 adapter を配置してください",
            )),
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
            message: text(
                format!("model_map.json is missing: {}", model_map_path.display()),
                format!(
                    "model_map.json が見つかりません: {}",
                    model_map_path.display()
                ),
            ),
            remediation: Some(text(
                "Register a model with hoshikage add",
                "hoshikage add でモデルを登録してください",
            )),
        });
        return Ok(checks);
    }

    let content = std::fs::read_to_string(&model_map_path)?;
    let models: HashMap<String, ModelConfig> = serde_json::from_str(&content)?;
    let Some(model_config) = models.get(model_name) else {
        checks.push(DiagnosticCheck {
            id: "model.registered".to_string(),
            status: DiagnosticStatus::Error,
            message: text(
                format!("Model '{model_name}' is not registered"),
                format!("モデル '{model_name}' は登録されていません"),
            ),
            remediation: Some(text(
                "Run hoshikage list --details to list registered models",
                "hoshikage list --details で登録済みモデルを確認してください",
            )),
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
        CapabilityStatus::Available => text(
            format!("{label} is available"),
            format!("{label} は利用できます"),
        ),
        CapabilityStatus::Missing => text(
            format!("{label} is unavailable"),
            format!("{label} は利用できません"),
        ),
        CapabilityStatus::AdapterRequired => text(
            format!("{label} exists in the runtime but requires an adapter"),
            format!("{label} は runtime 側に存在しますが adapter 接続が必要です"),
        ),
    };

    DiagnosticCheck {
        id: id.to_string(),
        status: diagnostic_status,
        message,
        remediation: (diagnostic_status != DiagnosticStatus::Ok).then(|| {
            text(
                "Install or enable the required Hoshikage runtime capability",
                remediation,
            )
        }),
    }
}

fn file_check(id: &str, path: &Path, ok_message: &str, err_message: &str) -> DiagnosticCheck {
    if path.is_file() {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Ok,
            message: text(
                format!("File is available: {}", path.display()),
                format!("{}: {}", ok_message, path.display()),
            ),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Error,
            message: text(
                format!("File is missing: {}", path.display()),
                format!("{}: {}", err_message, path.display()),
            ),
            remediation: Some(text(
                "Check the configured path or runtime installation",
                "登録内容または runtime 配置先のパスを確認してください",
            )),
        }
    }
}

fn directory_check(id: &str, path: &Path, ok_message: &str, err_message: &str) -> DiagnosticCheck {
    if path.is_dir() {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Ok,
            message: text(
                format!("Directory is available: {}", path.display()),
                format!("{}: {}", ok_message, path.display()),
            ),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Error,
            message: text(
                format!("Directory is missing: {}", path.display()),
                format!("{}: {}", err_message, path.display()),
            ),
            remediation: Some(text(
                "Install the llama.cpp runtime bundle in ~/.config/hoshikage/llama.cpp",
                "llama.cpp runtime bundle を ~/.config/hoshikage/llama.cpp に配置してください",
            )),
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
            message: text(
                format!("Optional backend is available: {}", path.display()),
                format!("{}: {}", ok_message, path.display()),
            ),
            remediation: None,
        }
    } else {
        DiagnosticCheck {
            id: id.to_string(),
            status: DiagnosticStatus::Warn,
            message: text(
                format!("Optional backend is missing: {}", path.display()),
                format!("{}: {}", warn_message, path.display()),
            ),
            remediation: Some(text(
                "Check the CUDA llama.cpp runtime bundle installation",
                "CUDA 版 llama.cpp runtime bundle の配置を確認してください",
            )),
        }
    }
}

fn vision_compat_check(runtime_report: Option<&RuntimeCapabilityReport>) -> DiagnosticCheck {
    match runtime_report.map(|report| report.vision.status) {
        Some(CapabilityStatus::Available) => DiagnosticCheck {
            id: "model.vision.compat".to_string(),
            status: DiagnosticStatus::Ok,
            message: text(
                "The mmproj configuration matches the Vision runtime",
                "mmproj 設定と Vision runtime は整合しています",
            ),
            remediation: None,
        },
        _ => DiagnosticCheck {
            id: "model.vision.compat".to_string(),
            status: DiagnosticStatus::Error,
            message: text(
                "mmproj is configured, but the Vision runtime is unavailable",
                "mmproj が設定されていますが Vision runtime が利用できません",
            ),
            remediation: Some(text(
                "Install a llama.cpp runtime bundle that includes libmtmd",
                "libmtmd を含む llama.cpp runtime bundle を配置してください",
            )),
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
            message: text("Speculation is disabled", "speculation は off です"),
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
            message: text(
                "The speculation configuration matches the runtime",
                "speculation 設定と runtime は整合しています",
            ),
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
        message: text(
            format!(
                "{:?} is configured, but the runtime adapter is not connected",
                model_config.speculation.modes
            ),
            format!(
                "{:?} が設定されていますが runtime adapter が未接続です",
                model_config.speculation.modes
            ),
        ),
        remediation: Some(if strict {
            text(
                "Set fallback to warn or enable speculation after installing the adapter",
                "fallback を warn にするか、adapter 実装後に有効化してください",
            )
        } else {
            text(
                "Hoshikage will fall back to normal inference according to the fallback policy",
                "fallback 方針に従い通常推論へ戻します",
            )
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
            message: text("Thinking mode is automatic", "Thinking mode は auto です"),
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
            message: text(
                "Thinking off is enforced by the prompt policy and safety filter; the runtime budget adapter is not connected",
                "Thinking off は prompt policy と safety filter で適用されます。runtime budget adapter は未接続です",
            ),
            remediation: Some(text(
                "Connect runtime budget control when the adapter is available",
                "adapter 実装フェーズで runtime budget 制御を接続します",
            )),
        }
    } else {
        DiagnosticCheck {
            id: "model.thinking.compat".to_string(),
            status: DiagnosticStatus::Error,
            message: text(
                "The Thinking-off prompt policy cannot be applied",
                "Thinking off の prompt policy を適用できません",
            ),
            remediation: Some(text(
                "Check the libllama chat template API",
                "libllama の chat template API を確認してください",
            )),
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

fn print_report(report: &DoctorReport, language: Language) {
    println!("{}", language.select("Hoshikage doctor", "Hoshikage診断"));
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
        println!(
            "[{:?}] {} - {}",
            check.status,
            check.id,
            check.message.get(language)
        );
        if let Some(remediation) = &check.remediation {
            println!(
                "      {}: {}",
                language.select("next", "対処"),
                remediation.get(language)
            );
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
                message: text("ok", "正常"),
                remediation: None,
            },
            DiagnosticCheck {
                id: "warn".to_string(),
                status: DiagnosticStatus::Warn,
                message: text("warning", "警告"),
                remediation: None,
            },
            DiagnosticCheck {
                id: "error".to_string(),
                status: DiagnosticStatus::Error,
                message: text("error", "エラー"),
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
                draft_n_max: None,
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
                draft_n_max: None,
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
                ..ThinkingConfig::default()
            },
            ..ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new())
        };

        let check = thinking_compat_check(&config, None);

        assert_eq!(check.status, DiagnosticStatus::Ok);
    }

    #[test]
    fn codex_checks_have_stable_order_and_warn_for_text_only_model() {
        let checks = codex_capability_checks(&CodexEndpointSnapshot {
            model_found: true,
            responses: true,
            streaming: true,
            tools: false,
            codex_compatible: true,
        });

        assert_eq!(
            checks
                .iter()
                .map(|check| check.id.as_str())
                .collect::<Vec<_>>(),
            [
                "codex.connection",
                "codex.model",
                "codex.responses",
                "codex.streaming",
                "codex.tools",
                "codex.context",
            ]
        );
        assert_eq!(checks[4].status, DiagnosticStatus::Warn);
    }

    #[test]
    fn codex_checks_fail_dependents_when_model_is_missing() {
        let checks = codex_capability_checks(&CodexEndpointSnapshot {
            model_found: false,
            responses: false,
            streaming: false,
            tools: false,
            codex_compatible: false,
        });

        assert_eq!(checks[0].status, DiagnosticStatus::Ok);
        assert_eq!(checks[1].status, DiagnosticStatus::Error);
        assert!(checks[2..]
            .iter()
            .all(|check| check.status == DiagnosticStatus::Error));
    }
}
