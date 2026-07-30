use clap::{Parser, Subcommand};
use hoshikage::api;
use hoshikage::codex::CodexProfileMode;
use hoshikage::commands::{
    add_model, codex_config, codex_model_catalog, create_token, doctor, list_models, list_tokens,
    remove_model, revoke_token, rotate_token, AddModelOptions,
};
use hoshikage::config::Config;
use hoshikage::error::HoshikageError;
use hoshikage::i18n::Language;
use hoshikage::model::ModelManager;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::num::NonZeroU32;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use tracing_subscriber::fmt::writer::BoxMakeWriter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long)]
    port: Option<u16>,

    #[arg(long, value_enum, global = true)]
    language: Option<LanguageArg>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Add {
        #[arg(value_name = "PATH")]
        path: String,
        #[arg(value_name = "LABEL")]
        label: String,
        #[arg(long, value_name = "PATH")]
        mmproj: Option<String>,
        #[arg(long)]
        mtp: bool,
        #[arg(long, value_name = "PATH")]
        mtp_drafter: Option<String>,
        #[arg(long, value_name = "PATH")]
        draft_model: Option<String>,
        #[arg(long, value_name = "N")]
        spec_draft_n_max: Option<NonZeroU32>,
        #[arg(long)]
        thinking_off: bool,
        #[arg(long, value_name = "N")]
        n_ctx: Option<u32>,
        #[arg(long, value_name = "N", allow_hyphen_values = true)]
        n_gpu_layers: Option<i32>,
        #[arg(long)]
        check: bool,
        #[arg(value_name = "STOP_WORDS", num_args = 0..)]
        stop_words: Vec<String>,
    },
    Rm {
        #[arg(value_name = "LABEL")]
        label: String,
    },
    List {
        #[arg(long)]
        details: bool,
    },
    Doctor {
        #[arg(long, value_name = "LABEL")]
        model: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long, value_name = "URL", requires = "model")]
        codex_base_url: Option<String>,
    },
    CodexConfig {
        #[arg(long, value_name = "MODEL_ID")]
        model: String,
        #[arg(long, value_enum, default_value_t = ProfileModeArg::Interactive)]
        mode: ProfileModeArg,
        #[arg(long, default_value = "http://127.0.0.1:3030/v1", value_name = "URL")]
        base_url: String,
        #[arg(long)]
        authenticated: bool,
    },
    CodexModelCatalog {
        #[arg(long)]
        json: bool,
    },
    Token {
        #[command(subcommand)]
        command: TokenCommands,
    },
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, Default)]
enum ProfileModeArg {
    #[default]
    Interactive,
    Unattended,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum LanguageArg {
    En,
    Ja,
}

impl From<LanguageArg> for Language {
    fn from(value: LanguageArg) -> Self {
        match value {
            LanguageArg::En => Self::En,
            LanguageArg::Ja => Self::Ja,
        }
    }
}

impl From<ProfileModeArg> for CodexProfileMode {
    fn from(value: ProfileModeArg) -> Self {
        match value {
            ProfileModeArg::Interactive => Self::Interactive,
            ProfileModeArg::Unattended => Self::Unattended,
        }
    }
}

#[derive(Subcommand, Debug)]
enum TokenCommands {
    Create {
        #[arg(value_name = "NAME")]
        name: String,
    },
    List,
    Rotate {
        #[arg(value_name = "NAME")]
        name: String,
    },
    Revoke {
        #[arg(value_name = "NAME")]
        name: String,
    },
}

#[tokio::main]
async fn main() -> hoshikage::Result<()> {
    let cli = Cli::parse();
    let command_port = cli.port.unwrap_or(3030);
    let config = Config::load()?;
    let language = Language::resolve(
        cli.language.map(Into::into),
        std::env::var("HOSHIKAGE_LANG").ok().as_deref(),
        std::env::var("LC_ALL")
            .ok()
            .filter(|value| !value.is_empty())
            .or_else(|| std::env::var("LANG").ok())
            .as_deref(),
    );

    if let Some(command) = cli.command {
        match command {
            Commands::Add {
                path,
                label,
                mmproj,
                mtp,
                mtp_drafter,
                draft_model,
                spec_draft_n_max,
                thinking_off,
                n_ctx,
                n_gpu_layers,
                check,
                stop_words,
            } => {
                add_model(AddModelOptions {
                    path,
                    stop_words,
                    mmproj,
                    mtp,
                    mtp_drafter,
                    draft_model,
                    spec_draft_n_max,
                    thinking_off,
                    n_ctx,
                    n_gpu_layers,
                    check,
                    label,
                    port: command_port,
                    language,
                })
                .await?;
            }
            Commands::Rm { label } => {
                remove_model(label, command_port, language).await?;
            }
            Commands::List { details } => {
                list_models(command_port, details, language).await?;
            }
            Commands::Doctor {
                model,
                json,
                codex_base_url,
            } => {
                doctor(model, json, codex_base_url, language).await?;
            }
            Commands::CodexConfig {
                model,
                mode,
                base_url,
                authenticated,
            } => {
                codex_config(model, mode.into(), base_url, authenticated).await?;
            }
            Commands::CodexModelCatalog { json } => {
                codex_model_catalog(json, language).await?;
            }
            Commands::Token { command } => match command {
                TokenCommands::Create { name } => create_token(name).await?,
                TokenCommands::List => list_tokens(language).await?,
                TokenCommands::Rotate { name } => rotate_token(name).await?,
                TokenCommands::Revoke { name } => revoke_token(name, language).await?,
            },
        }
        return Ok(());
    }

    let mut _log_guard = None;
    let mut _stderr_guard = None;
    if let Some(log_file_path) = &config.log_file_path {
        let (log_dir, file_prefix) = resolve_log_path(log_file_path);

        std::fs::create_dir_all(&log_dir)?;

        _stderr_guard = redirect_stderr_to_daily_file(&log_dir, &file_prefix)?;

        let file_appender = tracing_appender::rolling::daily(&log_dir, &file_prefix);
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        _log_guard = Some(guard);

        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(BoxMakeWriter::new(non_blocking))
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| HoshikageError::Other(format!("Failed to set logger: {}", e)))?;

        tracing::info!("Logging to file: {}", log_file_path);
    } else {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(BoxMakeWriter::new(std::io::stdout))
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| HoshikageError::Other(format!("Failed to set logger: {}", e)))?;
    }

    let manager = Arc::new(ModelManager::new(config.clone()));

    manager.load_models().await?;

    // タイムアウト監視タスクを開始 (IDLE_TIMEOUT: VRAMオフロード, GREAT_TIMEOUT: RAMディスク解放)
    manager.clone().start_idle_monitor();

    let auth_policy = hoshikage::security::AuthPolicy::for_bind_host(&config.host);
    let token_store = Arc::new(hoshikage::security::FileTokenStore::new(
        config.auth_token_path()?,
    ));
    let auth_state = hoshikage::security::AuthState::validated(auth_policy, token_store)
        .await
        .map_err(|error| HoshikageError::ConfigError(error.to_string()))?;
    let app = api::create_router_with_auth(manager, auth_state);

    let port = cli.port.unwrap_or(config.port);
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, port)).await?;

    tracing::info!("Hoshikage server starting on {}:{}", config.host, port);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;

    Ok(())
}

fn resolve_log_path(log_file_path: &str) -> (std::path::PathBuf, String) {
    let log_path = std::path::PathBuf::from(log_file_path);
    let log_dir = log_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let file_prefix = log_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("hoshikage.log")
        .to_string();

    (log_dir, file_prefix)
}

fn redirect_stderr_to_daily_file(
    _log_dir: &std::path::Path,
    _file_prefix: &str,
) -> hoshikage::Result<Option<std::fs::File>> {
    #[cfg(unix)]
    {
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let file_name = format!("{}.{}", _file_prefix, date);
        let file_path = _log_dir.join(file_name);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        unsafe {
            if libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) == -1 {
                return Err(HoshikageError::Other(
                    "Failed to redirect stderr to log file".to_string(),
                ));
            }
        }

        Ok(Some(file))
    }

    #[cfg(not(unix))]
    {
        Ok(None)
    }
}
