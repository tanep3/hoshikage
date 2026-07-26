use clap::{Parser, Subcommand};
use hoshikage::api;
use hoshikage::commands::{
    add_model, create_token, doctor, list_models, list_tokens, remove_model, revoke_token,
    rotate_token, AddModelOptions,
};
use hoshikage::config::Config;
use hoshikage::error::HoshikageError;
use hoshikage::model::ModelManager;
use std::fs::OpenOptions;
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
        #[arg(long, value_name = "PATH")]
        mtp_drafter: Option<String>,
        #[arg(long, value_name = "PATH")]
        draft_model: Option<String>,
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
    },
    Token {
        #[command(subcommand)]
        command: TokenCommands,
    },
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

    if let Some(command) = cli.command {
        match command {
            Commands::Add {
                path,
                label,
                mmproj,
                mtp_drafter,
                draft_model,
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
                    mtp_drafter,
                    draft_model,
                    thinking_off,
                    n_ctx,
                    n_gpu_layers,
                    check,
                    label,
                    port: command_port,
                })
                .await?;
            }
            Commands::Rm { label } => {
                remove_model(label, command_port).await?;
            }
            Commands::List { details } => {
                list_models(command_port, details).await?;
            }
            Commands::Doctor { model, json } => {
                doctor(model, json).await?;
            }
            Commands::Token { command } => match command {
                TokenCommands::Create { name } => create_token(name).await?,
                TokenCommands::List => list_tokens().await?,
                TokenCommands::Rotate { name } => rotate_token(name).await?,
                TokenCommands::Revoke { name } => revoke_token(name).await?,
            },
        }
        return Ok(());
    }

    let config = Config::load()?;

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

    axum::serve(listener, app).await?;

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
    log_dir: &std::path::Path,
    file_prefix: &str,
) -> hoshikage::Result<Option<std::fs::File>> {
    #[cfg(unix)]
    {
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let file_name = format!("{}.{}", file_prefix, date);
        let file_path = log_dir.join(file_name);

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
