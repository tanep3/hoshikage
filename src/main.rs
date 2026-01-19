use clap::{Parser, Subcommand};
use hoshikage::api;
use hoshikage::commands::{add_model, list_models, remove_model};
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

    #[arg(short, long, default_value_t = 3030)]
    port: u16,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Add {
        #[arg(value_name = "PATH")]
        path: String,
        #[arg(value_name = "LABEL")]
        label: String,
        #[arg(value_name = "STOP_WORDS", num_args = 0..)]
        stop_words: Vec<String>,
    },
    Rm {
        #[arg(value_name = "LABEL")]
        label: String,
    },
    List,
}

#[tokio::main]
async fn main() -> hoshikage::Result<()> {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Commands::Add {
                path,
                label,
                stop_words,
            } => {
                add_model(path, label, stop_words, cli.port).await?;
            }
            Commands::Rm { label } => {
                remove_model(label, cli.port).await?;
            }
            Commands::List => {
                list_models(cli.port).await?;
            }
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

    let app = api::create_router(manager);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, cli.port)).await?;

    tracing::info!("Hoshikage server starting on {}:{}", config.host, cli.port);

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
