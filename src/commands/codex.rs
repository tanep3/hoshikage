use crate::codex::{
    build_model_catalog, render_codex_config, CodexConfigOptions, CodexProfileMode,
};
use crate::config::Config;
use crate::error::Result;
use crate::i18n::Language;
use crate::model::ModelRegistry;

pub async fn codex_config(
    model: String,
    mode: CodexProfileMode,
    base_url: String,
    authenticated: bool,
) -> Result<()> {
    let config = Config::load()?;
    let registry = loaded_registry(config.clone()).await?;
    let model_config = registry.get(&model).await?;
    let rendered = render_codex_config(
        &model_config,
        &CodexConfigOptions {
            model,
            mode,
            base_url,
            authenticated,
            default_context_window: config.n_ctx,
        },
    )?;
    print!("{rendered}");
    Ok(())
}

pub async fn codex_model_catalog(json: bool, language: Language) -> Result<()> {
    let config = Config::load()?;
    let registry = loaded_registry(config.clone()).await?;
    let catalog = build_model_catalog(&registry.snapshot().await, config.n_ctx);
    if json {
        println!("{}", serde_json::to_string_pretty(&catalog)?);
        return Ok(());
    }

    println!(
        "{}",
        language.select(
            "Codex-compatible model catalog:",
            "Codex互換モデルカタログ:"
        )
    );
    for model in catalog.data {
        println!(
            "{}\tcontext={}\tcompatible={}\ttools={}\tvision={}",
            model.id,
            model.context_window,
            model.codex_compatible,
            model.capabilities.tools,
            model.capabilities.vision
        );
    }
    Ok(())
}

async fn loaded_registry(config: Config) -> Result<ModelRegistry> {
    let registry = ModelRegistry::new(config);
    registry.load().await?;
    Ok(registry)
}
