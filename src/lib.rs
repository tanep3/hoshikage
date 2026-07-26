pub mod api;
pub mod config;
pub mod conversation;
pub mod inference;
pub mod model;
pub mod runtime;
pub mod security;

pub mod commands;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::upper_case_acronyms
)]
mod ffi;

pub mod error;

pub use error::{HoshikageError, Result};
