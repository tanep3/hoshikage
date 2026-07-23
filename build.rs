use std::path::{Path, PathBuf};

fn main() {
    let include_dir = std::env::var_os("HOSHIKAGE_LLAMA_CPP_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("llama_cpp_local/include"));
    let llama_header = include_dir.join("llama.h");
    let ggml_header = include_dir.join("ggml.h");

    println!("cargo:rerun-if-env-changed=HOSHIKAGE_LLAMA_CPP_INCLUDE_DIR");
    println!("cargo:rerun-if-changed={}", llama_header.display());
    println!("cargo:rerun-if-changed={}", ggml_header.display());

    if !llama_header.is_file() {
        if Path::new("src/ffi.rs").is_file() {
            println!(
                "cargo:warning=llama.cpp headers were not found; using the checked-in src/ffi.rs bindings"
            );
            return;
        }

        panic!(
            "llama.cpp headers were not found at {} and src/ffi.rs is missing",
            llama_header.display()
        );
    }

    let bindings = bindgen::Builder::default()
        .header(llama_header.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .opaque_type("ggml_context")
        .opaque_type("ggml_backend")
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(std::path::PathBuf::from("src/ffi.rs"))
        .expect("Couldn't write bindings!");
}
