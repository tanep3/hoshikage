fn main() {
    println!("cargo:rerun-if-changed=llama_cpp_local/include/llama.h");
    println!("cargo:rerun-if-changed=llama_cpp_local/include/ggml.h");

    let bindings = bindgen::Builder::default()
        .header("llama_cpp_local/include/llama.h")
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
