use std::env;
use std::path::PathBuf;

fn main() {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
    println!("cargo:rerun-if-changed=build.rs");
    
    // llama.cppの静的ライブラリは外部スクリプトでコピーするため、
    // ここでは動的リンク用の設定のみ行います
    println!("cargo:warning=Using dynamic llama.cpp linking (libraries should be in ~/.config/hoshikage/lib)");
    
    // CUDAライブラリの検索パスを設定
    #[cfg(target_os = "linux")]
    {
        let cuda_paths = [
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
        ];
        
        for path in &cuda_paths {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        // ユーザーライブラリディレクトリを優先
        let user_lib_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".config/hoshikage/lib");
        
        if user_lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", user_lib_dir.display());
        }
    }
}
