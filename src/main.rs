include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));

fn main() {
    println!("星影 - 高速ローカル推論サーバー (CUDA + Flash Attention + KV Cache)");
    println!("Version: 1.0.0");
}
