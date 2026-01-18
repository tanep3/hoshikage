# 星影 (Hoshikage) - 高速ローカル推論サーバー

## 静的リンク成功！

### 結果

✅ **静的リンクが完了**
- Releaseバイナリサイズ: **311KB** (非常に小さい！)
- Debugバイナリサイズ: 3.8MB
- **llama.cpp, ggml, ggml-cpu, ggml-cuda が静的にリンク**
- CUDA + Flash Attention + KV Cache 対応済み

### 実行方法

```bash
# ビルド
cargo build --release

# 実行
LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH ./target/release/hoshikage
```

### デプロイ構成

```
/opt/hoshikage/
├── hoshikage (311KBのバイナリ)
└── models/
    └── *.gguf
```

### 注意点

1. **実行環境にCUDAドライバが必要**
   - NVIDIA Driver 470+ (GTX 1650以降)
   - CUDA Toolkitは不要（システムライブラリを使用）

2. **LD_LIBRARY_PATHを設定**
   - CUDAライブラリ（libcuda.so, libcublas.so, libcudart.so）を動的リンク
   - これらはシステムCUDAライブラリを使用

3. **バイナリサイズ**
   - 311KBの小さなバイナリ
   - llama.cppの静的ライブラリから必要なコードのみが抽出

### ビルド手順（完全）

```bash
# 1. llama.cppを静的ライブラリとしてビルド
cd llama.cpp
cmake -B build_static \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build_static --target llama -j8

# 2. 静的ライブラリをプロジェクトにコピー
cp build_static/src/libllama.a hoshikage/llama_cpp_local/lib/static/
cp build_static/ggml/src/libggml.a hoshikage/llama_cpp_local/lib/static/
cp build_static/ggml/src/libggml-cpu.a hoshikage/llama_cpp_local/lib/static/
cp build_static/ggml/src/ggml-cuda/libggml-cuda.a hoshikage/llama_cpp_local/lib/static/

# 3. Rustプロジェクトをビルド
cd hoshikage
cargo build --release
```

### 次のステップ

1. ✅ **完了**: 静的リンク
2. **次**: AxumでOpenAI互換API実装
3. **次**: モデル管理機能実装
4. **次**: ストリーミング対応（SSE）
5. **次**: デプロイスクリプト作成
