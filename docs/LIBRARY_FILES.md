# ライブラリ管理について

## ライブラリファイルの種類と配置

### 静的ライブラリ（.aファイル）

**場所**: `llama_cpp_local/lib/static/` ディレクトリ

**ファイル一覧:**
- `libllama.a` (6.7MB) - llama.cppのコアライブラリ
- `libggml.a` (55KB) - GGMLコアライブラリ
- `libggml-cpu.a` (1.5MB) - CPUバックエンド
- `libggml-cuda.a` (121MB) - CUDAバックエンド

**特徴:**
- これらはビルド済みの静的ライブラリです
- `build.rs` で静的リンク設定されています
- バイナリに埋め込まれます
- 単一バイナリでデプロイ可能

**注意点:**
- 静的ライブラリは動作時に動的リンクをしません
- すべてのCUDAコードとGPUリソースがバイナリに含まれます
- ファイルサイズが大きいため、バイナリサイズが増えます（~130MB）

### 動的リンク用ライブラリ（.so/.dllファイル）

**場所**: システムCUDAライブラリ（`/usr/local/cuda/targets/x86_64-linux/lib/` または `~/.config/hoshikage/lib/`）

**必要なファイル:**
- `libcuda.so` - CUDAコアライブラリ
- `libcublas.so` - cuBLASライブラリ
- `libcudart.so` - CUDAランタイムライブラリ

**特徴:**
- 動的リンク：バイナリ起動時にシステムライブラリを使用
- 共有ライブラリ：複数のプロセスから安全に共有
- メモリ効率：バイナリサイズが小さい
- OSレベルの最適化：システムが管理

**配置手順:**
1. システムCUDA Toolkitをインストール
2. 環境変数 `LD_LIBRARY_PATH` を設定
3. Rustプログラムは `dlopen` で自動的にライブラリを検出

## Rustプロジェクトでの設定

### build.rsの役割

`build.rs` は以下の役割を担います：

1. **静的リンクの制御**:
   - `cargo:rustc-link-lib=static=llama` などで静的リンクを指定
   - llama.cppの静的ライブラリを埋め込むように設定

2. **CUDAライブラリの探索パス追加**:
   - `cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib` など
   - `cargo:rustc-link-lib=dylib=cuda` でCUDAライブラリを動的リンク
   - システムCUDAライブラリの場所をプログラムが検出できるようにする

3. **rpath設定**:
   - `cargo:rustc-link-arg=-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib`
   - ライブラリの検索パスをバイナリに埋め込む

## ユーザーライブラリの配置

### オプション1: ユーザーライブラリのみを使用（推奨）

ユーザーが自分でビルドしたCUDAライブラリがある場合：

```bash
# ユーザーライブラリを配置（例: ~/.local/cuda/lib）
mkdir -p ~/.local/cuda/lib
cp libcuda.so libcublas.so libcudart.so ~/.local/cuda/lib/

# 環境変数を設定
export LD_LIBRARY_PATH=~/.local/cuda/lib:$LD_LIBRARY_PATH

# 実行
./target/release/hoshikage
```

### オプション2: プロジェクトに静的ライブラリを含める

プロジェクトに静的ライブラリをコピーして、全てを単一バイナリに含めます。

**メリット:**
- デプロイがさらに簡単（バイナリ1つだけで完結）
- ライブラリバージョン不一致の心配がない
- システム環境に依存しない

**デメリット:**
- バイナリサイズが大きい（~200MB）
- ビルド時間が長くなる

## 推奨構成

### 開発・テスト環境
```bash
# ユーザーライブラリを使用
export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH
```

### 本番環境（運用）
```bash
# システムCUDAライブラリを使用
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

---

**作成日:** 2026-01-18
