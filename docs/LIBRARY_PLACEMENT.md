# ライブラリ配置について

## 現状

現在、`~/.config/hoshikage/lib/` ディレクトリは空です。ライブラリの配置について説明します。

## ライブラリの種類と配置場所

### 1. llama.cppの静的ライブラリ（.aファイル）

**場所**: `llama_cpp_local/lib/static/`

**ファイル一覧:**
- `libllama.a` (6.7MB) - llama.cppのメインコアライブラリ
- `libggml.a` (55KB) - GGMLコアライブラリ
- `libggml-cpu.a` (1.5MB) - CPUバックエンド
- `libggml-cuda.a` (121MB) - CUDAバックエンド

**特徴:**
- ビルド済みの静的ライブラリ
- ビルド時にバイナリに埋め込まれます
- 閄密なRustプロジェクトの一部として扱われます

### 2. システムCUDAライブラリ（.soファイル）

**場所**: システムのCUDAインストール場所

例：
```
/usr/local/cuda/targets/x86_64-linux/lib/
├── libcuda.so
├── libcublas.so
├── libcudart.so
└── libcudadebugger.so
```

**特徴:**
- NVIDIAが提供するCUDAツールキットに含まれます
- システム全体で共有される共有ライブラリ
- 実行時にdlopenで動的リンクされます

## Rustプロジェクトでのライブラリ使用

### 静的リンク vs 動的リンク

**llama.cpp（静的ライブラリ）**
- ビルド時にバイナリに埋め込まれます
- Rustのbuild.rsで`cargo:rustc-link-lib=static=llama`などで静的リンクを設定
- 以下のライブラリが埋め込まれます：
  - libllama.a
  - libggml.a
  - libggml-cpu.a
  - libggml-cuda.a

**システムCUDAライブラリ（動的リンク）**
- 実行時にOSが`dlopen`でロードします
- 環境変数`LD_LIBRARY_PATH`を設定する必要があります
- 以下のライブラリが動的にリンクされます：
  - libcuda.so
  - libcublas.so
  - libcudart.so

## lib/ディレクトリの使用

### 現状の確認

`~/.config/hoshikage/lib/` は現在空です。システムCUDAライブラリを使用する場合は、このディレクトリは必要ありません。

### 推奨される構成

#### オプション1: システムCUDAライブラリのみを使用（推奨）

**メリット:**
- バイナリサイズが小さい（311KB）
- システムのライブラリを使用するため安定性が高い
- デプロイが簡単

**方法:**
- `~/.config/hoshikage/lib/` は使用しない
- システムCUDAライブラリに依存
- 環境変数で`LD_LIBRARY_PATH`を設定

**Rustプロジェクトの変更:**
- build.rsで静的リンク設定を維持
- `cargo:rustc-link-lib=static=llama` などを残す

#### オプション2: システムCUDAライブラリを~/.config/hoshikage/lib/にコピー

**メリット:**
- システムCUDAツールキットがなくても動作する
- CUDAライブラリのバージョンをユーザーが管理可能

**方法:**
- システムCUDAライブラリを`~/.config/hoshikage/lib/`にコピー
- 環境変数を設定
- Rustプロジェクトのbuild.rsで動的リンク設定を追加

**注意:**
- `~/.config/hoshikage/lib/` には`.a`ファイル（静的ライブラリ）を配置しないでください
- `.a`ファイルはビルド済みの静的ライブラリとして扱われます

#### オプション3: ライブラリを含めたバイナリ

**メリット:**
- 完全に単一バイナリでデプロイ可能
- ライブラリのバージョン不一致を心配不要

**デメリット:**
- バイナリサイズが非常に大きくなる（~200MB）

**方法:**
- llama.cppをRust用にビルドしなおす
- llama-cpp-rs crateを使用する

## 現状の実装

build.rsでは以下の設定のみ行っています：

```rust
// CUDAライブラリの検索パスを追加
println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");

// ユーザーライブラリ（優先）のパスを追加
println!("cargo:rustc-link-search=native={}", user_lib_dir.display());

// 動的リンク用のライブラリパスは追加されていません
```

これは正しい挙動です。llama.cppの静的ライブラリはシステムCUDAライブラリ（動的リンク）を介して使用されます。

## まとめ

| 項目 | 状況 | 説明 |
|--------|------|--------|
| llama.cpp静的ライブラリ | llama_cpp_local/lib/static/ | ビルド済み、Rustに静的リンク |
| システムCUDAライブラリ | システムディレクトリ | NVIDIA CUDA Toolkitに含まれている |
| 動的リンク用lib/ | 現在使用せず | 推奨：システムライブラリのみ使用 |
| 環境変数LD_LIBRARY_PATH | 必要 | システムライブラリのパスを設定 |

**推奨される方法:**

1. `~/.config/hoshikage/lib/` は使用しない（システムCUDAライブラリに依存）
2. build.rsの現在の設定を維持（システムCUDAライブラリ検索パスを追加）
3. 環境変数`LD_LIBRARY_PATH`を設定

---

**作成日:** 2026-01-18
