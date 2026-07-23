# 星影 (Hoshikage) - 高速ローカル推論サーバー

[![CI](https://github.com/tanep3/hoshikage/actions/workflows/ci.yml/badge.svg)](https://github.com/tanep3/hoshikage/actions/workflows/ci.yml)
[![Release](https://github.com/tanep3/hoshikage/actions/workflows/release.yml/badge.svg)](https://github.com/tanep3/hoshikage/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 概要

**星影（ほしかげ）** は、GGUFフォーマットの大規模言語モデルをローカル環境で高速かつ効率的に実行し、OpenAI互換のAPIを提供するRustアプリケーションです。プライバシーを重視し、外部へのデータ送信を最小限に抑えつつ、高品質な対話型AI体験を提供します。

「**静かなる知性**」という設計思想のもと、必要な時にのみリソースを活用し、非アクティブ時には自動的にメモリを解放します。

---

## ✨ 特徴

### 🚀 高速推論
- **managed llama-server runtime**: ユーザーが導入した llama.cpp runtime を管理して高速推論
- **GPU加速**: CUDA対応（RTX 1650以降）
- **Flash Attention + KV Cache**: 推論速度を最大化
- **Diffusion LLM サポート (v1.1.0)**: LLaDAやRND1などの拡散型言語モデルに対応
- **単一バイナリ**: 311KBのコンパクトなサイズ

### 🧭 次期モデルランタイム改訂
- **Model Bundle 管理**: メインモデル、Vision projector、MTP / Draft model、推論設定をモデル単位で管理
- **Vision / MTP / Draft model 対応**: 最新の GGUF モデル機能に追従する managed `llama-server` runtime
- **Thinking mode 制御**: 低レイテンシ用途向けにモデル単位で Thinking off を指定可能
- **CUDA 版 llama.cpp 導入**: Linux CUDA では source build、macOS / Windows / Linux CPU 系では prebuilt archive を使う導入手順を整備

### 🔌 OpenAI互換API
- **完全互換**: 既存のOpenAIクライアントライブラリがそのまま使用可能
- **ストリーミング対応**: リアルタイムで応答を逐次送信（AR/Diffusion両対応）
- **複数モデル対応**: 複数のGGUFモデルを登録・切り替え可能

### 💡 リソース効率化（静かなる知性）
- **自動モデルアンロード**: 非アクティブ時に自動でメモリ解放
- **セマフォ制御**: 同時リクエスト数を1に制限してVRAM枯渇を防止

---

## 📋 必要要件

### ハードウェア

| 項目 | 最小要件 | 推奨要件 |
|------|---------|---------|
| CPU | 8コア以上 | 16コア以上（Ryzen 7900相当） |
| メモリ | 16GB以上 | 32GB以上 |
| GPU | VRAM 8GB以上 | VRAM 12GB以上 |
| ストレージ | SSD 50GB以上 | NVMe SSD 100GB以上 |

### ソフトウェア

- **OS**: Linux（Ubuntu 20.04以降推奨）
- **CUDAドライバ**: 470+ (GTX 1650以降)
- **Rust**: 1.70以上

---

## 🚀 セットアップ

### 1. ライブラリの準備 (重要)
星影は llama.cpp 本体を配布しません。
ご利用の環境に合わせて公式の llama.cpp runtime を導入し、`~/.config/hoshikage/llama.cpp` に配置してください。

初心者向けの導入手順は **[llama.cpp 最新導入ガイド](docs/llama-cpp-install-guide.md)** を参照してください。
runtime の探索や運用方針は **[ライブラリ運用ガイド](docs/LIBRARY_GUIDE.md)** にまとめています。

### 2. インストール
Cargoを使ってインストールします。

```bash
cargo install --path .
```

---

## 📖 使い方

### サーバーの起動
```bash
# 標準ポート(3030)で起動
hoshikage

# ポート指定で起動
hoshikage --port 8080
```

モデルのダウンロード配置、APIの呼び出し方、Systemdによるデーモン化などの詳細は、
**[ユーザーマニュアル](docs/user-manual.md)** をご覧ください。

---



## 🏗️ アーキテクチャ

```
┌─────────────────────────────────┐
│         Rustバイナリ (311KB)          │
│  ┌────────────────────────────┐ │
│  │ Axum (OpenAI互換API)         │ │
│  ├────────────────────────────┤ │
│  │ managed llama-server runtime   │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
                 │
                 │ user-installed llama.cpp runtime
                 ▼
┌─────────────────────────────────┐
│   CUDA Driver (動的リンク)            │
│   - libcuda.so                    │
│   - libcublas.so                  │
│   - libcudart.so                  │
└─────────────────────────────────┘
```

**runtime の仕組み:**
- Hoshikage は `~/.config/hoshikage/llama.cpp/llama-server` を起動・監視します。
- CUDA / Vulkan / Metal / ROCm などの backend は、ユーザーが導入した llama.cpp runtime に従います。
- FFI 互換経路を使う場合は、同じ runtime directory の shared library を参照します。

---

## 📊 パフォーマンス

| 指標 | 値 |
|-------|-----|
| バイナリサイズ | 311KB |
| 起動時間 | <1秒 |
| 初回モデルロード | 5-10秒 |
| モデルスイッチ | <1秒 |
| 推論速度 (RTX 4070 SUPER) | 90 tokens/s 前後 (12B QAT + MTP + Vision bundle, Thinking off, `TOP_P=0.95`, `REPEAT_PENALTY=1.0`) |

---

## 📝 ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [user-manual.md](docs/user-manual.md) | ユーザーマニュアル |
| [requirements.md](docs/requirements.md) | 要件定義書 |
| [model-runtime-revision-requirements.md](docs/model-runtime-revision-requirements.md) | QAT / MTP / Draft model / Vision 対応の改訂要件 |
| [model-runtime-revision-system-design.md](docs/model-runtime-revision-system-design.md) | Model Runtime 改訂のシステム設計 |
| [llama-cpp-install-guide.md](docs/llama-cpp-install-guide.md) | llama.cpp 最新導入ガイド |
| [api-spec.md](docs/api-spec.md) | API仕様書 |
| [system-design.md](docs/system-design.md) | システム設計書 |
| [nfr-details.md](docs/nfr-details.md) | 非機能要件詳細 |
| [ci-cd-pipeline.md](docs/ci-cd-pipeline.md) | CI/CD パイプライン |

---

## 🙏 謝辞

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - 高速推論エンジン
  - Hoshikage は llama.cpp runtime を同梱しません。利用者が公式配布物または source build で導入します。
  - ライセンス: [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
- [Axum](https://github.com/tokio-rs/axum) - 高速Webフレームワーク
- [Rust](https://www.rust-lang.org/) - システムプログラミング言語

---

## 📜 ライセンス

MIT License

Copyright (c) 2026 Tane Channel Technology

---

**星影 - 暗闇の中で光を放つように、AI技術の可能性を照らす**
