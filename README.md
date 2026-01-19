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
- **llama.cpp (動的リンク)**: システムCUDAライブラリを利用して高速推論
- **GPU加速**: CUDA対応（RTX 1650以降）
- **Flash Attention + KV Cache**: 推論速度を最大化
- **単一バイナリ**: 311KBのコンパクトなサイズ

### 🔌 OpenAI互換API
- **完全互換**: 既存のOpenAIクライアントライブラリがそのまま使用可能
- **ストリーミング対応**: リアルタイムで応答を逐次送信
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
星影は `llama.cpp` の動的ライブラリ (`libllama.so` / `llama.dll`) を使用します。
ご利用の環境に合わせてライブラリを配置する必要があります。

詳細な手順は **[ライブラリ運用ガイド](docs/LIBRARY_GUIDE.md)** を参照してください。

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
│  │ llama.cpp (動的リンク)         │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
                 │
                 │ システムCUDAライブラリ
                 ▼
┌─────────────────────────────────┐
│   CUDA Driver (動的リンク)            │
│   - libcuda.so                    │
│   - libcublas.so                  │
│   - libcudart.so                  │
└─────────────────────────────────┘
```

**動的リンクの仕組み:**
- llama.cppはシステムのCUDAライブラリを動的リンクして使用します
- 環境変数 `LD_LIBRARY_PATH` を設定して検索パスを指定

---

## 📊 パフォーマンス

| 指標 | 値 |
|-------|-----|
| バイナリサイズ | 311KB |
| 起動時間 | <1秒 |
| 初回モデルロード | 5-10秒 |
| モデルスイッチ | <1秒 |
| 推論速度 (RTX 4070 SUPER) | 30-50 tokens/s |

---

## 📝 ドキュメント
(開発者向け)

| ドキュメント | 説明 |
|-------------|------|
| [requirements.md](docs/requirements.md) | 要件定義書 |
| [api-spec.md](docs/api-spec.md) | API仕様書 |
| [system-design.md](docs/system-design.md) | システム設計書 |
| [nfr-details.md](docs/nfr-details.md) | 非機能要件詳細 |
| [ci-cd-pipeline.md](docs/ci-cd-pipeline.md) | CI/CD パイプライン |

---

## 🙏 謝辞

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 高速推論エンジン
  - このソフトウェアは llama.cpp (Copyright (c) 2023 Georgi Gerganov) を含んでいます。
  - ライセンス: [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
- [Axum](https://github.com/tokio-rs/axum) - 高速Webフレームワーク
- [Rust](https://www.rust-lang.org/) - システムプログラミング言語

---

## 📜 ライセンス

MIT License

Copyright (c) 2026 Tane Channel Technology

---

**星影 - 暗闇の中で光を放つように、AI技術の可能性を照らす**
