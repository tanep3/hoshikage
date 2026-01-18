# ドキュメント整理完了

## 移動したファイル

以下のPython版ドキュメントを `legacy_python/docs/` に移動しました：

- LLMファイル管理要件定義書.md
- UI設計書.md
- api-spec.md
- deploy-guide.md
- ollama_efficiency_comparison.md
- requirements.md (旧)
- system-design.md (旧)
- user-manual.md (旧)
- 意味クラスタ要約アルゴリズム設計書（v1.0）.md
- 引き継ぎプロンプト.md
- 画面モックアップ.md
- 要件定義書.md (旧)

## 新規作成したRust版ドキュメント

以下のRust版用ドキュメントを作成しました：

- **docs/requirements.md**
  - 要件定義書と旧requirements.mdを統合
  - 長期記憶保持システムの記述を削除
  - Rustバージョン用に調整

- **docs/api-spec.md**
  - OpenAI互換API仕様
  - エンドポイント一覧
  - リクエスト/レスポンス形式
  - エラーコード一覧

- **docs/system-design.md**
  - システム構成
  - デザインパターン
  - データ構造
  - メモリ管理
  - モジュール構成

- **docs/user-manual.md**
  - インストール手順
  - モデル管理方法
  - サーバー起動方法
  - API使用方法
  - トラブルシューティング

- **README.md**
  - プロジェクト概要
  - クイックスタート
  - アーキテクチャ図
  - パフォーマンス目標
  - ドキュメントリンク

## 主な変更点

### 長期記憶保持システムの削除

以下の機能はRust版では不要のため削除しました：

- ChromaDB短期記憶
- sentence-transformers (ruri-small-v2)
- 意味クラスタリング要約エンジン (K-Means)
- 会話履歴の保持
- 文脈理解機能

### 技術スタックの変更

| 項目 | Python版 | Rust版 |
|--------|----------|---------|
| 言語 | Python 3.10+ | Rust 1.70+ |
| Webフレームワーク | FastAPI | Axum |
| 推論エンジン | llama-cpp-python | llama.cpp (静的リンク) |
| シリアライゼーション | Pydantic | serde |
| 非同期ランタイム | asyncio | tokio |

### デプロイ方法の変更

| 項目 | Python版 | Rust版 |
|--------|----------|---------|
| モデル管理 | llama-cpp-python + llama.cpp | llama.cpp (静的リンク) |
| デプロイ | Docker + libllama.so | 単一バイナリ (311KB) |
| ライブラリ配置 | libllama.soが必要 | システムCUDAライブラリのみ必要 |

---

## 現在のプロジェクト構成

```
hoshikage/
├── Cargo.toml              # Rustパッケージ管理
├── Cargo.lock              # 依存関係ロック
├── build.rs               # ビルドスクリプト（静的リンク設定）
├── README.md               # プロジェクト概要（Rust版）
├── RUST_IMPLEMENTATION.md # Rust実装記録
├── AGENTS.md              # エージェント用ガイド
├── docs/                  # ドキュメントディレクトリ
│   ├── requirements.md      # 要件定義書（Rust版）
│   ├── api-spec.md        # API仕様書（Rust版）
│   ├── system-design.md    # システム設計書（Rust版）
│   └── user-manual.md      # ユーザーマニュアル（Rust版）
├── legacy_python/          # Python版コード・ドキュメント
│   ├── src_python/         # Pythonコード
│   └── docs/             # Python版ドキュメント（移動済み）
├── llama_cpp_local/        # llama.cpp静的ライブラリ
│   ├── include/           # ヘッダーファイル
│   ├── lib/static/        # 静的ライブラリ (.a)
│   └── llama-cli         # コマンドラインツール
├── src/                   # Rustソースコード
│   └── main.rs           # エントリーポイント
├── models/                # モデルファイル格納場所
│   └── README.md
├── data/                  # データディレクトリ
└── old/                  # 古いバージョン
```

---

## 次のステップ

Rust版の実装を進める必要があります：

1. ✅ **完了**: 静的リンクによるllama.cpp統合
2. ✅ **完了**: Pythonコード・ドキュメントの整理
3. ✅ **完了**: Rust版ドキュメント作成
4. **次**: AxumでOpenAI互換API実装
5. **次**: モデル管理機能実装
6. **次**: ストリーミング対応（SSE）
7. **次**: デプロイスクリプト作成

---

**作成日:** 2026-01-18  
**バージョン:** 1.0.0
