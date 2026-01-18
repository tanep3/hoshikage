# CI/CD パイプライン提案 (CI/CD Proposal)

**プロジェクト:** 星影 (Hoshikage)  
**作成日:** 2026-01-18

---

## 提案概要
GitHub Actionsを使用した自動テストおよびビルドフローを提案します。
特に、**動的ライブラリ (llama.cpp) への依存** があるため、CI環境での適切なモック(Mock)またはライブラリダウンロード戦略が必要です。

## パイプライン構成

### 1. CI (Build & Test)
Pull Request および Main ブランチへの push でトリガー。

- **Lint Check**: `cargo clippy`, `cargo fmt`
- **Unit Test**: `cargo test`
  - *注意*: `libllama.so` が必要なテストは、CI環境上にダミーライブラリを用意するか、テスト設定でモック化してスキップする必要があります。
  - **戦略**: テスト実行時は `llama-cpp-2` の機能をモック化（`cfg(test)`）するか、Hoshikage側のラッパー層でモックを使用する設計にします。

### 2. CD (Release)
タグ push (`v*`) でトリガー。

- **Release Build**: `cargo build --release`
- **Artifact Upload**: バイナリ (`hoshikage`) をGitHub Releasesにアップロード
  - *注記*: `libllama.so` は同梱せず、ユーザーに別途用意してもらう運用方針（`LIBRARY_GUIDE.md` 準拠）とします。

---

## Skeleton YAML (`.github/workflows/ci.yml`)

```yaml
name: Hoshikage CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

env:
  CARGO_TERM_COLOR: always

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
        components: rustfmt, clippy

    - name: Check Formatting
      run: cargo fmt --all -- --check

    - name: Lint (Clippy)
      run: cargo clippy -- -D warnings

    # 動的ライブラリ依存のテストは、現状はCI上でスキップまたはモックが必要
    # ここでは基本的な単体テストのみ実行
    - name: Run Tests
      # LD_LIBRARY_PATH等は設定せず、純粋なLogicテストのみ通ることを目指す
      run: cargo test --lib --bins
```

## 承認のお願い
この方針でCI設定ファイルを作成してよろしいでしょうか？
承認いただければ `.github/workflows/ci.yml` を実装します。
