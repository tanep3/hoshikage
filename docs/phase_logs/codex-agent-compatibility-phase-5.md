# Codex Agent Compatibility Phase 5 作業ログ

## 2026-07-27

状態: Phase 5 Fix

### 目的

Codex互換Providerを、利用者が安全に設定・診断・運用できる状態へ完成させる。

### 実装順序

1. Codex configと全Bundleモデルカタログ
2. CapabilityとDoctorのCodex互換診断
3. Observabilityと本文Redaction
4. 英語・日本語CLI表示
5. 英語・日本語ユーザーマニュアル
6. CLI実行確認と全回帰

### 境界

- HoshikageはCodex設定を標準出力または明示された出力先へ生成する
- `$CODEX_HOME`や既存Profileを自動変更しない
- Providerとモデルの採用は利用者または上位アプリケーションが決定する
- カタログへGGUF path、Token、Tool本文を出力しない
- Yatagarasu固有Skillの統合はPhase 6で行う

### 実装

#### Codex設定とモデルカタログ

- `hoshikage codex-config`を追加した。
- 対話用は`approval_policy = "on-request"`、無人用は`never`を明示する。
- loopbackは`requires_openai_auth = false`、LAN認証は`env_key = "HOSHIKAGE_API_KEY"`を出力する。
- Bundleの実効contextからcontext、auto compact、Tool出力上限を生成する。
- `hoshikage codex-model-catalog --json`を追加し、全Bundleの能力をpathや秘密情報なしで返す。
- Codex実用下限を16K、推奨を32Kとして扱う。

#### CapabilityとDoctor

- Hoshikageモデル能力へcontext、Codex互換、Responses、streaming、Tool分類情報を追加した。
- `doctor --codex-base-url`でhealth、認証、モデル存在、Responses、streaming、tools、16K contextを接続前診断する。
- 診断IDと順序を固定した。
- `doctor --json`はlocale非依存のstatus、ID、`message_key`、`remediation_key`を返す。
- Bundle候補診断は設定を自動変更しない。

#### Observabilityとdebug capture

- Responsesへ一貫したrequest IDを付与し、`x-request-id`で返す。
- 通常ログはrequest/response ID、model、stream、tools数、token数、経過時間、終端状態、エラー分類だけを記録する。
- payload型の`Debug`表示を抑止し、`Redacted<T>`と安全なTool payload summaryを追加した。
- `HOSHIKAGE_DEBUG_CAPTURE=on`の明示opt-inを実装した。
- captureは固定directory、request単位JSONL、24時間、100 MiB、単一record 16 MiB、古い順削除とした。
- Unixではdirectory `0700`、file `0600`を強制する。
- Authorization、Token/API key field、metadataをcaptureから除外する。
- 起動時とcapture完了時に期限・容量超過分を削除し、並行request間の更新を直列化する。

#### CLIとマニュアル

- 人間向けCLI表示へ英語・日本語を追加した。
- 選択順は`--language`、`HOSHIKAGE_LANG`、OS locale、英語fallbackとした。
- `.env`読込後に言語を決め、command line指定が設定値より優先する。
- `add`、`rm`、`list`、`doctor`、Token管理、Codexカタログを対象とした。
- Token create/rotateは秘密値だけを返す機械契約を維持し、管理者用Token listはToken本文を含む全管理情報を返す契約へ改訂した。
- 日本語`user-manual.md`と英語`user-manual.en.md`を同じ9章構成で整備した。
- loopback、LAN Token、rotate/revoke/紛失、401、Codex設定、Profile分離、状態API、debug capture、TLSを記載した。
- 章番号とcode blockの完全一致を`tests/manual_parity.rs`で契約化した。
- `.env.example`へ認証、言語、unknown field policy、debug captureの安全な既定を追加した。

#### エンドユーザー向けCodex導入手順の再構成

- Tokenを「LAN上のHoshikage利用を許可する秘密の合言葉」として、OpenAI APIキーやChatGPTログインとの違いから説明した。
- Linux、macOS、Windowsを同格に扱い、Hoshikage server設定、server machine上の管理CLI、上位applicationからCodex child processへの環境変数注入、Codex利用者設定を別概念として説明した。
- Windows版Codexアプリを直接起動する場合は、Windows利用者環境変数への登録とアプリ再起動を一つの固有手順として残した。
- Windows版Codexアプリの設定先を`%USERPROFILE%\.codex\config.toml`と明記し、完全なTOML例とPowerShellでファイルを開く手順を追加した。
- Linux・macOSのCodex設定先と、OS別Hoshikage server設定directoryを明記した。
- project-local `.codex/config.toml`ではProvider選択が無視されること、`AGENTS.md`は接続設定ではないことを明記した。
- Windows版CodexアプリとWSL CLIで`CODEX_HOME`が自動共有されないことを明記した。
- CLIで用途別に選択する場合は`$CODEX_HOME/hoshikage.config.toml`と`--profile hoshikage`を使う現行形式を記載し、廃止済みの`[profiles.hoshikage]`を使用しないよう明記した。
- Token本文をTOMLへ保存せず、原則としてYatagarasu等の上位applicationが`HOSHIKAGE_API_KEY`をCodex child processへ渡す責務を具体化した。
- OS別設定先、Profile設定先、project設定の禁止事項、Token list、旧形式移行、実行確認コマンドを`tests/manual_parity.rs`の契約へ追加した。

#### 管理者用Token一覧とcross-platform保存

- Token保存形式をversion 2へ拡張し、owner限定領域へ復元可能なToken本文を保存する。
- 保存用`StoredTokenRecord`と認証用digest-only `TokenVerifierSet`を分離し、middlewareへToken本文を渡さない。
- `hoshikage token list`はserver machine上の管理者用CLIとしてname、Token本文、public ID、作成・更新日時を表示する。
- Linux・macOSは`0600`を設定・検証する。
- WindowsはownerとSYSTEMだけにfull controlを許可するprotected DACLをWindows APIで設定・検証する。
- 旧version 1 digest-only recordは認証を継続し、listでrotate必要を明示する。rotate後はversion 2へ移行する。

### 実機確認

- 現在の15 Bundleをモデルカタログへ出力し、pathとTokenが含まれないことを確認した。
- `unsloth-gemma4-12b-qat-thinking-off`はcontext 16384、Responses/streaming/tools/vision有効、native parserとして出力された。
- 認証付きCodex設定は`HOSHIKAGE_API_KEY`を参照し、Token本文を出力しないことを確認した。
- 検証専用loopback serverをport 3031で起動した。
- `doctor --codex-base-url http://127.0.0.1:3031/v1`で接続、モデル、Responses、streaming、tools、contextの6診断がすべてOKとなった。
- `.env`の`HOSHIKAGE_LANG=ja`と明示`--language en`の優先関係を実CLIで確認した。
- 個人設定のLAN bindかつToken未登録では、起動時に`at least one bearer token is required`で拒否されることを確認した。

### テスト結果

- `cargo test --all-targets`: unit 212 passed、0 failed、1 ignored。
- Codex/llama-server contract: 12 passed、0 failed。
- 日英マニュアル契約: 2 passed、0 failed。
- `cargo clippy --all-targets -- -D warnings`: passed。
- `cargo clippy --target x86_64-pc-windows-gnu -- -D warnings`: passed。
- `cargo clippy --target x86_64-apple-darwin -- -D warnings`: passed。
- `cargo fmt --check`: passed。
- `git diff --check`: passed。
- ignoredは既存の手動runtime probe `probe_local_llama_cpp_bundle`のみ。
- build scriptはllama.cpp header未検出時にchecked-in `src/ffi.rs`を使う旨を通知する。Rust/Clippy警告ではない。
- Windows ACLの実OS runtime試験とmacOS実OS runtime試験は、このLinux開発環境では未実施。両targetのcross compileとClippyは成功した。

### 検出して解消した事項

- Codex config/catalogの初期stub test 4件はTDDのRedとして失敗後、実装して成功した。
- Doctor Codex診断の初期stub test 2件はTDDのRedとして失敗後、実装して成功した。
- debug capture test 2件は未実装型を検出するRedを確認後、実装して成功した。
- エンドユーザー向けCodex設定契約はWindows設定先未記載を検出するRedを確認後、日英マニュアルを再構成して成功した。
- Token listの全情報契約は`TokenMetadata`にToken accessorがないcompile errorをRedとして確認後、保存用recordと認証用verifierを分離して成功した。
- 絞り込みtest名を2つ同時に渡した実行はCargoの引数仕様で失敗した。対象を個別実行し、双方成功した。
- 日英マニュアル契約testが「1.2に記載」「4.4または」という本文を見出しと誤認して失敗した。Markdown見出し行だけを解析するようtestを修正した。
- macOS cross compileは既存の`statvfs`型差により空き容量計算がcompile errorとなった。OS別整数型を`u64`へ構造的に正規化し、overflow testを追加して解消した。
- Linux Clippyは上記正規化の初期`TryFrom`を同型変換として拒否した。`u32`/`u64`を扱う内部traitへ変更し、Linux・macOS双方で警告なしとした。
- Windows/macOS targetの初回dependency取得はsandbox内DNS解決に失敗した。許可されたnetwork実行で取得後、cross compileを完了した。
- `cargo fmt --check`がDoctorの2整形差分を検出し、`cargo fmt`で解消した。
- Doctor本文をlocalized型へ移行中、未変換33箇所をコンパイラが検出し、全箇所を英日対へ移行した。
- 観測エラー分類追加時、正式variant名と`server_busy`の網羅漏れをコンパイラが検出し、解消した。
- sandbox内server起動は設定領域とnetwork bind制約で失敗したため、許可された実行で検証した。
- 最終の実server再実行は権限承認サービスの利用上限で拒否された。直前の実server接続成功後の変更はログ分類と内部整形であり、API契約は全HTTP統合テストとPhase 4実Codex E2Eで再確認した。

### 残存警告

- 実機DoctorはThinking runtime adapterについて警告する。
- 標準BundleではThinking offをprompt policyとsafety filterで適用し、runtime budget adapterは未接続である。
- Phase 5変更による回帰ではなく、既存runtime capabilityの明示である。

### Fix判定

- 日本語版と英語版のユーザーマニュアルで、Tokenの目的、管理者向けToken操作、Codexへの設定手順、Linux・macOS・Windowsの利用手順を同期した。
- 日英マニュアル契約テストと全品質ゲートの成功を確認した。
- 2026-07-27、Phase 5をFixとした。
