# Codex Agent Compatibility Phase 1 作業ログ

## 2026-07-27

状態: Phase 1 Fix

### 目的

Responses Wireを追加する前に、既存Chatと将来のResponsesが共有できる内部構造、
managed llama-serverの排他的実行権、LAN公開前の認証基盤を確立する。

### 実装

- Wire型から独立した`Conversation`、検証済みID、Tool Call、Tool Resultを追加
- `ModelRequest`、`ModelCompletion`、`ModelDelta`、`TokenUsage`を追加
- `RuntimeBackend`から`api::ChatMessage`依存を除去
- `ModelRegistry`へモデル定義の所有と原子的保存を分離
- Model Registryの保存失敗時にmemory snapshotをrollback
- `RuntimePhase`と純粋な`RuntimeTransition`を追加
- bounded queueを持つ`RuntimeCoordinator`を追加
- stream終了まで実行権を保持する`RuntimeLease`を既存Chatへ接続
- runtime generationを導入し、古いrequestによる新runtime破壊を防止
- 共有`LlamaServerClient`を追加し、API層からHTTP通信生成を分離
- thinking-off Bundleで`chat_template_kwargs.enable_thinking=false`を送信
- Responses用size、queue、timeout設定と厳格な数値検証を追加
- `.env`読込失敗を無視せず設定エラーとして停止
- CLIのport未指定時に`PORT`設定を利用するよう修正
- loopback無認証、non-loopback Bearer必須の`AuthPolicy`を追加
- 256 bit secretと128 bit public IDを持つ用途名付きTokenを追加
- digestのみを保存し、constant-time比較、zeroize、Unix `0600`を実装
- Tokenのcreate、list、rotate、revoke CLIを追加
- Token更新全体をfile lockで直列化し、複数CLIからの更新消失を防止
- 認証情報をrequestごとに再読込し、破損・権限不正時はfail-closed
- non-loopbackでTokenが0件ならRouter構築前に起動拒否

### 構造上の保証

- HoshikageはToolを実行しない
- API DTOは推論契約へ侵入しない
- Runtime切替は`RuntimeLease`の所有期間中に起こらない
- queue上限を超えたrequestは無制限に滞留しない
- Tokenのrotate/revokeは指定用途だけへ作用する
- plaintext Tokenはcreate/rotate時だけ表示し、保存しない
- API層はmanaged llama-server processを直接操作しない
- Model登録CLIと単一Token decodeは、引数列ではなく用途別Request構造で受け渡す

### テスト

- `cargo fmt --check`: 成功
- `cargo test`: 成功
  - unit test: 129 passed、0 failed、1 ignored
  - Phase 0 contract test: 12 passed、0 failed
  - doc test: 0 failed
- Token CLI隔離試験: create/list/rotate/revoke、`0600`確認すべて成功
- `cargo clippy --all-targets -- -D warnings`: 成功、Clippy warning 0件

主要な追加テスト:

- Conversationの孤立・重複Call検証
- Tool argumentsのJSON正規化
- Runtime stateの不正遷移拒否
- stream ownerがdropするまで別Modelを取得できないこと
- bounded queue超過拒否
- Model Registryの破損JSONによるmemory snapshot破壊防止
- Model Registryの保存失敗時rollback
- thinking-off request契約
- loopback無認証とBearer認証のRouter統合
- non-loopback + Token 0件の起動前拒否
- Token作成・更新・失効の即時反映
- 2用途Tokenの独立性
- Unixの危険なpermission拒否

### 作業中に検出した失敗

- `cargo test`へ複数のfilter引数を渡す誤った実行を2回行い、test開始前にCargoが拒否した。
  以後は単一filterまたはtest suite全体で実行した。
- `cargo fmt --check`は未整形差分を検出して2回失敗した。いずれも`cargo fmt`後は成功した。
- 大きなpatchの文脈不一致により適用前検証が2回失敗した。部分適用はなく、分割して適用した。
- 全回帰中、生成Tokenに`_`が含まれた場合だけparseに失敗するrandom依存bugを検出した。
  Base64URLと同じ文字を単純な区切りとしていたことが原因で、固定長構造のparseへ修正し、
  `_`を含む決定的な回帰testを追加した。
- `cargo clippy --all-targets -- -D warnings`は当初19項目で失敗した。
  今回差分3項目に加えて既存警告16件も構造改善を含めて修正し、最終実行はwarning 0件で成功した。

### 警告・未実施

- llama.cpp headerが見つからないため、checked-in FFI bindingを使用した。
- 実機依存の`probe_local_llama_cpp_bundle` 1件は既存どおりignored。
- GPUおよび実Modelを使うChat E2EはPhase 1では未実施。Phase 0 Fixture契約と既存Chat回帰で検証した。
- Windows nativeのToken file ACL検証は未実装。安全性を検証できないOSではToken作成・利用を
  成功扱いにせずfail-closedとする。Windows native対応時はACL検証または明示override設計が必要。

### Phase 1完了条件

- [x] 既存Chat挙動を維持
- [x] RuntimeBackendが`api::ChatMessage`へ依存しない
- [x] stream中のRuntime切替をConcurrency testで防止
- [x] non-loopback + Tokenなしで起動できない
- [x] 2つの用途名付きTokenを独立管理できる
- [x] 全回帰test成功

Phase 2へはユーザーによるPhase 1 Fixと明示承認後に進む。
