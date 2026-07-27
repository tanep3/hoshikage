# Codex Agent Compatibility Phase 2 作業ログ

## 2026-07-27

状態: Phase 2 Fix

### 目的

既存Chatを維持したまま、Responses APIの非ストリームText requestを受理し、
内部のWire非依存契約を通じてローカルモデル推論へ接続する。

### 実装

- `POST /v1/responses`を追加
- Responses Wire DTOとrequest normalizationをAPI境界へ分離
- `input`の文字列およびmessage Item配列、`instructions`を内部`Conversation`へ変換
- `temperature`、`top_p`、`max_output_tokens`を内部推論契約へ変換
- compatible/strictのunknown field policyを実装
- Phase 2未対応のstream、Tool、Vision、stateful継続を明示エラー化
- Application層へ`ResponsesService`を追加し、API型と推論型の逆依存を防止
- `InferenceGateway`を追加し、Application層から`ModelManager`を分離
- managed llama-server用の非ストリームText Adapterを追加
- thinking-off、既定stop、Bundle固有stopをupstream requestへ反映
- llama-serverのusageをResponses usageへ変換
- request timeoutをApplication境界で強制
- `max_output_tokens`がBundle context上限を超えるrequestを事前拒否
- OpenAI形式の安定したerror codeとHTTP statusを追加
- `GET /health`、`GET /ready`、`GET /v1/capabilities`を追加
- `/health`だけを認証外とし、`/ready`と`/v1/capabilities`は保護対象に設定
- Responses request body size上限をRouter境界で強制

### 構造上の保証

- API Wire型はApplication・Inference・Model管理層へ侵入しない
- HoshikageはToolを実行しない
- Phase 2ではTool能力とStreaming能力を過大表示しない
- 上流timeout、切断、翻訳失敗はHoshikage processを停止させず構造化エラーへ変換する
- model実行権は既存`RuntimeLease`で所有し、request終了時に解放する
- 既存`/v1/chat/completions`の経路を置換しない

### 実モデルE2E

標準Bundle `unsloth-gemma4-12b-qat-thinking-off`を使い、Hoshikageとmanaged
`llama-server`を一時起動して確認した。

- `GET /health`: `{"status":"ok"}`
- `GET /ready`: `{"status":"ready"}`
- `GET /v1/capabilities`: Responses有効、Streaming/Tool/Vision無効
- `POST /v1/responses`: `stream:false`で`OK`
- usage: input 19、output 2、total 21 tokens
- 外部OpenAI APIを使わず、Hoshikageとローカル`llama-server`内で推論完結

Codex CLI `0.144.5`は通常応答でも`stream:true`を送ることがPhase 0で確定している。
このためCodex CLIによるAC-001最終判定はPhase 4へ移し、Phase 2は同等requestの
直接API試験を完了条件とした。

### テスト

- `cargo test`: 成功
  - unit test: 149 passed、0 failed、1 ignored
  - Phase 0 contract test: 12 passed、0 failed
  - doc test: 0 failed
- Responses Wire変換p95: `12.373us`、目標50ms未満
- `cargo clippy --all-targets -- -D warnings`: 成功、Clippy warning 0件

主要な追加テスト:

- string input、instructions、message Item配列の順序保持
- nullable optional field
- compatible/strict unknown field policy
- Phase 2未対応のstream/Tool拒否
- OpenAI Responses output/usage形式
- model not found、timeoutの安定した公開error code
- Bundle context上限超過の`context_length_exceeded`
- malformed llama-server JSONとusage欠落の翻訳失敗
- thinking-offとstop sequence合成
- request body size上限
- `/health`公開、`/ready`認証保護

### 作業中に検出した失敗

- `cargo test`へ複数filterを渡す誤った実行を1回行い、test開始前にCargoが拒否した。
- `cargo fmt --check`が未整形差分2か所を検出して1回停止した。整形後は成功した。
- timeoutのHTTP testで、即時完了する偽推論とゼロ秒期限を組み合わせたため、
  Tokioが完了値を返して期待504に対し200となった。明示的に遅延する偽推論へ修正し、
  決定的にtimeoutを再現した。
- sandbox内のserver起動はlistener bindの`PermissionDenied`で失敗した。
  承認済みの実行環境で再試験し成功した。
- sandbox内のclientから承認済み環境のloopback serverへ接続できなかった。
  同一の承認済み環境から再試験し成功した。
- GPU確認用の`nvidia-smi`は環境に存在しなかったが、実モデルE2E自体は成功した。

### 警告・未実施

- llama.cpp headerが見つからないため、checked-in FFI bindingを使用した。
- 実機依存の`probe_local_llama_cpp_bundle` 1件は既存どおりignored。
- 標準BundleのPhase 2試験時の実設定は`n_ctx=8192`であり、正式なCodex互換最低保証16Kを
  満たさなかった。Phase 2 Fix時に16Kへの変更が承認され、Phase 3開始時に実設定へ反映する。
- Phase 2は仕様どおりSSE、Tool Call、Codex CLI E2Eを実装・実施していない。

### Phase 2完了条件

- [x] 標準Bundleへの非ストリームResponses requestで`OK`
- [x] 推論通信がローカル経路内で完結
- [x] Responses変換p95が50ms未満
- [x] Bundle context上限を超える出力token指定を事前拒否
- [x] Chatを含む全回帰test成功
- [x] Clippy warning 0件

2026-07-27、ユーザー承認によりPhase 2 Fix。Phase 2をコミット後、Phase 3へ進む。
