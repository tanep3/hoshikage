# Codex Agent Compatibility Phase 4 作業ログ

## 2026-07-27

状態: Phase 4 Fix

### 目的

Responses APIのTextおよび単一Function CallをSSEで配信し、Codex CLIのAgent Loopを
Hoshikageのローカルモデルへ接続する。

### 開始時の方針

- Response state machineを推論・HTTP Wireから分離する
- llama-server SSEを外部へpass-throughしない
- Native Tool streamは出力種別確定後にのみResponses output eventを開始する
- Generic JSON Strategyは分類完了までbufferする
- client disconnect時はupstream streamとRuntimeLeaseを同じ所有単位でdropする
- 成功、失敗、切断を異なるterminal stateとして扱う
- boundedまたはpull-based streamとし、unbounded channelを使わない

### 実装

#### Responses SSE境界

- `ResponseMachine`を追加し、`New -> InProgress -> Completed|Failed`を明示した
- Textは9種類、Function Callは7種類の必須イベント順を型で生成する
- Response ID、Item ID、Call ID、index、sequence numberを状態機械が一貫して管理する
- failureは`error`と`response.failed`で終端し、`response.completed`を生成しない
- Wire serializerを状態機械から分離し、Codex 0.144.x fixtureとの完全一致を検証した

#### llama-serverストリーム境界

- llama-serverのSSEをResponses SSEへ直接pass-throughせず、`ModelDelta`へdecodeする
- 任意byte境界、UTF-8 code point途中、LF/CRLF、`[DONE]`、usage chunkを処理する
- Native Tool nameを検証してからFunction Call outputを開始する
- Tool argumentsを増分配信しつつ、byte上限とstrict JSON Schemaを完了時に検証する
- TextとFunction Callの混在、複数Tool Call、`[DONE]`欠落をfail closedとした
- Generic JSON streamは分類完了までbufferし、TextまたはFunction Callへ変換する

#### Runtimeと回復

- pull-based streamがupstream bodyと`RuntimeLease`を同じ所有単位で保持する
- client disconnect時はstream dropによりupstreamとleaseを同時に解放する
- first-token、idle、generation timeoutを個別に適用する
- Nativeの意味エラーは、外部actionを未送信かつBundleが`fallback=json`の場合に一度だけ
  Generic JSONへ再試行する
- 部分出力後、transport failure、timeoutは再試行せずfailure eventへ変換する
- JSON再試行もcontext preflight、timeout、最終Tool Schema検証を通す

#### HTTPとCapability

- `POST /v1/responses`の`stream=true`を`text/event-stream`で返す
- setup前の失敗はOpenAI互換JSON error、開始後の失敗はterminal SSE eventで返す
- `/v1/capabilities`とモデル詳細の`streaming`を`true`へ更新した
- `parallel_tool_calls`は引き続き`false`である

### 実機End-to-End

環境:

- Codex CLI `0.144.5`
- llama-server `10075` (`b10075-76f46ad29`)
- Bundle `unsloth-gemma4-12b-qat-thinking-off`
- context 16,384 tokens
- Hoshikage `127.0.0.1:3031`

結果:

1. Text SSE: `codex exec "Return exactly the word OK."`が`OK`を返した
2. Native Tool: Codexが`ls -R`を実行した
3. Tool Result継続: Codexが`cat README.md`を実行し、結果をHoshikageへ再投入した
4. 複数step: 一覧取得、README読取、タイトル回答まで完走した
5. Generic JSON: 一時Bundle設定でも同じAgent Loopが完走した
6. Hoshikageの推論upstreamはローカルllama-serverのみを使用した

Generic JSON試験ではモデルが同じシェルToolを4回選択してから最終回答した。Hoshikageは
各Function Callをデータとして返しただけであり、実行と継続判断はCodexが行った。

### テスト結果

- `cargo test`: unit 194成功、0失敗、1 ignored
- contract fixture: 12成功、0失敗
- doc test: 0失敗
- `cargo clippy --all-targets --all-features -- -D warnings`: 成功
- `cargo fmt --check`: 成功
- ignoredは既存の手動実機probe `probe_local_llama_cpp_bundle`
- build scriptはllama.cpp headers未検出時にchecked-in `src/ffi.rs`を使用する既知通知を出す

### 発生した失敗と修正

- 状態機械のTDD初回は未実装stubのため期待どおり失敗し、イベント遷移実装後に成功した
- 状態機械実装時のID借用競合を、遷移前のID cloneで解消した
- SSE formatterのfixture差分を修正した
- typed upstream error不足を追加し、Gatewayへ安定codeとして写像した
- llama-server fixture末尾のblank lineなし`[DONE]`をEOF frameとして処理した
- Gateway編集時の余分なbraceとFake Gatewayのstream契約不足を修正した
- serviceのstream domain import不足を修正した
- 初回Codex E2Eはsandbox内loopback制限により5回再接続後に失敗した。ローカル通信を
  許可した同一試験で成功し、Hoshikage障害ではないことを確認した
- JSON再試行実装の初回compileはerror値の所有権移動順で失敗し、ログ記録順を修正した
- `cargo fmt --check`が追加コードの整形差分を検出し、`cargo fmt`後に成功した
- 最終ClippyでJSON再試行関数の引数過多を検出し、上限とtimeoutを
  `BufferedRetryLimits`へ構造化した
- errorなしで未完了のupstream streamが閉じるテストを追加し、初回はterminal event欠落で
  失敗した。Service境界で`upstream_disconnected`へ終端するよう修正した

### 既知制限

- 並列Tool Callは未対応で、Capabilityも`false`
- Native意味エラーでも、部分出力開始後はJSONへ再試行しない
- client disconnectは所有権dropで中止するため、切断後のSSE error eventは送信不能である
- Vision、上位Skill、Yatagarasu統合は後続Phaseで検証する
