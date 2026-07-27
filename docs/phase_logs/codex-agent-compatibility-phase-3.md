# Codex Agent Compatibility Phase 3 作業ログ

## 2026-07-27

状態: Phase 3 Fix

### 目的

Tool定義、単一Function Call、Function Call Output再入力を、HoshikageがToolを実行しない
構造のまま非ストリームResponses APIで成立させる。

### 開始時の決定反映

- 標準Bundle `unsloth-gemma4-12b-qat-thinking-off`の実設定を`n_ctx=8192`から
  `n_ctx=16384`へ変更
- `model_map.json`の15 Bundleと`0600`権限を維持
- managed llama-serverの実効値`n_ctx_slot=16384`を確認
- MTP、Vision、thinking-off、`n_gpu_layers=99`を維持
- 非ストリームResponses実モデル試験で`OK`、usage 19/2/21 tokens

### 作業中の失敗・警告

- Phase 2コミット時、sandboxの読み取り専用Git管理領域により`git add`が失敗した。
  承認済み権限で同一対象を再実行し成功した。
- `jsonschema`依存取得はsandbox内のDNS制限で失敗した。承認済みの`cargo test`で取得し、
  以後のbuildとtestは成功した。
- 開発中に`cargo test`へ複数filterを渡す誤った実行を1回行い、test開始前にCargoが拒否した。
- Context Planの最初の実モデル試験では、description未指定のFunction Toolを`null`として
  llama-serverへ送ったため、token計数と推論の双方がHTTP 500となった。optional fieldを
  省略するAdapterへ修正し、回帰testを追加した。
- 最終実モデル試験の起動時、既存Hoshikageが`3030`を利用中だったため新processは
  `Address already in use`で終了した。既存processを停止せず、試験serverを`3031`へ分離した。
- LAN公開用の`HOST=0.0.0.0`を継承した試験起動は、Bearer Token未作成のため
  `at least one bearer token is required`で3回fail-closedした。運用設定を変更せず、
  試験processだけを`127.0.0.1`へ限定して再実行した。
- 16K確認用一時serverは検証終了時のSIGINTにより終了code 1となった。異常終了ではなく
  明示停止であり、Hoshikageとmanaged llama-server processが残っていないことを確認した。
- llama.cpp headerが見つからないため、checked-in FFI bindingを使用した。

### 実装

- Model Bundleへ`ToolCallingConfig`を追加
  - mode: `native` / `json` / `disabled`
  - parser: `llama-server-native` / `generic-json`と将来parser ID
  - fallback、strict validation、決定的JSON修復、arguments上限
  - Tool Resultの既定`reject`と明示的なUTF-8安全head/tail policy
- 設定のない既存Bundleを`disabled`として後方互換を維持
- Function Tool、`auto` / `none` / `required` / named choiceをWire境界で正規化
- Tool名、重複、Draft 7 JSON Schema、個別・合計bytes、件数上限を推論前に検証
- `function_call`と`function_call_output`をCall ID付きDomain型へ変換
- orphan、重複Call、重複Result、不正argumentsをConversation validationで拒否
- 完全履歴がある場合だけ`previous_response_id`をwarning付きで無視し、
  履歴不足は`previous_response_not_supported`で拒否
- Native Strategyでllama-server Chat Tool形式へ変換し、単一Function Callを解析
- Generic JSON Strategyで動的JSON Schema grammarを生成
- JSON修復をcode fence除去とtrailing comma除去だけに限定
- semantic regeneration budgetを1 Responseにつき1回に統一
  - Native解析、複数Call、required違反、strict Schema違反はJSON fallback
  - JSON primary失敗はJSON形式で1回だけ再生成
- Hoshikageが`fc_` item IDと`call_` Call IDを割り当て、Toolを実行しない
- llama-serverの`/v1/chat/completions/input_tokens`で実promptを事前計数
- token計数APIを利用できない場合は過小評価しない保守的Context Planへ退避
- `/v1/capabilities`とHoshikageモデル詳細へTool能力を公開
- 実効mode、parser、strict、fallback、Tool件数を本文なしでログ出力
- 既存Text AdapterをTool履歴も扱うChat Adapterへ一般化

### 実モデルE2E

標準Bundle `unsloth-gemma4-12b-qat-thinking-off`を使い、外部APIを使用せず
Hoshikageとmanaged llama-serverだけで確認した。

1. `tool_choice=required`で`read_file`を提示
2. モデルが`{"path":"README.md"}`を持つ単一Function Callを返却
3. Hoshikageが`call_9f71...`を割り当て、Toolを実行せずCodex相当clientへ返却
4. 同じCall IDの`function_call_output`を完全履歴として再投入
5. モデルがREADMEのタイトルを最終Textとして回答

| 項目 | 結果 |
|---|---:|
| 初回usage | input 80 / output 17 / total 97 |
| 継続usage | input 120 / output 31 / total 151 |
| 実効context | 16,384 |
| Strategy | Native / `llama-server-native` |
| strict / fallback | `true` / JSON |
| MTP / Vision | 維持 |

別試験では標準Bundleを一時的にJSON primaryへ切り替え、required Function Callを確認後、
Native + JSON fallbackへ復元した。`model_map.json`の15 Bundleと`0600`権限は維持した。

### Context・異常系

- `max_output_tokens=16350`と実入力約64 tokensの組合せを、生成前に
  `context_length_exceeded`で拒否
- description未指定Toolを省略形式で送り、llama-server token計数・推論の双方が成功
- Native複数Callを`MultipleToolCalls`として検出し、semantic recovery対象へ変換
- 不正JSON、Schema不一致、Tool Choice違反、disabled Bundleを構造化errorへ変換
- Tool Result head/tail policyはUTF-8境界を壊さず、切り詰め情報をモデル入力へ付与

### テスト

- `cargo test`: 成功
  - unit test: 176 passed、0 failed、1 ignored
  - contract test: 12 passed、0 failed
  - doc test: 0 failed
- `cargo clippy --all-targets --all-features -- -D warnings`: 成功、Clippy warning 0件
- `cargo fmt --check`: 成功
- Responses Wire変換p95: 50ms未満
- `doctor --model unsloth-gemma4-12b-qat-thinking-off --json`
  - error 0
  - 既存Thinking adapter警告2件

既存の実機依存`probe_local_llama_cpp_bundle` 1件は従来どおりignored。

### Phase 3完了条件

- [x] Function ToolとTool Choiceを内部契約へ変換
- [x] NativeとGeneric JSON Strategyを実装
- [x] 単一Function CallをResponses Output Itemとして返却
- [x] Function Call OutputをCall ID付き履歴として再投入
- [x] HoshikageがToolを実行しないことをHTTP統合testで保証
- [x] invalid Tool Callを決定的修復、最大1回再生成、構造化errorで処理
- [x] 標準Bundleの16K化と実効値確認
- [x] 実モデルの非ストリームTool Loop完走
- [x] Chatを含む全回帰test成功
- [x] Clippy warning 0件

Phase 4のSSE、Codex CLIによる最終Agent Loop、Vision、並列Tool Callは未実装であり、
Capabilityでも有効と表明しない。

2026-07-27、ユーザー承認によりPhase 3 Fix。Phase 3をコミット後、Phase 4へ進む。
