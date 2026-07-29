# Codex Agent Compatibility Phase 7 Vision 作業ログ

## 2026-07-29

状態: Fix承認済み（Codex CLIのTool自動選択はモデル依存事項として許容）

### 目的

Codex CLIの画像添付をHoshikage Responses APIから既存のmanaged llama-server Vision経路へ接続し、モデル能力表示と実際の受付能力を一致させる。

### 対象

- Responses `input_image`のJPEG/PNG Data URI
- `function_call_output.output`の文字列およびContent Item配列
- Codex CLI 0.145.x `view_image`結果の`input_text`/`input_image`
- 非stream・stream共通のmultimodal変換
- 非Vision Bundleと不正画像の構造化エラー
- `/v1/models`、Hoshikage model metadata、server capabilityの整合
- 画像入力時のcontext fallback

並列Tool Call、reasoning Item、stateful Responsesは本項目に含めない。

### TDDで確認した失敗

- Responses wire testは既存の`unsupported_parameter`で失敗した。
- llama-server adapter testは`Responses text request contained image content`で失敗した。
- 非Vision Bundleの事前検証testは必要なerror variantとvalidation境界がなくcompile失敗した。
- 画像context fallback testはBase64を除外する見積もり関数がなくcompile失敗した。
- Tool結果配列の最初のwire testは`ToolOutputContent`と本文を分離した`ToolOutcome`がなくcompile失敗した。
- Tool結果の型変更中、一度だけtest内の`match`編集を誤りcompile失敗した。対象箇所を修正し、同じRED testを継続した。

### 実装

- `input_image`を内部`ContentPart::Image`へ正規化した。
- Data URIのBase64をdecodeし、JPEG/PNG MIME、signature、decoded sizeを検証した。
- textとimageの順序を保ったllama-server multimodal messageを構築した。
- stream/non-streamを同じ`ModelRequest`とmanaged Vision経路へ接続した。
- `mmproj`未設定または非managed runtimeを`vision_not_supported`として推論前に拒否した。
- effective Vision能力だけをモデル一覧、モデル詳細、server capabilityへ公開した。
- upstream token計測失敗時は、Base64本文を除外して画像ごとに4K token相当の保守枠を使用する。
- `ToolOutcome`を実行状態、`ToolOutputContent`を本文として分離し、文字列結果とContent Item配列を型で表現した。
- Tool結果配列を通常messageと同じcontent正規化へ接続し、`input_text`、`input_image`、複数Item、`high`、`original`を処理した。
- Tool結果内の画像をBundle Vision能力検証とllama-server multimodal変換へ接続した。
- Codex CLI 0.145.xの`view_image` request Fixtureを追加した。

### 本番反映

- release binaryを`/home/tane/bin/hoshikage`へ配備した。
- Phase 7初回更新前binaryを`/home/tane/bin/hoshikage.backup-20260729-phase7-vision`へ保存した。
- Tool結果画像対応前binaryを`/home/tane/bin/hoshikage.backup-20260729-phase7-initial-image`へ保存した。
- user `hoshikage.service`を再起動し、active、`/health`成功、配備元とbinary hash一致を確認した。
- `/v1/capabilities`は`vision=true`を返した。
- 標準Gemma Bundleの`/v1/models`は`input_modalities=["text","image"]`、context 65536を返した。

### 実機E2E

| 試験 | 結果 |
|---|---|
| go2rtc `tapo_tc70`の実JPEG取得 | PASS、1920x1080、105962 bytes |
| Responses JPEG非stream | PASS、HTTP 200、304 input tokens |
| Responses JPEG stream | PASS、正規SSE完了、295 input tokens |
| Responses実PNG非stream | PASS、HTTP 200、298 input tokens |
| 不正Base64 | PASS、HTTP 400 `invalid_image` |
| 未対応GIF MIME | PASS、HTTP 400 `invalid_image` |
| 非Vision Bundleへの画像 | PASS、HTTP 400 `vision_not_supported`、runtime起動前拒否 |
| Yatagarasu本番Codex CLI 0.145.0 `--image` | PASS、終了コード0 |
| Tool結果内JPEG `original` 非stream | PASS、HTTP 200、実画像に基づく回答 |
| Tool結果内JPEG `high` stream | PASS、正規SSE完了、実画像に基づく回答 |
| Tool結果内の不正Base64 | PASS、HTTP 400 `invalid_image` |
| Tool結果内の未対応GIF | PASS、HTTP 400 `invalid_image` |
| Tool結果内画像を非Vision Bundleへ送信 | PASS、HTTP 400 `vision_not_supported` |
| Codex CLI 0.145.0による`view_image`自動選択 | 未成立、下記参照 |

JPEG回答は手前の透明な容器、緑色の内容物、背景の緑のカーテンと額縁を認識した。
PNG回答はCodex CLIのerror画面であることを認識した。Codex CLI E2Eは
`unsloth-gemma4-12b-qat-thinking-off`とHoshikage Responses providerを使用し、
同じ実カメラframeの内容に基づく最終回答を返した。

Tool結果画像の直接E2Eでは、`function_call`と対応する`function_call_output`を同一履歴に置き、
Codex 0.145.x形式のContent Item配列から実カメラJPEGをVision推論できた。streamingは
`response.created`から`response.completed`までの規定系列を返した。

Codex CLI 0.145.0の実行では12個のFunction ToolがHoshikageへ到達したことをログで確認したが、
thinking-off 2回、thinking-on 1回の試行はいずれもモデルが`view_image`を選択せず直接回答した。
このためCLIの自動Tool選択をPASSとはしない。Hoshikage以降のTool結果画像経路は直接API、
handler test、Codex Fixtureで確認済みであり、残る不成立点はモデルによるTool選択である。

通常ログはrequest ID、model、stream、tools数、token数、経過時間、terminalだけを記録し、
画像Data URIや回答本文を記録していないことを確認した。

### 回帰結果

- `cargo fmt --all -- --check`: PASS
- `cargo clippy --all-targets -- -D warnings`: PASS
- `cargo test --all-targets`: 248 PASS、1 ignored
  - unit: 233 PASS、1 ignored
  - contract fixture: 13 PASS
  - manual parity: 2 PASS

ignored 1件は既存のローカルllama.cpp実体依存probeであり、本項目で追加したskipではない。
build時はllama.cpp header未検出によりchecked-in `src/ffi.rs`を使用する既知のbuild-script警告が
出たが、Rust/Clippy警告は0件である。

### 発生した失敗と対応

- 最初のwire testはproduct実装ではなく、成功型に`Debug`がない状態で`expect_err`を使いcompile失敗した。RED対象を変えずにerror取り出しを修正した。
- Clippy初回はtestの`.err().expect()`を警告として拒否した。成功型が`Debug`可能になった後、`expect_err`へ戻して0警告を確認した。
- 最初の実JPEG request組立はBase64全体を`jq --arg`の単一OS引数へ展開し、`Argument list too long`で失敗した。Base64をprivateな一時fileから読み込む方式へ変更し、同じ画像で成功した。
- 上記shell失敗時に不完全payloadがHoshikageへ送られHTTP 400になった。Vision推論は開始されていない。

### 残存事項

- feature branchのNAS push、main merge
