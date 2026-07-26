# Codex Agent Compatibility Phase 0 作業ログ

## 2026-07-27

状態: Phase 0完了

### 完了

- Codex CLI `0.144.5`のResponses requestをCapture
- Text SSEをCodexで受信確認
- Function CallをCodexで実行確認
- Function Call Output再入力をCapture
- llama-server build `10075`のNative Tool Callを確認
- Native Tool streamのdelta分割とusage chunkを確認
- Tool Result再入力を確認
- Qwen3.5-0.8B-Q4のNative経路を確認
- LFM2.5-1.2B-Instruct-Q4のGeneric JSON required経路を確認
- `unsloth-gemma4-12b-qat-thinking-off`のNative Tool Callを確認
- 同Gemma 4のTool Result継続とNative streamを確認
- 同Gemma 4のGeneric JSON auto Tool選択を確認
- 同Gemma 4で複合JSON argumentsの意味・型保持を確認
- Gemma 4 tokenizerでCodex初回入力6,871 tokensを測定
- Codex context overheadを測定
- 匿名化Fixtureを追加
- Fixture契約テスト12件を追加し、全件成功
- `cargo fmt --check`成功
- 全回帰テスト成功
  - 既存unit test: 100 passed、0 failed、1 ignored
  - Phase 0 contract test: 12 passed、0 failed
  - doc test: 0 failed

### テスト時の警告・除外

- llama.cpp headerが見つからないため、既存のchecked-in FFI bindingを使用した。
- 既存の`probe_local_llama_cpp_bundle` 1件は実機依存testとしてignoredのままである。
- Phase 0中の`cargo fmt --check`は、新規testの改行差分で2回失敗した。いずれも`cargo fmt`適用後の再実行は成功した。

### 確定

- `namespace`と`web_search`はcompatible時にwarning付きで除外し、strict時に明示エラーとする。
- Phase 0報告第7章のsize、queue、timeout既定値を採用する。
- 標準Bundleを`unsloth-gemma4-12b-qat-thinking-off`とする。
- 同一Gemma 4でNative主経路とGeneric JSON fallback経路を個別検証する。
- Qwen/LFMは異種Adapterの補助回帰Fixtureとする。
- Codex互換contextは16Kを最低保証、32K以上を推奨し、8Kを対象外とする。

### 未実施

- CUDAドライバ不整合のためGPU実機測定は未実施
- Phase 1以降の製品コード実装

### 追加観測時の失敗

- Gemma 4複合JSONの初回試験はthinking-offをupstreamへ指定せず、reasoningだけで256 tokensを消費して失敗した。
- `chat_template_kwargs.enable_thinking = false`を明示した再試験では、複合JSONを正しく生成した。

詳細は`docs/research/codex-agent-compatibility-phase-0.md`を参照する。
