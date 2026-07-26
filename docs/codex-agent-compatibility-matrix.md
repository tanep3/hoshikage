# Hoshikage Codex Agent Compatibility Matrix

**更新日:** 2026-07-27
**状態:** Phase 0 Fix

## 1. Codex CLI

| Codex CLI | Text SSE | Function Call | Function Call Output | 複数step前提 | 状態 |
|---|---:|---:|---:|---:|---|
| `0.144.5` | Pass | Pass | Pass | Wire契約確認済み | 検証済み |
| `0.144.x`の他patch | - | - | - | - | 未検証 |
| 他minor系列 | - | - | - | - | 未検証 |

`0.144.5`は実際の`codex exec`をカスタムResponses Providerへ接続して検証した。
「複数step前提」はCall/Result再入力のwire shapeを確認したことを表し、Hoshikage製品実装の
End-to-End完了を表さない。

## 2. llama-server

| Build | Native Tool | Native Stream | Tool Result | `json_schema` | 状態 |
|---|---:|---:|---:|---:|---|
| `10075` (`b10075-76f46ad29`) | Pass | Pass | Pass | Pass | 検証済み |
| 他build | - | - | - | - | 未検証 |

Native streamではargumentsが任意位置で分割され、Tool完了後に`choices = []`のusage chunkと
`data: [DONE]`が送信される。

## 3. 標準Bundle

| 項目 | 値 |
|---|---|
| Bundle | `unsloth-gemma4-12b-qat-thinking-off` |
| GGUF | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` |
| Primary Strategy | `native` |
| Parser | `llama-server-native` |
| Fallback Strategy | `json` |
| Thinking | `off` |
| 最低保証context | 16,384 tokens |
| 推奨context | 32,768 tokens |

| 検証項目 | 結果 |
|---|---:|
| Native required Function Call | Pass |
| Native arguments JSON | Pass |
| Function Call Output後の最終回答 | Pass |
| Native Tool stream | Pass |
| Generic JSON auto Tool選択 | Pass |
| Generic JSON複合arguments | Pass |
| Unicode・array・boolean・integer・nested object保持 | Pass |
| Thinking-off反映なし | Fail |
| Thinking-off明示後 | Pass |

Hoshikage AdapterはBundleの`thinking.mode = "off"`を、llama-server requestの
`chat_template_kwargs.enable_thinking = false`へ明示的に写像しなければならない。

Gemma tokenizerによるCodex初回入力は、5 Function Toolを含む場合6,871 tokensだった。
8K contextは実用余裕がないためCodex互換対象外とする。32Kは推奨値だが、Phase 0環境では
CUDAドライバ不整合によりGPU VRAMとlatencyを未検証である。

## 4. 補助Bundle

| Bundle | Strategy | 検証結果 | 製品上の位置づけ |
|---|---|---|---|
| Qwen3.5-0.8B-Q4_K_M | Native | Pass | 異種Native template回帰 |
| LFM2.5-1.2B-Instruct-Q4_K_M | Generic JSON required | Pass | 異種JSON template回帰 |
| LFM2.5-1.2B-Instruct-Q4_K_M | Generic JSON auto | Tool未選択 | 標準・品質保証対象外 |

補助BundleはAdapter一般性の回帰用であり、標準モデルまたは同等品質を表明しない。

## 5. 既知制限

- `namespace`と`web_search`はcompatible時にwarning付きで受理し、初期版ではモデル入力から除外する。
- strict時の未対応補助Toolは`unsupported_tool_type`とする。
- parallel Tool Callは初期版では最大1 Callへ制約する。
- GPU性能、VRAM、32K contextのGPU実機値は未検証である。
- Vision、MTP、上位SkillのEnd-to-Endは後続Phaseで検証する。
