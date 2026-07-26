# Codex Agent Compatibility Phase 0 契約観測報告

## 1. 文書状態

| 項目 | 値 |
|---|---|
| 対象Phase | Phase 0: 契約観測 |
| 観測日 | 2026-07-27 |
| 状態 | Phase 0 Fix |
| Codex CLI | `0.144.5` |
| llama-server | build `10075` (`b10075-76f46ad29`) |
| Hoshikage branch | `feature/codex-agent-compatibility` |

本書は実装前の外部契約を、実際のCodex CLIとllama-serverで観測した結果である。
観測値と推奨値を区別し、未承認の推奨値は正式要件として扱わない。

## 2. 観測環境

### 2.1 Codex

- カスタムProviderを`wire_api = "responses"`、`requires_openai_auth = false`で接続した。
- localhost上のCapture Serverへ`POST /v1/responses`を送信させた。
- Responses SSEで通常応答、Function Call、Function Call Output再入力を完走した。
- CodexによるTool実行は、無害な`printf phase0-tool-ok`で確認した。

### 2.2 llama-server

- managed binary: Hoshikage設定配下の`llama-server`
- build: `b10075-76f46ad29`
- context: 32,768 tokens
- backend: CPU
- slot数: 4
- Hoshikage側の推論同時実行数は、設計どおり1を前提とする。

CUDA初期化は、ドライバがCUDA runtimeより古いため失敗した。そのため本Phaseの
モデル出力契約はCPUで検証し、GPU性能値は取得していない。

## 3. Codex Responses入力契約

### 3.1 Top-level

Codex `0.144.5`が送信した主要値は次のとおりである。

| Field | 観測値 |
|---|---|
| `stream` | `true` |
| `store` | `false` |
| `tool_choice` | `"auto"` |
| `parallel_tool_calls` | `false` |
| `reasoning` | `null` |
| `text` | `null` |
| `include` | `[]` |

`client_metadata`、`prompt_cache_key`も送信された。これらは互換モードで受理し、
意味を変更せずに無視またはログ用メタデータへ分類できる。

### 3.2 Input Item

初回リクエストは、developer context、environment context、ユーザー入力を
`message` Itemとして送信した。

Tool実行後のリクエストでは、既存のmessage列に次を追加した。

1. `function_call`
2. 同じ`call_id`を持つ`function_call_output`

`function_call.arguments`はJSON objectではなくJSON文字列だった。
`function_call_output.output`は、終了コード、実行時間、標準出力等を含む単純文字列だった。

### 3.3 Tool定義

通常の`codex exec`でも、次のTool Typeが同時に送信された。

| Type | Top-level数 | 内容 |
|---|---:|---|
| `function` | 5 | shell、stdin、plan、user input、image view |
| `namespace` | 1 | `multi_agent_v1`配下に5 Tool |
| `web_search` | 1 | `external_web_access = false` |

これは現行設計の「未知Input Item / Tool Typeは常にreject」と衝突する。
`namespace`と`web_search`を拒否すると、Toolを使わない通常応答もリクエスト受付時に失敗する。

### 3.4 Request size

compact JSONでの観測値は次のとおりである。

| 対象 | Bytes |
|---|---:|
| request body | 43,813 |
| `instructions` | 21,210 |
| `input` | 4,016 |
| `tools`全体 | 17,635 |
| `function` Toolのみ | 5,481 |

Qwen chat templateへ変換した場合、全体は7,025 tokens、Toolなしは5,400 tokensだった。
5つのFunction Toolによる増分は1,625 tokensである。`namespace`と`web_search`を
モデル入力へ変換したtoken数は、この測定に含まれない。

## 4. Responses SSE契約

Capture Serverが送信し、Codexが受理したイベント順をFixtureへ固定した。

### 4.1 Text

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

### 4.2 Function Call

```text
response.created
response.in_progress
response.output_item.added
response.function_call_arguments.delta
response.function_call_arguments.done
response.output_item.done
response.completed
```

同一Response内でresponse ID、item ID、call ID、indexを維持し、
`sequence_number`を0から単調増加させる必要がある。

## 5. llama-server契約

### 5.1 Native Bundle

| 項目 | 値 |
|---|---|
| Model | Qwen3.5-0.8B-Q4_K_M |
| Strategy候補 | `llama-server-native` |
| Native tools | 対応 |
| Object arguments入力 | 対応 |
| Parallel Tool Calls template | 対応 |

非streamでは`finish_reason = "tool_calls"`となり、Function argumentsはJSON文字列で返った。
Tool Result再入力では、assistant側argumentsをobjectにした入力と、`role = "tool"`および
`tool_call_id`の組み合わせをchat templateが受理した。

streamではTool名と最初のarguments断片が同一deltaに入り、その後のargumentsは任意位置で
分割された。Tool完了chunkの後、`choices = []`とusageを持つ独立chunkが送信され、最後に
`data: [DONE]`が送信された。

thinking有効時は、短い`max_tokens`をreasoningだけで消費して`finish_reason = "length"`となった。
Codex Bundleではthinkingの扱いと出力token予算を明示する必要がある。

### 5.2 Generic JSON Bundle

| 項目 | 値 |
|---|---|
| Model | LFM2.5-1.2B-Instruct-Q4_K_M |
| Strategy候補 | `generic-json` |
| `json_schema`出力 | 対応 |
| Native Tool Call出力 | 非対応 |

Function CallとFinal Answerを選択できるschemaでは、モデルはToolを呼ばず根拠のないFinal Answerを
選択した。required相当としてFunction Callだけを許すschemaでは、正しいTool名とargumentsを持つ
JSONを生成した。

したがってこのBundleはJSON構文契約の検証には使えるが、`tool_choice = auto`におけるTool選択品質を
保証するBundleではない。Bundle診断では「形式対応」と「判断品質」を分けて表示すべきである。

### 5.3 標準利用予定Gemma 4 Bundle

ユーザーが標準利用を予定する次のBundleを追加観測した。

| 項目 | 値 |
|---|---|
| Bundle | `unsloth-gemma4-12b-qat-thinking-off` |
| Model | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` |
| 起動context | 16,384 tokens |
| slot | 1 |
| backend | CPU |
| llama-server Native tools | 対応 |
| Generic JSON `json_schema` | 対応 |

観測結果:

1. Native required Tool Callは`finish_reason = "tool_calls"`となり、正しいTool名とJSON文字列argumentsを返した。
2. Tool Result再入力後、Tool結果に基づく正しい最終回答を返した。
3. Native streamは任意分割されたarguments、Tool完了chunk、独立usage chunk、`[DONE]`を返した。
4. Generic JSONのauto相当で、自発的に`read_file` Function Callを選択した。
5. Unicode、空白、integer、boolean、array、nested objectを含むrequired schemaで、意味と型を保った有効JSONを返した。

複合JSONの最初の試験では`chat_template_kwargs.enable_thinking = false`を指定しなかったため、
256 output tokensを`reasoning_content`だけで消費し、`finish_reason = "length"`となった。
同指定を追加すると正しい複合JSONを生成した。Bundleの`thinking.mode = "off"`は、
Adapterがllama-server requestへ明示的に写像しなければならない。

Gemma tokenizerでCodex実リクエストを測定した結果は次のとおりである。

| 対象 | Tokens |
|---|---:|
| 5 Function Toolを含む初回入力 | 6,871 |
| Toolなし | 5,535 |
| Function Tool overhead | 1,336 |

現在のBundle設定8,192 tokensでは初回から残りが約1,300 tokensしかない。Codex互換Bundleとしては
16Kを最低保証候補、32Kを推奨とし、現Bundleのcontext設定を引き上げる必要がある。

## 6. Contract Fixture

次をrepositoryへ格納した。

```text
tests/fixtures/
  codex/0.144.x/
  llama-server/10075/
```

Codex Fixtureはモデル名、instructions、client ID、developer context、environment contextを置換した。
Tool schema、Input Item構造、検証用Tool arguments、Tool Result形式は保持した。
不正arguments Fixtureだけは、観測済みNative応答から終端delimiterを除いた合成変異であり、
`provenance = "synthetic-mutation"`を明記した。

## 7. 確定既定値

実測値、単一生成、LAN上の少数クライアント利用を前提とし、P0-D02として承認された。

| 設定 | 推奨値 | 根拠 |
|---|---:|---|
| request body上限 | 8 MiB | 実測44KBに余裕を持ち、4MiB Tool Resultも収容 |
| Tool Schema合計 | 1 MiB | 実測18KB。異常なschema膨張を推論前に拒否 |
| 単一Tool Schema | 256 KiB | 通常schemaに十分で、1 Toolによる占有を制限 |
| Tool数 | 128 | Codex実測はtop-level 7、展開後10 |
| Tool arguments | 64 KiB | 既存Bundle例と一致 |
| Tool Result | 4 MiB | 既定は切り詰めず、超過を明示エラー |
| Queue capacity | 4 | 単一利用中心で、古い要求を溜めすぎない |
| Queue timeout | 30秒 | Agent Loopは逐次。別利用者の長時間生成を無期限待機しない |
| Request timeout | 900秒 | Queue、load、generation、変換を包含 |
| First token timeout | 120秒 | 大型Bundleのload/prompt evaluationを許容 |
| Stream idle timeout | 120秒 | reasoningや低速CPU生成を考慮 |
| Generation timeout | 600秒 | ローカル生成の暴走を制限しつつ長文を許容 |

要件定義書D-026とシステム設計15.2へ反映済みである。

## 8. 未決事項

### P0-D01 Codex補助Tool Type

**A: 現行要件どおりreject**

通常のCodexリクエストも失敗するため、非推奨。

**決定: B。既知のCodex補助Toolとして受理し、初期版ではモデル入力から除外**

`namespace`と`web_search`はcompatible時にwarning付きで受理し、Function Toolだけをモデルへ渡す。
通常Agent Loopは利用できる一方、除外されたToolはローカルモデルから選択できない。
strict時は`unsupported_tool_type`として明示エラーにする。

### P0-D02 既定値

**決定: 承認。** 第7章の値をPhase 1以降で使用する正式な既定値とする。

### P0-D03 検証Bundle

**決定: 承認。** 標準製品Bundleを`unsloth-gemma4-12b-qat-thinking-off`とし、
同じGGUFでNative主経路とGeneric JSON fallback経路の両方を検証する。

運用Bundleは次の設定を基本とする。

```yaml
tool_calling:
  mode: native
  parser: llama-server-native
  fallback: json
  strict: true
  repair_invalid_json: true
```

検証時はNativeとGeneric JSONを個別に強制し、Strategyごとの成否を分離する。QwenとLFMのFixtureは、
標準モデルではなく異種chat templateに対するAdapter回帰用として残す。LFMのauto Tool選択品質を
製品保証へ含めない。

Gemma 4のCodex互換Bundleは16Kを最低保証、32Kを推奨とし、現在の8K設定を互換対象にしない。

## 9. 失敗・未実施

隠さず記録する。

1. sandbox内のlocalhost待受と接続は権限制約で失敗し、承認済みの外部実行で再検証した。
2. 最初のcontext測定用変換式に構文誤りがあり、空入力による無効なupstream probeを発生させた。式を修正し、7,025 tokensを再測定した。
3. Qwenのthinking有効Text probeはtoken上限で終了した。thinking無効でwire shapeを確認した。
4. LFMのauto相当probeはFunction Callを選択しなかった。required相当probeは成功した。
5. CUDAドライバ不整合によりGPU推論、GPU latency、VRAM測定は未実施。
6. 検証用port `18081`が使用中だったため、LFM検証は`18082`で実施した。

## 10. 結論

Responses SSEによるTextと単一Tool Loop、llama-server Native Tool Calling、Generic JSON出力は
実装可能であり、主要なwire shapeをFixtureとして固定できた。

Codex補助Tool Type、運用上限、標準Bundle、context保証の未確定事項は解消した。
Gemma 4 Fixtureと正式な互換性マトリクスへ反映し、Phase 0をFixする。
