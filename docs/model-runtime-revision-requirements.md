# Hoshikage Model Runtime Revision Requirements

**作成日:** 2026-07-21  
**位置づけ:** 改訂要件定義  
**対象:** 一般的な Gemma 系 12B クラスの GGUF モデルを含む、複数モデルで再利用可能な推論基盤

---

## 1. 目的

Hoshikage の推論基盤を、従来のテキスト専用 GGUF 実行から、次のモデル機能に対応できる形へ拡張する。

- QAT などの量子化済みモデル運用
- Multi-Token Prediction (MTP)
- Draft model を用いた speculative decoding
- Vision input を含む multimodal 推論
- Thinking mode 制御

本ドキュメントでは、今回の改訂で満たすべき要件、用語整理、受け入れ条件、確定した運用方針を定義する。システム設計は `docs/model-runtime-revision-system-design.md` で扱う。

---

## 2. 方針

### 2.1 モデル非依存の記述

要件、設定、ユーザー向け文書、ログ、CLI 表示では、特定の派生モデルに依存した説明を避ける。

Hoshikage は、一般的な GGUF モデル、Gemma 系モデル、Vision 対応モデル、speculative decoding 対応モデルを扱える推論サーバーとして整理する。

### 2.2 llama.cpp 最新機能への追従

MTP、Draft model、Vision などは llama.cpp 側の実装更新に依存する。Hoshikage は固定バンドルされた古い llama.cpp だけを前提にせず、ユーザーが新しい llama.cpp runtime を導入できる運用を前提にする。

今回の主 runtime は `llama-server` を Hoshikage が管理する方式とする。Hoshikage は `llama-server` を子プロセスとして起動・監視し、OpenAI 互換 API、モデル登録、診断、RAM ディスク、status、fallback 導線を提供する。`libllama` FFI は既存互換または限定的な補助経路として扱い、最新モデル機能の第一経路にはしない。

### 2.3 実装より先に運用要件を固める

要件定義では、どの機能をユーザー体験・設定・検証項目としてサポートするかを定める。Rust 側の詳細構造と runtime backend 境界はシステム設計書で扱う。

### 2.4 llama.cpp runtime の推奨運用

推奨運用は、ユーザーが自分の環境に合った llama.cpp runtime を用意し、Hoshikage が指定する runtime directory に配置する方式とする。

Hoshikage は次を提供する。

- runtime directory の標準配置先
- `llama-server` / `llama-cli` / 依存 shared library の存在診断
- `llama-server --version` による version 確認
- CUDA backend の存在確認
- モデル登録時・起動時・推論時の runtime 整合性診断
- 不足時の修正行動の案内

Hoshikage は初期段階では全環境の自動ビルドを責務にしない。導入の楽さは、公式配布物の取得手順、source build 手順、配置先、診断コマンドを明確にすることで担保する。

---

## 3. 用語整理

### 3.1 QAT

QAT は Quantization-Aware Training の略で、量子化後の推論品質を考慮して学習または調整されたモデルを指す。

Hoshikage における QAT 対応は、QAT の学習処理を実装することではなく、QAT 済みまたは量子化済み GGUF モデルを通常のモデル候補として扱い、ロード・実行・設定できることを意味する。

### 3.2 MTP

MTP は Multi-Token Prediction の略で、複数トークンを先読みして推論速度を改善するための仕組みである。

llama.cpp では speculative decoding の方式の一つとして扱われ、`draft-mtp` が指定可能である。MTP は、単純な別小型モデルによる draft とは区別して扱う。

### 3.3 Draft Model

Draft model は、target model より軽量なモデルが候補トークンを生成し、target model がそれを検証する speculative decoding の代表的な方式である。

Hoshikage では、MTP と Draft model を混同せず、どの方式を有効化しているかを設定・ログ・検証で区別できるようにする。

### 3.4 Vision / Multimodal

Vision 対応モデルは、テキストだけでなく画像入力を含むメッセージを処理する。llama.cpp では multimodal projector、つまり `mmproj` GGUF を伴う構成がある。

Hoshikage では、OpenAI 互換 Chat Completions の message content が文字列だけでなく、画像入力を含む構造化 content に拡張される可能性を前提にする。

### 3.5 Thinking Mode

Thinking mode は、モデルが最終回答の前に内部推論用の thought block を生成する挙動を指す。複雑な推論やエージェント用途では有効だが、音声会話 BOT では応答遅延が大きくなる。

Hoshikage では、Thinking をモデルごとに無効化できるようにする。省略時はモデル・chat template・runtime の既定動作に任せる。

---

## 4. 機能要件

### MR-001: 量子化済み GGUF モデル運用

**優先度:** 高

Hoshikage は、QAT 済みモデルを含む量子化済み GGUF モデルを通常のモデルとして登録・ロード・実行できる。

**受け入れ条件:**
- 既存の単一 GGUF モデル登録が維持される。
- 量子化方式に依存した特殊分岐をユーザーに強制しない。
- モデルごとのメタデータ確認結果をログまたは状態確認で追跡できる。

### MR-002: MTP 対応

**優先度:** 高

Hoshikage は、llama.cpp が MTP に対応している環境で、MTP を有効化した推論設定を扱える。

**受け入れ条件:**
- MTP 有効/無効をモデル設定で指定できる。
- MTP が利用できない llama.cpp では、明示的なエラーまたは無効化ログを出す。
- MTP は Draft model とは別の方式として扱う。
- 検証時に、MTP 有効時の起動コマンドまたは runtime 引数を確認できる。

### MR-003: Draft Model 対応

**優先度:** 中

Hoshikage は、target model と draft model の組み合わせを設定できる。

**受け入れ条件:**
- target model と draft model のパスまたは Hugging Face repo 指定を区別できる。
- draft model が未指定の場合は従来通り通常推論を行う。
- draft model のロード失敗時に target model の失敗と区別できるエラーを返す。
- draft model 使用時の token acceptance など、llama.cpp が出力する統計情報をログに残せる。

### MR-004: Vision 入力対応

**優先度:** 高

Hoshikage は、Vision 対応 GGUF モデルに対して、画像を含む入力を受け付けられるようにする。

**受け入れ条件:**
- Chat message content が文字列のみの既存形式と互換である。
- 画像入力を含む OpenAI 互換形式を受け付ける。
- `mmproj` が必要なモデルでは、メインモデルと projector の対応関係を設定できる。
- projector 未指定または不整合の場合は、推論前に明示的なエラーを返す。
- 画像なしの通常チャットは既存通り動作する。

### MR-005: llama.cpp 導入方式の刷新

**優先度:** 高

Hoshikage は、ユーザーが最新の llama.cpp runtime を導入し、Hoshikage が管理できる形で配置できる手順を文書化する。

**受け入れ条件:**
- Linux、macOS、Windows の導入手順を文書化する。
- `llama.app` の installer と GitHub Releases の prebuilt binaries の違いを説明する。
- Hoshikage が管理する runtime directory に `llama-server` と `llama-cli` を配置する手順を説明する。
- 標準 runtime directory は `~/.config/hoshikage/llama.cpp` とし、開発時は runtime directory override で任意の外部ビルド成果物を診断対象にできる。
- 旧 runtime 配置は自動探索対象にしない。`doctor` が旧配置らしきファイルを検出した場合は、新標準配置への移行案内を出す。
- `llama-server` が必要とする shared library 一式を同じ runtime directory または platform 標準の探索パスから解決できるようにする。
- `llama-cli` は単体動作確認と `doctor` 補助用、`llama-server` は Hoshikage の主 runtime 用として役割を区別する。
- Linux CUDA では source build を正式な導入手順として扱う。
- `LD_LIBRARY_PATH` / `PATH` / platform 固有の配置先を整理する。
- `hoshikage doctor` は runtime directory、`llama-server --version`、CUDA backend、Vision / MTP / Draft model に必要な起動 option の利用可否を確認する。
- runtime が未配置または不整合の場合、Hoshikage は推論前に分かる形でエラーまたは警告を出す。

### MR-006: Model Bundle 登録

**優先度:** 高

Hoshikage は、単一 GGUF ファイルではなく、推論に必要なファイルと実行設定をまとめた Model Bundle を登録・管理できる。

**方針:**
- `.env` はサーバー全体の既定値を扱う。
- モデルごとの設定は `model_map.json` に保持する。
- DB 管理は今回の要件には含めない。
- 既存の `path` / `model` / `stop` 形式は後方互換として維持する。

**Model Bundle に含める候補:**
- `main_model`: メイン GGUF モデル
- `mmproj`: Vision projector GGUF
- `draft_model`: Draft model GGUF
- `stop`: モデル別 stop sequence
- `n_ctx`: モデル別 context length
- `n_gpu_layers`: モデル別 GPU offload 設定
- `vision`: Vision 入力を許可するか
- `speculation`: MTP / Draft model / draft token上限 / fallback mode
- `thinking`: Thinking mode 制御
- `chat_template`: 必要に応じた chat template 指定

**受け入れ条件:**
- 既存の `hoshikage add <PATH> <LABEL> [STOP_WORDS]...` は従来通り動作する。
- 新しい CLI または管理 API で、`mmproj`、`draft_model`、内蔵MTP、`spec_draft_n_max`、`speculation`、`thinking`、`n_ctx`、`n_gpu_layers` を登録できる。
- `hoshikage add` は `--mmproj`, `--mtp`, `--mtp-drafter`, `--draft-model`, `--spec-draft-n-max`, `--thinking-off`, `--n-ctx`, `--n-gpu-layers` を受け付ける。
- `speculation.draft_n_max`は0より大きい整数とし、speculation modeが有効なBundleにだけ指定できる。
- `draft_n_max`未指定時は、利用するllama-server runtimeの既定値を変更しない。
- MTP と Draft model の同時指定は llama.cpp runtime の対応に準拠する。Hoshikage は独自制限を設けず、`llama-server` が受け付けない組み合わせは `doctor` / 起動時診断で明示する。
- モデルごとの設定が `.env` に散らばらない。
- `model_map.json` を読めば、そのモデルをどう実行するかが追跡できる。
- 旧 `path` field は読み込み互換を維持し、新規保存は `base_path` を使う。

### MR-007: Runtime Capability 診断

**優先度:** 高

Hoshikage は、最新モデル機能を実行できる環境かどうかを診断できる。

**診断導線:**
- `hoshikage doctor`: ライブラリ、CUDA、モデル、projector、MTP / Draft model 対応をまとめて診断する。
- `hoshikage add --check`: Model Bundle 登録時に整合性を確認する。
- サーバー起動時診断: 登録済み bundle を軽く検査し、致命的でない問題は `WARN` として出す。
- API 実行時診断: 実際のロード・推論時に発見した問題は、原因が分かるエラーとして返す。

**診断項目:**
- runtime directory の存在
- `llama-server` の存在と実行可否
- `llama-cli` の存在と実行可否
- `llama-server --version` の取得可否
- `libggml-cuda` など backend shared library の存在
- llama.cpp version
- Hoshikage が利用する `llama-server` 起動 option の利用可否
- CUDA backend の利用可否
- main model の存在
- `mmproj` の存在と Vision 設定の整合性
- draft model の存在
- MTP / Draft model 設定と runtime 対応の整合性

**受け入れ条件:**
- 診断エラーは、原因と次の修正行動を表示する。
- Vision 不整合はエラーとして扱う。
- MTP / Draft model 不可は、fallback mode に従ってエラーまたは警告にする。
- Phase 5 では `hoshikage doctor [--model <label>] [--json]` を追加する。
- Phase 5 では `hoshikage add --check` を追加し、登録前に候補 bundle を診断する。ERROR がある場合は登録しない。
- CUDA backend shared library は存在チェックとして表示する。
- managed `llama-server` 起動に必要な port、working directory、environment、起動 command preview を表示できる。

### MR-008: OpenAI 互換 Vision Message 対応

**優先度:** 高

Hoshikage は、OpenAI 互換 Chat Completions の Vision message を受け付ける。

**対応形式:**
- 既存形式: `content: "text"`
- 追加形式: `content: [{ "type": "text", ... }, { "type": "image_url", ... }]`

**入力例:**

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "この画像を説明して" },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "auto"
      }
    }
  ]
}
```

**画像入力:**
- `data:image/png;base64,...`
- `data:image/jpeg;base64,...`
- `file:///absolute/path/image.png`
- ローカル絶対パス

**初期方針:**
- 外部 URL は、ローカル推論サーバーの性質上、初期実装では必須対象にしない。
- 外部 URL を扱う場合は明示的な設定で許可する。

**受け入れ条件:**
- text-only の既存 API は壊さない。
- Vision 非対応モデルに画像入力が渡された場合は明示エラーを返す。
- `mmproj` 未設定または不整合の場合は推論前に明示エラーを返す。
- base64 data URL とローカルファイルパスの両方を受け付ける。
- 画像入力は OpenAI 互換 message 構造を保持したまま managed `llama-server` へ中継する。
- 外部 URL は初期実装では明示エラーとする。LAN 内の別 PC から呼び出す場合は base64 data URL を使う。

### MR-009: Bundle 単位 RAM ディスク管理

**優先度:** 高

Hoshikage は、RAM ディスク利用時に単一モデルファイルだけでなく、Model Bundle に含まれる複数ファイルをまとめて RAM ディスクへ配置できる。

**対象ファイル:**
- main model
- `mmproj`
- draft model

**方針:**
- Linux では `RAMDISK_PATH` 配下へ bundle 単位でコピーする。
- main model と `mmproj` は原則として両方 RAM ディスクへ配置する。
- コピー前に必要容量と空き容量を確認する。
- `great_timeout` 到達時は bundle 単位で削除する。
- RAM ディスク上の Hoshikage 管理領域は `RAMDISK_PATH/hoshikage/current` とする。
- bundle 配置時は Hoshikage 管理領域だけを空にし、同じ `RAMDISK_PATH` 直下のユーザーファイルは削除しない。

**受け入れ条件:**
- bundle 内の複数 `.gguf` が同じ推論セッションで一貫したパスに解決される。
- RAM ディスク容量不足時は明示エラーを返す。
- 別モデルへ切り替える場合、不要になった bundle cache を整理できる。
- `manifest.json` に source / dest / role / size を記録できる。

### MR-010: VRAM 12GB 前提の GPU Offload 方針

**優先度:** 高

Hoshikage は、VRAM 12GB を最低保証の目安として、GPU offload 設定を扱う。

**方針:**
- 既定値は `n_gpu_layers = -1` とし、全 GPU offload を試す。
- モデルごとの `n_gpu_layers` は Model Bundle 側で上書きできる。
- ロード失敗時は、VRAM 不足か backend 不整合かを可能な範囲で区別する。
- 将来的に `hoshikage tune-gpu-layers` で段階的に `n_gpu_layers` を下げて試行し、成功値を bundle に保存できるようにする。

**受け入れ条件:**
- 12GB VRAM 環境を基準にした推奨設定を文書化する。
- `n_gpu_layers` の実効値を status またはログで確認できる。
- ロード失敗時に、設定変更候補を表示する。
- Model Bundle に `n_ctx` / `n_gpu_layers` が指定されている場合、runtime load 時は global config より bundle の値を優先する。
- runtime load 時の実効 `n_ctx` / `n_gpu_layers` はログまたは診断に出せる形で保持する。

### MR-011: 推論性能メトリクス

**優先度:** 高

Hoshikage は、爆速環境を実現できているかを確認するため、推論性能メトリクスを記録・表示できる。

**目標値:**
- 生成速度 100〜140+ tokens/sec を野心的な目標とする。

**測定項目:**
- Time To First Token (TTFT)
- prompt eval tokens/sec
- generation tokens/sec
- total tokens/sec
- MTP / Draft model 有効・無効の比較
- Vision 入力時の初回応答時間

**測定条件:**
- 性能検証は、同一 GPU 上で他の重い推論処理が動いていない状態で行う。
- MTP / Draft model の比較では、prompt、`max_tokens`、sampling 設定、`n_ctx`、`n_gpu_layers` を揃える。
- 音声会話BOTや単独チャット応答を優先する場合、managed `llama-server` は 1 slot 起動を標準とし、不要な並列用 KV cache / 作業領域を確保しない。
- `llama-server` の timings に draft token 数と accepted token 数が含まれる場合、MTP / Draft model が実際に効いている根拠として記録する。
- MTP / Draft model 有効時は、`llama-server` の speculative decoding option に準拠し、Flash Attention と draft 側 GPU offload の指定を runtime command に反映する。
- MTP / Draft model 有効時の初期チューニング値として、低信頼 draft を早めに止める `--spec-draft-p-min 0.10` を使う。より高い値や ngram 併用は環境・prompt により悪化するため、標準値にはしない。
- 12B QAT + MTP + Vision bundle の低レイテンシ運用では、`TOP_P=0.95`、`REPEAT_PENALTY=1.0` を推奨初期値とする。`TOP_P=0.8` や `REPEAT_PENALTY=1.1` は MTP の draft acceptance を下げ、生成速度を落とす場合がある。
- CUDA で全層 GPU offload を狙う Model Bundle では、`n_gpu_layers=99` など明示的に十分大きい値を保存する。`-1` は runtime version により最速条件と一致しない場合があるため、性能検証では明示値を使う。
- RTX 4070 SUPER 12GB の実測では、12B QAT + MTP + Vision bundle、Thinking off、`n_ctx=8192`、`n_gpu_layers=99`、`TOP_P=0.95`、`REPEAT_PENALTY=1.0` により Hoshikage 経由で約 90 tokens/sec を確認した。

**受け入れ条件:**
- ログまたは status API で直近推論のメトリクスを確認できる。
- MTP / Draft model 有効時に、通常推論との差分を検証できる。
- 性能値はモデル、backend、`n_ctx`、`n_gpu_layers` と紐づけて追跡できる。

### MR-012: Fallback / Strict Mode

**優先度:** 高

Hoshikage は、MTP / Draft model が使えない場合の挙動をモデルごとに制御できる。

**fallback mode:**
- `strict`: 指定機能が使えなければエラー
- `warn`: 警告を出して通常推論へ fallback
- `off`: speculation を使わない

**方針:**
- Vision 不整合は fallback せずエラーにする。
- MTP / Draft model 不可は fallback mode に従う。
- fallback mode は feature fallback を指す。たとえば MTP / Draft model が使えない場合に、同じ `llama-server-managed` backend 内で通常推論へ落とすかどうかを制御する。
- `llama-server-managed` が指定されている場合、`llama-server` 未配置・起動失敗・異常終了から `libllama` FFI へ自動的に backend fallback しない。
- runtime backend の default は `llama-server-managed` とする。旧 FFI 経路は `HOSHIKAGE_RUNTIME_BACKEND=llama-ffi` の明示指定時のみ使用する。
- managed `llama-server` の内部 endpoint は default `127.0.0.1:13030` とし、Hoshikage 公開 port とは分離する。
- managed `llama-server` は待機中もモデルを保持するため、VRAM は残る。低遅延応答を優先する標準運用では runtime の idle sleep option は使わない。待機時の演算高止まりが問題になる環境では、明示設定でのみ idle sleep を有効化する。

**受け入れ条件:**
- fallback した場合はログと status に残す。
- `strict` では通常推論へ黙って落とさない。
- ユーザーがモデルごとに fallback mode を選べる。

### MR-013: Status / Models API 拡張

**優先度:** 高

Hoshikage は、現在の runtime 状態とモデル能力を API から確認できる。

**`/v1/status` 拡張候補:**
- loaded model
- backend (`llama-server-managed` / `ffi`)
- managed process pid
- managed process health
- upstream endpoint
- llama.cpp version
- CUDA availability
- current `n_gpu_layers`
- current `n_ctx`
- Vision enabled
- MTP / Draft model enabled
- active request count
- last fallback reason
- last performance metrics

**`/v1/models` 拡張候補:**
- model id
- main model path
- Vision capability
- `mmproj` configured
- MTP configured
- draft model configured
- fallback mode

**受け入れ条件:**
- 既存の OpenAI 互換レスポンスを壊さない範囲で拡張する。
- 詳細情報が OpenAI 互換形式に乗せづらい場合は、Hoshikage 独自の status endpoint に集約する。
- CLI `hoshikage list` でも主要能力を確認できる。
- `/v1/models` は OpenAI 互換の最小形を維持し、詳細情報は `/v1/hoshikage/models` と `/v1/hoshikage/models/:name` に分離する。

### MR-014: Thinking Mode 制御

**優先度:** 高

Hoshikage は、音声会話 BOT など低レイテンシが重要な用途向けに、モデルごとに Thinking mode を無効化できる。

**方針:**
- CLI のユーザー向け option は `--thinking-off` のみ追加する。
- `--thinking-off` が指定された場合、Model Bundle に `thinking.mode = "off"` を保存する。
- `--thinking-off` が指定されない場合は `thinking.mode = "auto"` とみなし、モデル・chat template・runtime の既定動作に任せる。
- `--thinking-on` は追加しない。
- `thinking_budget_tokens` などの詳細値はユーザーに通常指定させず、runtime backend 側の実装詳細として扱う。

**受け入れ条件:**
- `hoshikage add /path/to/model.gguf label --thinking-off` で Thinking 無効モデルとして登録できる。
- `thinking.mode = "off"` の場合、Gemma 系モデルでは Thinking を有効化する control token を挿入しない。
- llama.cpp runtime が reasoning budget を扱える場合、Thinking off に相当する runtime 設定を適用する。
- llama.cpp runtime が Thinking off 相当の runtime option を提供しない場合、Hoshikage は警告を記録して prompt / template policy と safety filter で続行する。
- Thinking off の主制御は、chat template / prompt / runtime backend によって thought block を生成させないこととする。
- Thinking off でも thought block が出力された場合のみ、Hoshikage 側で safety filter として final answer から除去する。
- safety filter で除去された thought block は通常の Chat response body には含めない。
- Thinking off が適用されたか、または runtime 側で適用不能だったかを status / doctor / log で確認できる。
- 既存モデルは、設定を追加しなくても従来通り動作する。

---

## 5. 非機能要件

### NFR-MR-001: 既存 API 互換性

既存の text-only `/v1/chat/completions` は壊さない。

### NFR-MR-002: 明示的なエラー

次の状態は曖昧に失敗させず、原因を区別できるエラーにする。

- llama.cpp が要求機能に未対応
- projector ファイルがない
- draft model がない
- target model と draft model の tokenizer または構成が不整合
- Vision 入力が text-only model に送信された

### NFR-MR-003: 実行確認

ビルド成功だけでなく、最低限の runtime smoke test を行う。

- text-only chat
- Vision model に対する画像付き chat
- MTP または draft model 有効時の起動確認
- 非対応環境での明示的な失敗確認
- Runtime Capability 診断
- RAM ディスク bundle 配置
- 性能メトリクス出力
- Thinking off 時の低レイテンシ確認

### NFR-MR-004: ローカルファイル入力の安全性

Vision 入力でローカルファイルパスを受け付ける場合、パス解決とアクセス範囲を明確化する。

- ローカル絶対パスを受け付ける。
- `file://` URL を受け付ける。
- 存在しないファイル、非画像ファイル、読取不可ファイルは明示エラーにする。
- 外部 URL 取得は初期実装では必須対象外とする。

---

## 6. 確定事項と将来候補

今回の改訂で確定した運用:

- Vision bundle では `mmproj` をモデル登録時に明示指定する。
- `llama-server` の必要 option は `hoshikage doctor` と起動時診断で検査する。
- `hoshikage tune-gpu-layers` は今回範囲外とし、bundle 登録時の `n_gpu_layers` 明示指定を基本にする。

将来候補:

- 環境ごとの自動 tuning。
- 外部 URL 画像入力の明示許可 option。
- 複数同時 request 向けの `llama-server` slot 数設定。

---

## 7. 参考資料

- llama.cpp README: https://github.com/ggml-org/llama.cpp
- llama.cpp speculative decoding: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- llama.cpp multimodal: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- llama.app installer: https://github.com/ggml-org/llama-install.sh
- Google Gemma MTP: https://ai.google.dev/gemma/docs/mtp/overview
- Google Gemma Vision: https://ai.google.dev/gemma/docs/capabilities/vision
- Hugging Face image-text-to-text: https://huggingface.co/docs/transformers/tasks/image_text_to_text
