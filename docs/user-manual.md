# Hoshikage ユーザーマニュアル

[English](user-manual.en.md) | 日本語

HoshikageはGGUFモデルを管理し、Chat Completions APIとResponses APIを提供するローカル推論サーバーです。Codex CLIなどの上位エージェントがツール実行と反復を担当し、Hoshikageはモデル実行とプロトコル変換を担当します。

## 1. インストール

### 1.1 必要環境

- Rust stable toolchain
- Hoshikage用llama.cpp runtime bundle
- 利用するGGUFモデル
- GPU利用時は対応するCUDA環境

```bash
cargo install --path .
hoshikage --version
```

runtime bundleは標準でOS別Hoshikage設定directoryの`llama.cpp`へ配置します。別の場所を使う場合は`HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR`を設定します。

### 1.2 設定ファイル

Hoshikage serverの標準設定directoryはOSごとに異なります。

| OS | Hoshikage設定directory |
|---|---|
| Linux | `~/.config/hoshikage` |
| macOS | `~/Library/Application Support/hoshikage` |
| Windows | `%APPDATA%\hoshikage` |

Hoshikageはこのdirectoryの`.env`を読み込みます。別ファイルを使う場合は`HOSHIKAGE_CONFIG_PATH`を設定します。この場所はHoshikage server自身の設定・Token管理用であり、Codexや上位アプリケーションの設定場所ではありません。

```dotenv
HOST=127.0.0.1
PORT=3030
N_CTX=16384
HOSHIKAGE_LANG=ja
```

`HOST=127.0.0.1`または`localhost`は認証なしのloopback利用です。LANアドレスまたは`0.0.0.0`へbindするとBearer Token認証が必須になります。

## 2. モデル管理

### 2.1 モデル登録

```bash
hoshikage add /models/gemma4/model.gguf unsloth-gemma4-12b-qat-thinking-off --n-ctx 16384 --thinking-off
hoshikage list --details
```

Codex利用の実用下限は16K context、推奨は32Kです。モデルごとの`n_ctx`が未指定の場合は`.env`の`N_CTX`を使います。

Tool CallingはBundleの`tool_calling`設定を正とします。未設定Bundleは安全側で`disabled`です。`doctor`は候補や矛盾を診断しますが、自動書換えしません。

### 2.2 Bundle診断

```bash
hoshikage doctor --model unsloth-gemma4-12b-qat-thinking-off
hoshikage doctor --model unsloth-gemma4-12b-qat-thinking-off --json
```

`--json`のfield名、status、ID、`message_key`は言語設定で変化しません。自動処理では表示文ではなくこれらを使用してください。

## 3. Loopback最短手順

### 3.1 サーバー起動

`.env`で`HOST=127.0.0.1`を指定し、Hoshikageを起動します。

```bash
hoshikage
curl http://127.0.0.1:3030/health
curl http://127.0.0.1:3030/ready
```

### 3.2 Responses API確認

```bash
curl http://127.0.0.1:3030/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth-gemma4-12b-qat-thinking-off","input":"Return exactly OK."}'
```

loopbackではTokenは不要です。`/health`はプロセスの生存確認、`/ready`は設定とruntimeの受付準備、`/v1/status`はモデルのロード状態確認に使います。

## 4. LANとToken

### 4.1 Tokenとは何か

Tokenは、LAN上のHoshikageを利用してよい端末であることを確認するための秘密の合言葉です。OpenAIのAPIキー、ChatGPTのログイン情報、モデルのライセンスキーではありません。同じPC内の`127.0.0.1`接続では不要ですが、別のPCからLAN経由で接続する場合は必須です。

Token本文を知っている端末はHoshikageへ推論を依頼できます。パスワードと同じように扱い、Git、Model Bundle、Issue、ログ、チャットへ記載しないでください。CodexのTOMLへToken本文を直接書かず、起動元の上位アプリケーションがprocess環境変数`HOSHIKAGE_API_KEY`として渡します。

### 4.2 Hoshikage側でTokenを作る

Hoshikage serverを動かすマシンへ管理者としてloginし、接続元ごとに用途名付きTokenを作成します。`hoshikage token`はremote APIではなく、そのマシンのToken storeを直接管理するCLIです。

```bash
hoshikage token create codex-desktop
hoshikage token list
```

`token list`はname、Token本文、public ID、作成日時、更新日時を表示します。server machineの管理者用ツールなので全情報を表示します。画面共有中、端末出力の収集中、第三者が見られる場所では実行しないでください。`codex-desktop`は管理用の用途名であり、Token本文ではありません。

```text
codex-desktop	hsk_xxx_xxx	public_id=xxx	created=1780000000	updated=1780000000
```

Token storeはHoshikage serverの標準設定directoryにある`auth_tokens.json`です。Linux・macOSではowner限定`0600`、WindowsではownerとSYSTEMだけにfull controlを許可するprotected ACLをHoshikageが設定・検証します。上位アプリケーションやCodexはこのファイルを直接読みません。

### 4.3 HoshikageをLANで待ち受ける

1.2に記載したHoshikage server側の`.env`を次のように設定し、Hoshikageを再起動します。

```dotenv
HOST=0.0.0.0
PORT=3030
```

本書ではHoshikage serverのLANアドレスを`192.168.1.50`として説明します。Linux・macOSのclientでは次を実行します。

```bash
curl http://192.168.1.50:3030/health
```

WindowsのPowerShellでは次を実行し、`TcpTestSucceeded : True`とhealth情報が返ることを確認します。

```powershell
Test-NetConnection 192.168.1.50 -Port 3030
Invoke-RestMethod http://192.168.1.50:3030/health
```

失敗する場合は、IPアドレス、Hoshikageの起動状態、OSのファイアウォールを確認します。LAN公開は信頼できる家庭内・組織内ネットワークに限定し、ルーターでWANからのport forwardを行わないでください。

### 4.4 上位アプリケーションからCodexへ渡す

Tokenの標準的な受け渡し責務は、YatagarasuなどCodexを起動する上位アプリケーションにあります。上位アプリケーションは選択したTokenを子processの`HOSHIKAGE_API_KEY`へ設定してCodexを起動し、OS全体へ永続登録しません。

Linux・macOSで手動確認する場合は、Tokenをshell historyへ残さないよう非表示入力できます。

```bash
printf "HOSHIKAGE_API_KEY: "
IFS= read -rs HOSHIKAGE_API_KEY
printf "\n"
export HOSHIKAGE_API_KEY
codex exec --profile hoshikage "Return exactly the word OK."
unset HOSHIKAGE_API_KEY
```

Windows PowerShellの現在のprocessだけへ設定して確認する例です。入力内容は画面に表示されるため、周囲と画面共有に注意してください。

```powershell
$env:HOSHIKAGE_API_KEY = Read-Host "HOSHIKAGE_API_KEY"
codex exec --profile hoshikage "Return exactly the word OK."
Remove-Item Env:HOSHIKAGE_API_KEY
```

### 4.5 Windows版Codexアプリへ渡す

Windows版Codexアプリを直接起動し、Tokenを注入する上位アプリケーションがない場合は、Windowsの利用者環境変数へ登録します。

1. Windowsのスタートメニューで「環境変数」を検索します。
2. 「環境変数を編集」または「アカウントの環境変数を編集」を開きます。
3. 「ユーザー環境変数」の「新規」を選びます。
4. 変数名へ`HOSHIKAGE_API_KEY`、変数値へToken本文を入力します。
5. OKですべて閉じます。
6. 起動中のCodexアプリを完全に終了し、もう一度起動します。

PowerShellで永続登録する場合は次でも同じです。ただしTokenがPowerShellの履歴に残る可能性があるため、通常は上記の画面操作を推奨します。

```powershell
[Environment]::SetEnvironmentVariable("HOSHIKAGE_API_KEY", "<token>", "User")
```

Token本文を表示せず、登録の有無だけを確認できます。

```powershell
if ([Environment]::GetEnvironmentVariable("HOSHIKAGE_API_KEY", "User")) { "HOSHIKAGE_API_KEY is set" }
```

環境変数は、それを読み込んだ時点のアプリに保持されます。登録や更新の後にCodexアプリの再起動が必要なのはこのためです。

### 4.6 Rotate・Revoke

```bash
hoshikage token rotate codex-desktop
hoshikage token revoke codex-desktop
```

rotateすると旧Tokenは直ちに無効になります。`token list`で新Tokenを確認し、上位アプリケーションがCodexへ渡す値を更新してください。Windows利用者環境変数を使う場合は値を更新し、Codexアプリを完全に再起動します。端末を廃止した場合や漏えいが疑われる場合はrevokeします。

旧digest-only形式のTokenは認証には引き続き使えますが、平文を復元できません。listに`<unavailable: rotate required>`と表示された用途名はrotateして新形式へ移行してください。

### 4.7 401診断

1. `hoshikage token list`で用途名が存在するか確認します。
2. Codex processの`HOSHIKAGE_API_KEY`がlistに表示されたTokenと一致するか確認します。
3. 上位アプリケーションまたはCodexをToken更新後に再起動したか確認します。
4. Codex設定の`env_key`が`HOSHIKAGE_API_KEY`か確認します。
5. Token本文をログ、Issue、チャットへ貼らないでください。

## 5. Codex接続

### 5.1 どこへ設定するか

Codexの利用者設定はHoshikage server設定とは別です。

| 利用環境 | Codex利用者設定 |
|---|---|
| Linux CLI | `~/.codex/config.toml` |
| macOS CLI | `~/.codex/config.toml` |
| Windows CLI・Codexアプリ | `%USERPROFILE%\.codex\config.toml` |

作業ディレクトリの`.codex/config.toml`へProvider設定を置いてはいけません。Codexは安全上の理由から、プロジェクト設定内の`model_provider`と`model_providers`を無視します。`AGENTS.md`も作業指示を書くファイルであり、接続先やモデルを設定するファイルではありません。

Windows版CodexアプリとWSL版Codex CLIの設定場所は別です。WindowsアプリはWindows側、WSL CLIは通常WSL側の`~/.codex`を読みます。

### 5.2 Provider設定を生成する

Hoshikage server machineで、実際のIPアドレスを指定して設定を生成します。

```bash
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --base-url http://192.168.1.50:3030/v1 \
  --authenticated
```

このコマンドは設定を画面に表示するだけで、Codex側のファイルを変更しません。表示結果は次の形式です。

```toml
model = "unsloth-gemma4-12b-qat-thinking-off"
model_provider = "hoshikage"
approval_policy = "on-request"
sandbox_mode = "workspace-write"
model_context_window = 16384
model_auto_compact_token_limit = 12288
tool_output_token_limit = 4096
model_reasoning_summary = "none"

[model_providers.hoshikage]
name = "Hoshikage"
base_url = "http://192.168.1.50:3030/v1"
wire_api = "responses"
env_key = "HOSHIKAGE_API_KEY"
request_max_retries = 1
stream_max_retries = 1
```

Token本文はこのTOMLへ書きません。`env_key`は「Codex processのどの環境変数からTokenを読むか」という指定です。

### 5.3 OSごとに保存する

Linux・macOSでは次の場所へ生成結果を保存します。

```bash
mkdir -p ~/.codex
nano ~/.codex/config.toml
```

WindowsではPowerShellから次の場所を開きます。

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex"
notepad "$env:USERPROFILE\.codex\config.toml"
```

生成されたTOMLを保存します。既存の`config.toml`がある場合は先にバックアップし、他の必要な設定を消さずに内容を統合してください。

4.4または4.5の方法でTokenをCodex processへ渡して起動し、新しい作業で「Return exactly the word OK.」と依頼します。`OK`が返り、Hoshikage serverのログにrequestが記録されれば接続完了です。

### 5.4 同じマシンから接続する場合

HoshikageとCodexが同じマシン上で動き、`127.0.0.1`で接続できる場合はTokenが不要です。次の生成結果を5.1のOS別Codex利用者設定へ保存します。

```bash
hoshikage codex-config --model unsloth-gemma4-12b-qat-thinking-off
```

WSL内のHoshikageへWindowsアプリから接続する構成では、`127.0.0.1`で到達できるかを先に確認してください。到達できない場合はLAN接続と同じ認証付き構成を使用します。

### 5.5 CLIで用途別に切り替える

Codex CLIだけでHoshikageを選んで起動したい場合は、名前付きProfileを使えます。Profile名を`hoshikage`にする場合、Linux・macOS・WSLは`~/.codex/hoshikage.config.toml`、Windowsは`%USERPROFILE%\.codex\hoshikage.config.toml`へ生成結果を保存します。

Linux・macOS・WSLで同一マシンのHoshikageを使う例です。

```bash
mkdir -p ~/.codex
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  > ~/.codex/hoshikage.config.toml
codex exec --profile hoshikage "Return exactly the word OK."
```

Windowsでは`%USERPROFILE%\.codex\hoshikage.config.toml`へ同じTOMLを保存し、PowerShellで次を実行します。

```powershell
codex exec --profile hoshikage "Return exactly the word OK."
```

Codex 0.134以降では、旧形式の`[profiles.hoshikage]`を使いません。Profileごとの独立した`hoshikage.config.toml`を使います。Windows版Codexアプリの通常利用は5.3の利用者設定を推奨します。

### 5.6 対話用と無人用

標準は対話用で`approval_policy = "on-request"`です。無人実行が必要な専用環境だけで次を使います。

```bash
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --mode unattended
```

無人用は`approval_policy = "never"`を生成します。対話用設定を無人用へ流用せず、用途と実行環境を分離してください。Hoshikageは承認やsandboxを制御せず、Codex側が制御します。

### 5.7 設定要素の違い

- **Provider**: Hoshikage APIのURL、Responses wire、認証環境変数を定義します。
- **Profile**: 使用モデル、Provider、承認方針、sandboxなどCodexの実行条件を定義します。
- **Hoshikage server設定**: serverのbind、モデル、Token storeなどを定義します。Codex設定ではありません。
- **process環境変数**: 上位アプリケーションがTokenを、設定ファイルへ直接書かずCodex child processへ渡します。
- **モデルカタログ**: 上位アプリケーションが選択可能なBundleと能力を取得する機械可読一覧です。
- **`AGENTS.md`**: 作業方針やリポジトリ固有指示です。接続先やモデルを選ぶ設定ではありません。

モデル、Provider、Tokenの選択責務はCodexを起動するYatagarasuなどのアプリケーション層にあります。Hoshikageが上位アプリケーションの設定を変更することはありません。

### 5.8 モデルカタログと接続診断

```bash
hoshikage codex-model-catalog --json
hoshikage doctor \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --codex-base-url http://127.0.0.1:3030/v1
```

カタログはすべてのBundleを列挙し、`codex_compatible`、context、Responses、streaming、toolsなどを返します。モデルファイルのpathやTokenは出力しません。

Tool Callingが`disabled`ならCodexはテキスト応答だけ利用できます。Agent Loopでファイルやshell toolを使うには、Bundleに適切な`native`または`json` modeを設定し、`doctor`で確認してください。

## 6. APIと状態確認

### 6.1 エンドポイント

- `GET /health`: 認証不要のliveness
- `GET /ready`: 設定とruntimeのreadiness
- `GET /v1/models`: OpenAI互換モデル一覧
- `GET /v1/hoshikage/models`: Hoshikage能力付きモデル一覧
- `GET /v1/status`: モデルロード状態
- `POST /v1/chat/completions`: 既存Chat Completions
- `POST /v1/responses`: Codex向けResponses API

### 6.2 エラーとrequest ID

Responses APIはOpenAI互換エラーを返し、`x-request-id` headerを付与します。問い合わせやログ照合では本文やTokenではなくrequest IDを使用してください。

## 7. ログとdebug capture

通常ログにはprompt、Tool引数、Tool結果、Tokenを記録しません。必要な診断情報はrequest ID、model、時間、token数、terminal statusなどの安全なsummaryです。

本文が必要な短時間の障害調査だけで次を明示設定できます。

```dotenv
HOSHIKAGE_DEBUG_CAPTURE=on
```

captureはOS別Hoshikage設定directoryの`debug-capture`へrequest単位で保存され、Authorization、Token名のfield、metadataを除外します。既定保持は24時間、directory上限は100 MiB、Unixではdirectory `0700`、file `0600`です。起動時に警告されます。調査後は必ず`off`へ戻し、保存ファイルを機密情報として扱ってください。

## 8. TLSとネットワーク

Hoshikage自身のLAN接続はHTTPです。盗聴可能なネットワークや複数セグメントを越える場合は、Caddy、nginxなどのreverse proxyでTLSを終端し、Hoshikageはloopbackまたは保護された内部interfaceだけで待受けます。

```text
Codex -> HTTPS reverse proxy -> HTTP Hoshikage
```

reverse proxyでも`Authorization` headerをHoshikageへ転送し、request bodyをaccess logへ記録しないでください。WANへ直接公開しないでください。

## 9. トラブルシューティング

### 9.1 Codexが接続できない

- `curl http://HOST:3030/health`でprocessを確認します。
- `/ready`でruntime準備を確認します。
- LANでは401診断手順を実施します。
- Codex Providerの`base_url`が`/v1`まで含むか確認します。
- `doctor --codex-base-url`で接続とモデル能力をまとめて確認します。

### 9.2 Toolを使わない

- `hoshikage codex-model-catalog --json`で`tools`を確認します。
- Bundleの`tool_calling.mode`が`disabled`でないか確認します。
- parserがモデルのchat templateと一致するか確認します。
- contextが16K以上か確認します。
- HoshikageはToolを実行しません。Codex側のTool Registry、approval、sandboxも確認します。

### 9.3 言語を切り替える

```bash
hoshikage --language ja doctor
hoshikage --language en doctor
```

優先順は`--language`、`HOSHIKAGE_LANG`、OS locale、英語fallbackです。エラーcode、JSON field、診断IDは言語に依存しません。
