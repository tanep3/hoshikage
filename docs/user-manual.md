# ユーザーマニュアル：星影 - Rust版高速ローカル推論サーバー

**バージョン:** 1.0.0  
**作成日:** 2026-01-18  
**言語:** Rust

---

## 1. インストール

### 1.1 システム要件

| 項目 | 最小要件 | 推奨要件 |
|--------|---------|---------|
| CPU | 8コア以上 | 16コア以上 |
| メモリ | 16GB以上 | 32GB以上 |
| GPU | VRAM 8GB以上 | VRAM 12GB以上 |
| ストレージ | SSD 50GB以上 | NVMe SSD 100GB以上 |

### 1.2 ソフトウェア要件

- **OS**: Linux（Ubuntu 20.04以降推奨）
- **CUDAドライバ**: 470+ (GPU使用時)
- **Rust**: 1.70以上（Cargo経由でインストールされます）

### 1.3 依存関係のインストール

**Linuxの場合:**
```bash
# 1. 実行パスの設定（.bashrc推奨）
echo 'export LD_LIBRARY_PATH=$HOME/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 2. Cargo経由でローカルインストール
cargo install --path .
```

**Windowsの場合 (PowerShell):**
```powershell
# 1. ライブラリ配置用ディレクトリ作成
mkdir -p "$env:APPDATA\hoshikage\lib"

# 2. 環境変数(PATH)設定 (永続的)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\hoshikage\lib", [System.EnvironmentVariableTarget]::User)

# 3. Cargo経由でローカルインストール
cargo install --path .
```

これにより、`hoshikage` コマンドがターミナルから直接利用可能になります。
※ 別途 `libllama.so` (Linux) または `llama.dll` (Windows) を上記設定パスに配置する必要があります。
現行の配置方針は `docs/LIBRARY_GUIDE.md`、最新 llama.cpp の導入手順は `docs/llama-cpp-install-guide.md` を参照してください。

---

## 2. モデル管理

### 2.1 モデルの登録

GGUFモデルを `models/` ディレクトリに配置し、`~/.config/hoshikage/model_map.json` に登録します。

**model_map.jsonのフォーマット:**

```json
{
  "model-alias": {
    "base_path": "/path/to/models",
    "model": "model-file.gguf",
    "stop": ["<|im_end|>", "</s>"]
  }
}
```

既存の `path` field も読み込み互換のため使用できます。新しく保存する設定では `base_path` を使用します。

`stop` はデフォルトのストップシーケンスにマージされ、重複は除去されます。
デフォルトには `<|im_start|>`, `<|im_end|>`, `</s>`, `<|eot_id|>`, `<|endoftext|>` が含まれます。

**例:**
```bash
mkdir -p models
cp /path/to/LFM2.5-1.2B-JP-Q8_0.gguf models/

# model_map.json を作成
cat > ~/.config/hoshikage/model_map.json << 'EOF'
{
  "LFM2.5_Q8": {
    "base_path": "./models",
    "model": "LFM2.5-1.2B-JP-Q8_0.gguf",
    "stop": ["<|im_end|>", "<|eot_id|>", "</s>"]
  }
}
EOF
```

Vision や speculative decoding 用の追加ファイルがあるモデルは、同じ `model_map.json` に bundle 設定として保存できます。

```json
{
  "gemma4-local": {
    "base_path": "/models/gemma4-local",
    "model": "main.gguf",
    "mmproj": "mmproj.gguf",
    "drafter": "mtp.gguf",
    "speculation": {
      "mode": "mtp",
      "fallback": "warn"
    },
    "thinking": {
      "mode": "off"
    },
    "n_ctx": 8192,
    "n_gpu_layers": 99
  }
}
```

### 2.2 モデルの管理 (CLI)

`hoshikage` コマンドでモデルを簡単に管理できます。サーバー起動中でも、停止中でも、いつでも実行可能です。

#### モデルの追加
```bash
# 基本 (パスとラベルのみ)
hoshikage add /path/to/LFM.gguf LFM-v2

# ストップワードを指定する場合
hoshikage add /path/to/LFM.gguf LFM-v2 "</s>" "<|im_end|>"

# Vision 用 projector を登録する場合
hoshikage add /models/gemma4/main.gguf gemma4-local --mmproj /models/gemma4/mmproj.gguf

# MTP 用 drafter と Thinking off を登録する場合
hoshikage add /models/gemma4/main.gguf gemma4-local --mtp-drafter /models/gemma4/mtp.gguf --thinking-off

# モデル別 context / GPU offload を登録する場合
# CUDA で全層GPU offloadを狙う場合は 99 など十分大きい値を指定します。
hoshikage add /models/gemma4/main.gguf gemma4-local --n-ctx 8192 --n-gpu-layers 99

# 登録前に bundle と runtime の整合性を確認する場合
hoshikage add /models/gemma4/main.gguf gemma4-local --mmproj /models/gemma4/mmproj.gguf --check
```

`--n-ctx` と `--n-gpu-layers` はモデル単位の設定として保存され、該当モデルのロード時にサーバー全体の既定値より優先されます。

#### モデルの削除
```bash
hoshikage rm LFM-v2
```

#### モデルの一覧表示
```bash
hoshikage list

# 詳細表示
hoshikage list --details
```

#### runtime / bundle 診断
```bash
# runtime library と backend の診断
hoshikage doctor

# 登録済みモデルの bundle 整合性も診断
hoshikage doctor --model gemma4-local

# JSON で出力
hoshikage doctor --model gemma4-local --json
```

### 2.3 モデルの切り替え
リクエストの`model`パラメータで登録したモデルラベル（例: `LFM-v2`）を指定することで、動的に使用するモデルを切り替えられます。

### 2.4 次期 Model Bundle 方針

現行版では、モデルごとの設定は `~/.config/hoshikage/model_map.json` に保存されます。
`.env` はサーバー全体の既定値を扱い、モデルごとの差分は `model_map.json` 側で管理する方針です。

次期モデルランタイム改訂では、次のようなファイルと設定を Model Bundle としてモデル単位に管理する予定です。

- メイン GGUF モデル
- Vision projector (`mmproj`)
- Draft model
- stop sequence
- context length
- GPU offload 設定
- MTP / Draft model の fallback mode

詳細は `docs/model-runtime-revision-requirements.md` を参照してください。

---

## 3. 設定
(高度な設定)

### 3.1 環境変数の設定 (.env)
サーバーの動作を環境変数ファイル (`.env`) でカスタマイズできます。
`~/.config/hoshikage/.env` に配置すると自動的に読み込まれます。

**設定ファイルの例 (.env.example):**
```bash
# サーバーポート
PORT=3030

# ログファイル出力パス (ファイルパスとして扱う)
# 例: ~/.config/hoshikage/logs/hoshikage.log
# 出力は日次ローテーションされ、LOG_FILE_PATH.YYYY-MM-DD になります。
# LOG_FILE_PATH=~/.config/hoshikage/logs/hoshikage.log

# 非アクティブ時の自動アンロードまでの時間 (秒)
# 0 にすると自動アンロード無効（デフォルト: 300）
IDLE_TIMEOUT=300

# RAMディスク設定 (高速ロード用)
# Linuxの場合: /dev/shm などを指定できます。sudo権限も不要です。
# Model Bundle は RAMDISK_PATH/hoshikage/current に配置されます。
# Windows / Mac はRAMディスク非対応のため、自動的にSSDからの直接ロードになります。
RAMDISK_PATH=/dev/shm

# 長時間非アクティブ時のRAMディスク解放 (分)
# メモリを完全にOSに返すまでの時間（デフォルト: 60分）
GREAT_TIMEOUT=60

# コンテキスト長 (トークン数)
# デフォルト: 4096
N_CTX=4096

# 生成パラメータ
# デフォルト: TEMPERATURE=0.2, TOP_P=0.95
TEMPERATURE=0.2
TOP_P=0.95

# MTP 有効時の速度を優先する場合は 1.0 を推奨します。
REPEAT_PENALTY=1.0

# llama-server の独自 idle sleep は標準では無効にします。
# VRAM 滞在は IDLE_TIMEOUT、RAM ディスク滞在は GREAT_TIMEOUT で管理します。
HOSHIKAGE_LLAMA_SERVER_SLEEP_IDLE_SECS=off

```

詳細なパラメータは、プロジェクトに含まれる `.env.example` を参照してください。

`LOG_FILE_PATH` を指定しない場合、ログは標準出力/標準エラーに出力されます。

---

## 4. サーバー起動

### 3.1 準備
ライブラリが正しく設定されていれば、特別な環境変数は不要です。
もし一時的にパスを通したい場合は以下のようにします。

```bash
export LD_LIBRARY_PATH=~/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH
```

### 4.2 起動コマンド

```bash
# 標準起動
hoshikage

# カスタムポートで起動
hoshikage --port 3030
```

### 4.3 デーモンとして実行 (ユーザーモード)

`systemd` のユーザーユニット機能を使って、管理者権限なしで常駐させることができます。

```bash
# 1. ユニットファイル配置用ディレクトリ作成
mkdir -p ~/.config/systemd/user

# 2. ユニットファイル作成
nano ~/.config/systemd/user/hoshikage.service
```

**hoshikage.service の内容:**
```ini
[Unit]
Description=星影 (Hoshikage) - AI Inference Server
After=network.target

[Service]
Type=simple
# 環境変数を指定（絶対パスで記述）
Environment=LD_LIBRARY_PATH=%h/.config/hoshikage/llama.cpp
WorkingDirectory=%h/dev/AI/hoshikage
ExecStart=%h/dev/AI/hoshikage/target/release/hoshikage
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```
※ `%h` はホームディレクトリ（`/home/ユーザー名`）に自動置換されます。

```bash
# 3. サービスの有効化と起動
systemctl --user daemon-reload
systemctl --user enable hoshikage
systemctl --user start hoshikage

# 4. ステータス確認
systemctl --user status hoshikage

# (任意) ログアウト後も実行し続ける場合
loginctl enable-linger $USER
```

## 5. APIの使用

### 5.1 curlでテスト

#### 5.1.1 チャット補完（非ストリーミング）

```bash
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5_Q8",
    "messages": [
      {"role": "user", "content": "こんにちは、よろしくお願いします。"}
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "stream": false
  }'
```

#### 5.1.2 チャット補完（ストリーミング）

```bash
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5_Q8",
    "messages": [
      {"role": "user", "content": "猫について説明してください。"}
    ],
    "stream": true
  }'
```

#### 5.1.3 画像入力形式

Chat Completions API の `messages[].content` は、従来の文字列に加えて OpenAI 互換の parts 配列も受け付けます。

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "この画像を説明してください。" },
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

対応する画像指定は `data:image/png;base64,...`、`data:image/jpeg;base64,...`、`file:///absolute/path/image.png`、ローカル絶対パスです。外部 URL は初期実装では受け付けません。

画像入力には、モデル登録時に `--mmproj` を指定した Model Bundle が必要です。Vision 非対応モデルへ画像入力を送った場合は明示エラーになります。

### 5.2 Pythonで使用

#### 5.2.1 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3030/v1",
    api_key="dummy"
)

# 非ストリーミング
response = client.chat.completions.create(
    model="LFM2.5_Q8",
    messages=[
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
        {"role": "user", "content": "こんにちは"}
    ],
    temperature=0.2
)

print(response.choices[0].message.content)

# ストリーミング
stream = client.chat.completions.create(
    model="LFM2.5_Q8",
    messages=[
        {"role": "user", "content": "猫について説明してください。"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 6. モデル一覧の確認

```bash
curl http://localhost:3030/v1/models
```

**レスポンス例:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "LFM2.5_Q8",
      "object": "model",
      "created": 1686935002,
      "owned_by": "tane"
    }
  ]
}
```

---

## 7. ライブラリのトラブルシューティング

### 7.1 CUDAライブラリが見つからない

**エラー:** `libllama.so: cannot open shared object file`

**解決策:**

システムCUDAライブラリを使用する場合、環境変数を設定してください。

```bash
# システムCUDAライブラリのパスを確認
echo $LD_LIBRARY_PATH

# システムCUDAライブラリを使用する場合
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Hoshikage runtime directory を明示する場合
export LD_LIBRARY_PATH=~/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH

# ライブラリの存在を確認
ls /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so
ls /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so
ls /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so

# Hoshikage runtime の存在を確認
ls ~/.config/hoshikage/llama.cpp/llama-server 2>/dev/null || echo "llama-server が見つかりません"
ls ~/.config/hoshikage/llama.cpp/libllama.so 2>/dev/null || echo "libllama が見つかりません"
```

**Windowsの場合:**
`%APPDATA%\hoshikage\llama.cpp` に `llama-server.exe` と `llama.dll` があるか確認してください。


### 7.2 ポートが競合している

**エラー:** `address already in use`

**解決策:**
```bash
# 使用中のポートを確認
sudo netstat -tulpn | grep :3030

# 別のポートで起動
./target/release/hoshikage --port 3031
```
