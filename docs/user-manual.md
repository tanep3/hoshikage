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
echo 'export LD_LIBRARY_PATH=$HOME/.config/hoshikage/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
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
※ 別途 `libllama.so` (Linux) または `llama.dll` (Windows) を上記設定パスに配置する必要があります（詳細は `docs/LIBRARY_GUIDE.md` 参照）。

---

## 2. モデル管理

### 2.1 モデルの登録

GGUFモデルを `models/` ディレクトリに配置し、`~/.config/hoshikage/model_map.json` に登録します。

**model_map.jsonのフォーマット:**

```json
{
  "model-alias": {
    "path": "/path/to/models",
    "model": "model-file.gguf",
    "stop": ["<|im_end|>", "</s>"]
  }
}
```

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
    "path": "./models",
    "model": "LFM2.5-1.2B-JP-Q8_0.gguf",
    "stop": ["<|im_end|>", "<|eot_id|>", "</s>"]
  }
}
EOF
```

### 2.2 モデルの管理 (CLI)

`hoshikage` コマンドでモデルを簡単に管理できます。サーバー起動中でも、停止中でも、いつでも実行可能です。

#### モデルの追加
```bash
# 基本 (パスとラベルのみ)
hoshikage add /path/to/LFM.gguf LFM-v2

# ストップワードを指定する場合
hoshikage add /path/to/LFM.gguf LFM-v2 "</s>" "<|im_end|>"
```

#### モデルの削除
```bash
hoshikage rm LFM-v2
```

#### モデルの一覧表示
```bash
hoshikage list
```

### 2.3 モデルの切り替え
リクエストの`model`パラメータで登録したモデルラベル（例: `LFM-v2`）を指定することで、動的に使用するモデルを切り替えられます。

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
# Linuxの場合: /dev/shm (デフォルト) を使用するため設定不要です。sudo権限も不要です。
# Windows / Mac はRAMディスク非対応のため、自動的にSSDからの直接ロードになります。
RAMDISK_PATH=/dev/shm

# 長時間非アクティブ時のRAMディスク解放 (分)
# メモリを完全にOSに返すまでの時間（デフォルト: 60分）
GREAT_TIMEOUT=60

# コンテキスト長 (トークン数)
# デフォルト: 4096
N_CTX=4096

# 生成パラメータ
# デフォルト: TEMPERATURE=0.2, TOP_P=0.8
TEMPERATURE=0.2
TOP_P=0.8

```

詳細なパラメータは、プロジェクトに含まれる `.env.example` を参照してください。

`LOG_FILE_PATH` を指定しない場合、ログは標準出力/標準エラーに出力されます。

---

## 4. サーバー起動

### 3.1 準備
ライブラリが正しく設定されていれば、特別な環境変数は不要です。
もし一時的にパスを通したい場合は以下のようにします。

```bash
export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH
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
Environment=LD_LIBRARY_PATH=%h/.config/hoshikage/lib
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

# カスタムCUDAライブラリを使用する場合
export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH

# ライブラリの存在を確認
ls /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so
ls /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so
ls /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so

# カスタムライブラリの存在を確認
ls ~/.config/hoshikage/lib/libllama.so 2>/dev/null || echo "カスタムライブラリはありません"
```

**Windowsの場合:**
`%APPDATA%\hoshikage\lib` に `llama.dll` があるか確認してください。


### 7.2 ポートが競合している

**エラー:** `address already in use`

**解決策:**
```bash
# 使用中のポートを確認
sudo netstat -tulpn | grep :3030

# 別のポートで起動
./target/release/hoshikage --port 3031
```
