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

```bash
# Cargo経由でローカルインストール
cargo install --path .
```

これにより、星影バイナリと必要なライブラリがシステムにインストールされます。

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

### 2.2 モデルの切り替え

リクエストの`model`パラメータで動的に切り替えます。

---

## 3. サーバー起動

### 3.1 環境変数の設定

```bash
# CUDAライブラリのパスを設定（システムCUDAライブラリを使用する場合）
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# （オプション）カスタムCUDAライブラリを使用する場合
# export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH
```

### 3.2 起動コマンド

```bash
# 標準起動
hoshikage

# カスタムポートで起動
hoshikage --port 3030
```

### 3.3 デーモンとして実行

```bash
# systemdサービスを作成
sudo nano /etc/systemd/system/hoshikage.service
```

```
[Unit]
Description=星影 - 高速ローカル推論サーバー
After=network.target

[Service]
Type=simple
User=hoshikage
Environment=LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
WorkingDirectory=/home/tane/dev/AI/hoshikage
ExecStart=/home/tane/dev/AI/hoshikage/hoshikage
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# サービスを有効化・起動
sudo systemctl daemon-reload
sudo systemctl enable hoshikage
sudo systemctl start hoshikage

# サービスの状態確認
sudo systemctl status hoshikage
```

**例:**
```bash
mkdir -p /opt/hoshikage/models
cp /path/to/LFM2.5-1.2B-JP-Q8_0.gguf /opt/hoshikage/models/
```

```json
{
  "LFM2.5_Q8": {
    "path": "/opt/hoshikage/models",
    "model": "LFM2.5-1.2B-JP-Q8_0.gguf",
    "stop": ["<|im_end|>", "</s>"]
  }
}
```

### 2.2 モデルの切り替え

リクエストの`model`パラメータで動的に切り替えます。

---

## 3. サーバー起動

### 3.1 環境変数の設定

```bash
# CUDAライブラリのパスを設定
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# （オプション）モデルの配置場所を設定
export MODEL_MAP_FILE=/opt/hoshikage/model_map.json
```

### 3.2 起動コマンド

```bash
# デバッグモードで起動
./target/debug/hoshikage

# リリースモードで起動
./target/release/hoshikage

# カスタムポートで起動
./target/release/hoshikage --port 3030
```

### 3.3 デーモンとして実行

```bash
# systemdサービスを作成
sudo nano /etc/systemd/system/hoshikage.service
```

```
[Unit]
Description=星影 - 高速ローカル推論サーバー
After=network.target

[Service]
Type=simple
User=hoshikage
Environment=LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
WorkingDirectory=/opt/hoshikage
ExecStart=/opt/hoshikage/hoshikage
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# サービスを有効化・起動
sudo systemctl daemon-reload
sudo systemctl enable hoshikage
sudo systemctl start hoshikage

# サービスの状態確認
sudo systemctl status hoshikage
```

---

## 4. APIの使用

### 4.1 curlでテスト

#### 4.1.1 チャット補完（非ストリーミング）

```bash
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5_Q8",
    "messages": [
      {"role": "user", "content": "こんにちは、よろしくお願いします。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'
```

#### 4.1.2 チャット補完（ストリーミング）

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

### 4.2 Pythonで使用

#### 4.2.1 OpenAI SDK

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
    temperature=0.7
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

## 5. モデル一覧の確認

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

## 6. ライブラリのトラブルシューティング

### 6.1 CUDAライブラリが見つからない

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
ls ~/.config/hoshikage/lib/libcuda.so 2>/dev/null || echo "カスタムライブラリはありません"
```

---

**作成日:** 2026-01-18

### 6.2 CUDAライブラリが見つからない

**エラー:** `libllama.so: cannot open shared object file`

**解決策:**
```bash
# CUDAライブラリのパスを確認
echo $LD_LIBRARY_PATH

# パスを設定
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# ライブラリの存在を確認
ls /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so
```

### 6.3 ポートが競合している

**エラー:** `address already in use`

**解決策:**
```bash
# 使用中のポートを確認
sudo netstat -tulpn | grep :3030

# 別のポートで起動
./target/release/hoshikage --port 3031
```

---

**作成日:** 2026-01-18
**バージョン:** 1.0.0
