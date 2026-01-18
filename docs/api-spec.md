# API仕様書：星影 - OpenAI互換API

**バージョン:** 1.0.0  
**作成日:** 2026-01-18  
**言語:** Rust

---

## 1. 概要

星影はOpenAI APIと完全互換のチャット補完APIを提供します。既存のOpenAIクライアントライブラリを変更なしで使用可能です。

### 1.1 基本的な仕様

- **プロトコル**: HTTP/1.1
- **認証**: 現在未実装（将来的に実装予定）
- **フォーマット**: JSON
- **エンコーディング**: UTF-8

---

## 2. エンドポイント

### 2.1 チャット補完

#### 2.1.1 チャット補完（ストリーミング/非ストリーミング）

**エンドポイント**: `POST /v1/chat/completions`

**リクエストボディ:**

```json
{
  "model": "string",
  "messages": [
    {
      "role": "system|user|assistant",
      "content": "string"
    }
  ],
  "temperature": 0.2,
  "top_p": 0.8,
  "max_tokens": 256,
  "stream": false,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

**パラメータ説明:**

| パラメータ | 型 | 必須 | 説明 | デフォルト値 |
|-----------|------|--------|------|-------------|
| model | string | はい | モデル名（エイリアス） | - |
| messages | array | はい | 会話メッセージの配列 | - |
| temperature | number | いいえ | 0.0-2.0、高いほどランダム | 0.2 |
| top_p | number | いいえ | 0.0-1.0、トークンサンプリング | 0.8 |
| max_tokens | number | いいえ | 生成する最大トークン数 | 1024 |
| stream | boolean | いいえ | ストリーミング応答を有効化 | false |
| presence_penalty | number | いいえ | 新しいトピックへのペナルティ | 0.0 |
| frequency_penalty | number | いいえ | 繰り返しへのペナルティ | 0.0 |

**レスポンスボディ（非ストリーミング）:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1695123456,
  "model": "model-alias",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "生成されたテキスト"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

**レスポンスボディ（ストリーミング）:**

Server-Sent Events (SSE) 形式で逐次送信

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1695123456,"model":"model-alias","choices":[{"index":0,"delta":{"content":"生成"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1695123456,"model":"model-alias","choices":[{"index":0,"delta":{"content":"され"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1695123456,"model":"model-alias","choices":[{"index":0,"delta":{"content":"た"},"finish_reason":"stop"}]}

data: [DONE]
```

**補足:**
- `max_tokens` は非ストリーミングで最大 1024、ストリーミングで最大 2096 に制限されます。
- `temperature` / `top_p` が省略された場合はサーバー設定のデフォルト値が使用されます。

**ステータスコード:**

| コード | 説明 |
|--------|------|
| 200 | 成功 |
| 400 | 不正なリクエスト |
| 422 | バリデーションエラー |
| 500 | 内部サーバーエラー |

---

### 2.2 モデル一覧取得

**エンドポイント**: `GET /v1/models`

**リクエストパラメータ**: なし

**レスポンスボディ:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-alias",
      "object": "model",
      "created": 1686935002,
      "owned_by": "tane"
    }
  ]
}
```

---

### 2.3 ステータス確認

**エンドポイント**: `GET /v1/status`

**リクエストパラメータ**: なし

**レスポンスボディ:**

```json
{
  "status": "ok"
}
```

---

### 2.4 バージョン情報取得

**エンドポイント**: `GET /v1/api/version`

**リクエストパラメータ**: なし

**レスポンスボディ:**

```json
{
  "version": "1.0.0"
}
```

---

### 2.5 管理用API (CLI連携用)
ユーザーがCLIから設定を変更するための内部APIです。

#### 2.5.1 モデル追加
**エンドポイント**: `POST /admin/models`
**リクエストボディ**:
```json
{
  "name": "model_alias",
  "path": "/abs/path/to/model.gguf",
  "stop": ["</s>", "<|im_end|>"]
}
```

#### 2.5.2 モデル削除
**エンドポイント**: `DELETE /admin/models/:name`

#### 2.5.3 設定再読み込み
**エンドポイント**: `POST /admin/reload`
`model_map.json`や`models/`ディレクトリの内容を再スキャンします。

---

## 3. エラーハンドリング

### 3.1 エラーレスポンス形式

```json
{
  "error": {
    "code": "error_code",
    "message": "エラーメッセージ",
    "type": "error_type",
    "param": null
  }
}
```

### 3.2 エラーコード一覧

| コード | タイプ | 説明 |
|--------|--------|------|
| model_not_found | invalid_request | 指定されたモデルが見つかりません |
| model_load_failed | internal_server_error | モデルのロードに失敗しました |
| inference_failed | internal_server_error | 推論に失敗しました |
| processing_timeout | internal_server_error | 処理タイムアウトが発生しました |
| validation_error | invalid_request | リクエストのバリデーションに失敗しました |

---

## 4. 使用例

### 4.1 curl

```bash
# 非ストリーミング
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5_Q8",
    "messages": [
      {"role": "user", "content": "こんにちは"}
    ],
    "stream": false
  }'

# ストリーミング
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5_Q8",
    "messages": [
      {"role": "user", "content": "こんにちは"}
    ],
    "stream": true
  }'
```

### 4.2 Python (OpenAI SDK)

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
        {"role": "user", "content": "こんにちは"}
    ],
    stream=False
)

print(response.choices[0].message.content)

# ストリーミング
stream = client.chat.completions.create(
    model="LFM2.5_Q8",
    messages=[
        {"role": "user", "content": "こんにちは"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 5. 制限事項

- 同時接続数: 1リクエスト（VRAM枯渇防止のため）
- 最大コンテキスト長: モデル依存（通常4096-8192トークン）
- 最大トークン生成数: 1024トークン（非ストリーミング）、2096トークン（ストリーミング）
- ストップシーケンスはデフォルト値と `model_map.json` の `stop` をマージして適用
