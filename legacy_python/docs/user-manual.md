# ユーザーマニュアル：星影 - Hoshikage

**バージョン:** 1.0.0  
**最終更新日:** 2026-01-16  
**著者:** Tane Channel Technology

---

## 目次

1. [はじめに](#1-はじめに)
2. [セットアップガイド](#2-セットアップガイド)
3. [モデル管理](#3-モデル管理)
4. [サーバーの起動と停止](#4-サーバーの起動と停止)
5. [API使用方法](#5-api使用方法)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [FAQ](#7-faq)

---

## 1. はじめに

星影（ほしかげ）は、GGUFフォーマットの大規模言語モデルをローカル環境で高速に実行し、OpenAI互換のAPIを提供するシステムです。このマニュアルでは、星影のセットアップから基本的な使用方法までを説明します。

### 1.1 このマニュアルの対象読者

- 星影を初めて使用する方
- ローカル環境でLLMを実行したい方
- OpenAI APIの代替を探している方

---

## 2. セットアップガイド

### 2.1 システム要件の確認

#### ハードウェア要件

最小要件を満たしているか確認してください：

```bash
# CPU コア数を確認
nproc

# メモリ容量を確認
free -h

# GPU情報を確認（NVIDIA GPUの場合）
nvidia-smi
```

**必要要件:**
- CPU: 8コア以上
- メモリ: 16GB以上
- GPU: VRAM 8GB以上（推奨）
- ストレージ: SSD 50GB以上

#### ソフトウェア要件

```bash
# Pythonバージョンを確認
python3 --version  # 3.10以上が必要

# CUDAバージョンを確認（GPU使用時）
nvcc --version  # 11.8以上が必要
```

### 2.2 依存関係のインストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/hoshikage.git
cd hoshikage/src

# 依存関係をインストール
pip install -r requirements.txt
```

**インストールされるパッケージ:**
- `fastapi`: Webフレームワーク
- `uvicorn`: ASGIサーバー
- `pydantic`: データバリデーション
- `python-dotenv`: 環境変数管理
- `llama-cpp-python`: 推論エンジン
- `chromadb`: ベクトルデータベース
- `sentence-transformers`: 埋め込みモデル
- `sentencepiece`: トークン化
- `fugashi`, `unidic_lite`: 日本語処理

### 2.3 環境変数の設定

`src/.env`ファイルを作成し、以下の内容を設定：

```bash
# RAMディスク設定
RAMDISK_PATH=/mnt/temp/hoshikage
RAMDISK_SIZE=12

# タイムアウト設定
IDLE_TIMEOUT_SECONDS=300
GREAT_TIMEOUT=60

# モデル管理
MODEL_MAP_FILE=./models/model_map.json
TAG_CACHE_FILE=./models/tags_cache.json
TAG_OLLAMA_FILE=./models/tags_ollama.json

# ChromaDB設定
CHROMA_PATH=./data/hoshikage_chroma_db
SENTENCE_BERT_MODEL=cl-nagoya/ruri-small-v2
```

**各設定項目の説明:**

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `RAMDISK_PATH` | RAMディスクのマウントパス | `/mnt/temp/hoshikage` |
| `RAMDISK_SIZE` | RAMディスクのサイズ（GB） | `12` |
| `IDLE_TIMEOUT_SECONDS` | 非アクティブ検出閾値（秒） | `300` |
| `GREAT_TIMEOUT` | RAMディスクアンマウント閾値（分） | `60` |
| `MODEL_MAP_FILE` | モデルマップファイルパス | `./models/model_map.json` |
| `TAG_CACHE_FILE` | タグキャッシュファイルパス | `./models/tags_cache.json` |
| `TAG_OLLAMA_FILE` | Ollama互換タグファイルパス | `./models/tags_ollama.json` |
| `CHROMA_PATH` | ChromaDBデータパス | `./data/hoshikage_chroma_db` |
| `SENTENCE_BERT_MODEL` | 埋め込みモデル名 | `cl-nagoya/ruri-small-v2` |

### 2.4 RAMディスクの設定（sudo権限が必要）

RAMディスクを使用するには、sudo権限が必要です。以下のコマンドでsudoersファイルを編集します：

```bash
sudo visudo
```

以下の行を追加（`your-username`を実際のユーザー名に置き換え）：

```
your-username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs * /mnt/temp/hoshikage
your-username ALL=(ALL) NOPASSWD: /bin/umount /mnt/temp/hoshikage
```

### 2.5 モデルのダウンロード

GGUFフォーマットのモデルをダウンロードします。

**推奨モデル:**
- [Gemma-3 12B GGUF](https://huggingface.co/google/gemma-3-12b-it-GGUF)
- [Gemma-3 4B GGUF](https://huggingface.co/google/gemma-3-4b-it-GGUF)

```bash
# Hugging Face CLIを使用してダウンロード
huggingface-cli download google/gemma-3-12b-it-GGUF gemma-3-12b-it-q4_0.gguf --local-dir /path/to/models
```

---

## 3. モデル管理

### 3.1 モデルの登録

```bash
cd /home/tane/dev/AI/hoshikage/src

# モデルを登録
python hoshikage.py add /path/to/your/model.gguf your-model-name
```

**例:**
```bash
python hoshikage.py add /home/tane/datas/LLM/google/gemma-3-12b-it-q4_0.gguf hoshikage-gemma3-12B-google
```

**成功メッセージ:**
```
✅ モデル 'hoshikage-gemma3-12B-google' を追加しました。
```

### 3.2 モデルの一覧表示

```bash
python hoshikage.py list
```

**出力例:**
```
📦 登録済みモデル一覧（3件）:
 - hoshikage-gemma3-12B-google: /home/tane/datas/LLM/google/gemma-3-12b-it-q4_0.gguf (8192.50 MB)
 - hoshikage-gemma3-4B-google: /home/tane/datas/LLM/google/gemma-3-4b-it-q4_0.gguf (3072.25 MB)
 - hoshikage-TinySwallow-1.5B: /home/tane/datas/LLM/sakana_ai/tinyswallow-1.5b-instruct-q8_0.gguf (1536.75 MB)
```

### 3.3 モデルの削除

```bash
python hoshikage.py remove your-model-name
```

**例:**
```bash
python hoshikage.py remove hoshikage-TinySwallow-1.5B
```

**成功メッセージ:**
```
🗑️ モデル 'hoshikage-TinySwallow-1.5B' を削除しました。
```

---

## 4. サーバーの起動と停止

### 4.1 サーバーの起動

```bash
cd /home/tane/dev/AI/hoshikage/src

# uvicornで起動
uvicorn main:app --host 0.0.0.0 --port 8000

# または、起動スクリプトを使用
bash start.sh
```

**起動成功メッセージ:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4.2 サーバーの停止

```
Ctrl + C
```

### 4.3 バックグラウンドで起動

```bash
# バックグラウンドで起動
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > hoshikage.log 2>&1 &

# プロセスIDを確認
ps aux | grep uvicorn

# 停止する場合
kill <プロセスID>
```

---

## 5. API使用方法

### 5.1 ステータス確認

```bash
curl http://localhost:8000/v1/status
```

**レスポンス:**
```json
{
  "status": "ok"
}
```

### 5.2 モデル一覧取得

```bash
curl http://localhost:8000/v1/models
```

**レスポンス:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "hoshikage-gemma3-12B-google",
      "object": "model",
      "created": 1686935002,
      "owned_by": "tane"
    }
  ]
}
```

### 5.3 チャット補完（非ストリーミング）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hoshikage-gemma3-12B-google",
    "messages": [
      {"role": "user", "content": "こんにちは"}
    ],
    "stream": false
  }'
```

### 5.4 チャット補完（ストリーミング）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hoshikage-gemma3-12B-google",
    "messages": [
      {"role": "user", "content": "こんにちは"}
    ],
    "stream": true
  }' \
  --no-buffer
```

### 5.5 Pythonでの使用例

```python
from openai import OpenAI

# 星影APIを使用するように設定
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 認証なしでも必要
)

# チャット補完
response = client.chat.completions.create(
    model="hoshikage-gemma3-12B-google",
    messages=[
        {"role": "user", "content": "こんにちは"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

---

## 6. トラブルシューティング

### 6.1 サーバーが起動しない

**症状:**
```
ERROR: Could not find module 'main'
```

**解決方法:**
```bash
# 正しいディレクトリにいるか確認
pwd  # /home/tane/dev/AI/hoshikage/src であるべき

# main.pyが存在するか確認
ls main.py
```

---

### 6.2 モデルのロードに失敗する

**症状:**
```
ERROR: モデルのロードに失敗しました
```

**解決方法:**

1. **モデルファイルが存在するか確認:**
```bash
python hoshikage.py list
```

2. **RAMディスクがマウントされているか確認:**
```bash
mount | grep /mnt/temp/hoshikage
```

3. **sudo権限が設定されているか確認:**
```bash
sudo -l | grep mount
```

---

### 6.3 VRAM不足エラー

**症状:**
```
CUDA out of memory
```

**解決方法:**

1. **より小さいモデルを使用:**
   - 12Bモデル → 4Bモデル

2. **コンテキスト長を減らす:**
   - `main.py`の`n_ctx`を`4096`から`2048`に変更

3. **GPU レイヤー数を減らす:**
   - `main.py`の`n_gpu_layers`を`-1`から`30`などに変更

---

### 6.4 応答が遅い

**症状:**
- 初回応答に時間がかかる

**解決方法:**

1. **RAMディスクが有効か確認:**
```bash
mount | grep /mnt/temp/hoshikage
```

2. **GPUが使用されているか確認:**
```bash
nvidia-smi
```

3. **モデルサイズを確認:**
   - 大きいモデルほど遅くなります

---

### 6.5 ChromaDBエラー

**症状:**
```
ERROR: ChromaDBの保存エラー
```

**解決方法:**

1. **ChromaDBディレクトリを削除して再作成:**
```bash
rm -rf ./data/hoshikage_chroma_db
```

2. **サーバーを再起動**

---

## 7. FAQ

### Q1: OpenAI APIと完全に互換性がありますか？

**A:** はい、`/v1/chat/completions`エンドポイントはOpenAI APIと完全に互換性があります。既存のOpenAIクライアントライブラリがそのまま使用できます。

---

### Q2: 複数のモデルを同時に使用できますか？

**A:** いいえ、現在は同時に1つのモデルのみ使用可能です。ただし、リクエストごとに異なるモデルを指定することで、自動的に切り替わります。

---

### Q3: 非アクティブ時にメモリは自動的に解放されますか？

**A:** はい、300秒（5分）間非アクティブの場合、モデルが自動的にアンロードされます。60分間非アクティブの場合、RAMディスクもアンマウントされます。

---

### Q4: ストリーミング応答はサポートされていますか？

**A:** はい、`stream: true`を指定することで、リアルタイムストリーミング応答が利用できます。

---

### Q5: 会話履歴はどのように管理されますか？

**A:** ChromaDBに会話履歴が保存され、意味クラスタリングで自動的に要約されます。直近3往復の会話は原文のまま保持されます。

---

### Q6: RAMディスクは必須ですか？

**A:** いいえ、RAMディスクは任意です。ただし、RAMディスクを使用することで、モデルのロード時間を大幅に短縮できます。

---

### Q7: どのようなモデル形式がサポートされていますか？

**A:** GGUFフォーマットのモデルのみサポートされています。

---

### Q8: 認証機能はありますか？

**A:** 現在のバージョンでは認証機能は実装されていません。将来のバージョンで追加予定です。

---

## 付録

### A. 推奨モデル

| モデル名 | サイズ | VRAM要件 | 用途 |
|---------|-------|---------|------|
| Gemma-3 12B Q4_0 | 8GB | 10GB | 高品質な応答 |
| Gemma-3 4B Q4_0 | 3GB | 4GB | バランス型 |
| TinySwallow 1.5B Q8_0 | 1.5GB | 2GB | 軽量・高速 |

### B. パフォーマンスチューニング

**高速化のヒント:**
1. RAMディスクを使用
2. GPUを最大限活用（`n_gpu_layers=-1`）
3. バッチサイズを調整（`n_batch=512`）
4. スレッド数を最適化（`n_threads=20`）

**メモリ削減のヒント:**
1. 小さいモデルを使用
2. コンテキスト長を減らす（`n_ctx=2048`）
3. GPU レイヤー数を減らす（`n_gpu_layers=30`）

---

**著者:** Tane Channel Technology  
**最終更新日:** 2026-01-16  
**バージョン:** 1.0.0
