# デプロイガイド：星影 - Hoshikage

**バージョン:** 1.0.0  
**最終更新日:** 2026-01-16  
**著者:** Tane Channel Technology

---

## 目次

1. [Docker環境でのデプロイ](#1-docker環境でのデプロイ)
2. [本番環境への展開](#2-本番環境への展開)
3. [環境変数の設定](#3-環境変数の設定)
4. [トラブルシューティング](#4-トラブルシューティング)

---

## 1. Docker環境でのデプロイ

### 1.1 前提条件

- **Docker**: 20.10以上
- **Docker Compose**: 2.0以上
- **NVIDIA Container Toolkit**: GPU使用時に必要

#### NVIDIA Container Toolkitのインストール

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 1.2 ビルドと起動

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/hoshikage.git
cd hoshikage

# 環境変数ファイルを作成
cp .env.example .env

# 必要に応じて.envを編集
nano .env

# Dockerイメージをビルド（BuildKit cacheを使用）
DOCKER_BUILDKIT=1 docker-compose build

# コンテナを起動
docker-compose up -d
```

### 1.3 動作確認

```bash
# ログを確認
docker-compose logs -f

# ステータス確認
curl http://localhost:8000/v1/status

# ヘルスチェック
docker-compose ps
```

### 1.4 停止と再起動

```bash
# 停止
docker-compose down

# 再起動
docker-compose restart

# 完全削除（ボリュームも削除）
docker-compose down -v
```

### モデルの登録

Docker環境でモデルを使用する場合、コンテナ内のパス（デフォルトでは `/models`）として登録する必要があります。

```bash
# コンテナ内でhoshikage.pyを実行して登録
docker-compose exec hoshikage python src/hoshikage.py add /models/gemma.gguf gemma
```

これにより、`model_map.json` に正しく `/models` フォルダとしてのパスが記録され、マウント処理が正常に動作します。

---

## 2. 本番環境への展開

### 2.1 推奨構成

```
┌─────────────────────────────────────┐
│ リバースプロキシ（Nginx/Caddy）     │
│ - SSL/TLS終端                       │
│ - レート制限                        │
│ - ロードバランシング                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 星影 Docker コンテナ                │
│ - FastAPI                           │
│ - llama-cpp-python                  │
│ - ChromaDB                          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 永続化ストレージ                    │
│ - モデルファイル                    │
│ - ChromaDBデータ                    │
│ - モデル管理JSON                    │
└─────────────────────────────────────┘
```

### 2.2 Nginxリバースプロキシの設定例

```nginx
upstream hoshikage {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # HTTPSにリダイレクト
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL証明書
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # セキュリティヘッダー
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # レート制限
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
    
    location / {
        proxy_pass http://hoshikage;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # ストリーミング対応
        proxy_buffering off;
        proxy_cache off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # タイムアウト設定
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
}
```

### 2.3 systemdサービスとして登録

```bash
# /etc/systemd/system/hoshikage.service
[Unit]
Description=Hoshikage LLM API Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/hoshikage
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
# サービスを有効化
sudo systemctl daemon-reload
sudo systemctl enable hoshikage
sudo systemctl start hoshikage

# ステータス確認
sudo systemctl status hoshikage
```

---

## 3. 環境変数の設定

### 3.1 本番環境用の設定

```bash
# RAMディスク設定（本番環境では無効化を推奨）
RAMDISK_PATH=/mnt/temp/hoshikage
RAMDISK_SIZE=12

# タイムアウト設定（本番環境では長めに設定）
IDLE_TIMEOUT_SECONDS=600
GREAT_TIMEOUT=120

# モデル管理
MODEL_MAP_FILE=./src/models/model_map.json
TAG_CACHE_FILE=./src/models/tags_cache.json
TAG_OLLAMA_FILE=./src/models/tags_ollama.json

# ChromaDB設定
CHROMA_PATH=./data/hoshikage_chroma_db
SENTENCE_BERT_MODEL=cl-nagoya/ruri-small-v2
```

### 3.2 セキュリティ設定

**将来実装予定:**
- APIキー認証
- レート制限
- アクセスログ

---

## 4. トラブルシューティング

### 4.1 コンテナが起動しない

**症状:**
```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**解決方法:**

1. **NVIDIA Container Toolkitがインストールされているか確認:**
```bash
nvidia-container-cli --version
```

2. **Dockerを再起動:**
```bash
sudo systemctl restart docker
```

3. **GPUが認識されているか確認:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 4.2 ビルドが遅い

**症状:**
- Dockerイメージのビルドに時間がかかる

**解決方法:**

1. **BuildKit cacheを使用:**
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

2. **不要なイメージを削除:**
```bash
docker system prune -a
```

### 4.3 メモリ不足

**症状:**
```
CUDA out of memory
```

**解決方法:**

1. **docker-compose.ymlでメモリ制限を設定:**
```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

2. **より小さいモデルを使用**

---

## 5. モニタリング

### 5.1 ログの確認

```bash
# リアルタイムログ
docker-compose logs -f

# 最新100行
docker-compose logs --tail=100

# 特定のサービスのログ
docker-compose logs hoshikage
```

### 5.2 リソース使用状況

```bash
# コンテナのリソース使用状況
docker stats

# GPU使用状況
nvidia-smi
```

---

## 6. バックアップ

### 6.1 バックアップ対象

- `hoshikage_chroma_db/`: ChromaDBデータ
- `src/models/`: モデル管理JSON

### 6.2 バックアップスクリプト

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backup"
DATE=$(date +%Y%m%d_%H%M%S)

# ChromaDBをバックアップ
tar -czf $BACKUP_DIR/chroma_$DATE.tar.gz hoshikage_chroma_db/

# モデル管理JSONをバックアップ
tar -czf $BACKUP_DIR/models_$DATE.tar.gz src/models/

echo "Backup completed: $DATE"
```

---

## 7. アップデート

### 7.1 アップデート手順

```bash
# 最新のコードを取得
git pull origin main

# コンテナを停止
docker-compose down

# イメージを再ビルド（BuildKit cacheを使用）
DOCKER_BUILDKIT=1 docker-compose build

# コンテナを起動
docker-compose up -d

# ログを確認
docker-compose logs -f
```

### 7.2 ロールバック

```bash
# 以前のバージョンにチェックアウト
git checkout <commit-hash>

# イメージを再ビルド
DOCKER_BUILDKIT=1 docker-compose build

# コンテナを起動
docker-compose up -d
```

---

**著者:** Tane Channel Technology  
**最終更新日:** 2026-01-16  
**バージョン:** 1.0.0
