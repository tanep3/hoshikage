#!/bin/bash
# 星影 - デプロイメントパッケージ作成スクリプト

set -e

echo "🚀 星影デプロイメントパッケージを作成します..."

# 1. Dockerイメージをビルド
echo "📦 Dockerイメージをビルド中..."
DOCKER_BUILDKIT=1 docker-compose build

# 2. イメージをtarファイルに保存
echo "💾 Dockerイメージを保存中..."
docker save hoshikage:latest -o hoshikage-image.tar

# 3. デプロイメントパッケージを作成
echo "📁 デプロイメントパッケージを作成中..."
mkdir -p deploy-package
cp docker-compose.prod.yml deploy-package/docker-compose.yml
cp .env.example deploy-package/.env.example
cp -r models deploy-package/
mkdir -p deploy-package/src/models
cp src/models/*.json deploy-package/src/models/ 2>/dev/null || true

# 4. README作成
cat > deploy-package/README.md << 'EOF'
# 星影 - デプロイメントパッケージ

## セットアップ手順

1. Dockerイメージをロード
```bash
docker load -i ../hoshikage-image.tar
```

2. 環境変数ファイルを作成
```bash
cp .env.example .env
nano .env  # 必要に応じて編集
```

3. モデルファイルを配置
```bash
# models/ディレクトリにGGUFモデルを配置
cp /path/to/your/model.gguf models/
```

4. コンテナを起動
```bash
docker-compose up -d
```

5. ステータス確認
```bash
curl http://localhost:3030/v1/status
```

## ディレクトリ構成

```
deploy-package/
├── docker-compose.yml    # Docker Compose設定
├── .env.example          # 環境変数テンプレート
├── models/               # モデルファイル配置ディレクトリ
├── src/models/           # モデル管理JSON
└── README.md             # このファイル
```

## トラブルシューティング

### GPUが認識されない
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### ログ確認
```bash
docker-compose logs -f
```
EOF

# 5. パッケージを圧縮
echo "🗜️ パッケージを圧縮中..."
tar -czf hoshikage-deploy-$(date +%Y%m%d).tar.gz deploy-package/ hoshikage-image.tar

echo "✅ デプロイメントパッケージ作成完了！"
echo "📦 ファイル: hoshikage-deploy-$(date +%Y%m%d).tar.gz"
echo ""
echo "📋 デプロイ手順:"
echo "1. パッケージを本番環境に転送"
echo "2. tar -xzf hoshikage-deploy-*.tar.gz"
echo "3. cd deploy-package"
echo "4. README.mdの手順に従ってセットアップ"
