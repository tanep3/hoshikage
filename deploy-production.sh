#!/bin/bash
# 星影 - 本番環境デプロイスクリプト

set -e

# Usage表示
show_usage() {
    cat << EOF
使い方: $0 <デプロイ先ディレクトリ>

説明:
  星影を本番環境にデプロイします。
  指定したディレクトリにGitリポジトリをクローン（または更新）し、
  BuildKit cacheを活用してDockerイメージをビルド、起動します。

引数:
  <デプロイ先ディレクトリ>  デプロイ先のディレクトリパス（絶対パスまたは相対パス）

例:
  $0 ~/hoshikage-deploy
  $0 /opt/hoshikage
  $0 ./production

オプション:
  -h, --help  このヘルプを表示

EOF
}

# 引数チェック
if [ $# -eq 0 ]; then
    echo "❌ エラー: デプロイ先ディレクトリを指定してください"
    echo ""
    show_usage
    exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

DEPLOY_DIR="$1"

echo "🚀 星影を本番環境にデプロイします..."
echo "📁 デプロイ先: $DEPLOY_DIR"
echo ""

# 1. デプロイディレクトリを作成
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

# 2. Gitからクローン（または最新版を取得）
if [ ! -d ".git" ]; then
    echo "📥 リポジトリをクローン中..."
    git clone /mnt/pluto/Programming/git/AI/hoshikage.git .
else
    echo "🔄 最新版を取得中..."
    git pull
fi

# 3. 環境変数設定
if [ ! -f ".env" ]; then
    echo "⚙️ 環境変数ファイルを作成中..."
    cp .env.example .env
    echo "⚠️ .envファイルを編集してください"
    nano .env
fi

# 4. Dockerイメージをビルド（BuildKit cache使用）
echo "🔨 Dockerイメージをビルド中（BuildKit cache使用）..."
DOCKER_BUILDKIT=1 docker-compose build

# 5. コンテナを起動
echo "🚀 コンテナを起動中..."
docker-compose up -d

# 6. ステータス確認
echo "✅ デプロイ完了！"
echo ""
echo "📊 ステータス確認:"
docker-compose ps
echo ""
echo "🔍 ヘルスチェック:"
sleep 5
if command -v jq &> /dev/null; then
    curl -s http://localhost:3030/v1/status | jq .
else
    curl -s http://localhost:3030/v1/status
fi

echo ""
echo "📝 ログ確認: cd $DEPLOY_DIR && docker-compose logs -f"
echo "🛑 停止: cd $DEPLOY_DIR && docker-compose down"
echo "🔄 再起動: cd $DEPLOY_DIR && docker-compose restart"
