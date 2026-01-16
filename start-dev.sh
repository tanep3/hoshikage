#!/bin/bash
# 星影 - 開発環境起動スクリプト

# プロジェクトルートを取得
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# CUDAライブラリパスを設定
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 仮想環境をアクティベート
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "❌ 仮想環境 (venv) が見つかりません。INSTALL.mdに従って作成してください。"
    exit 1
fi

# srcディレクトリに移動
cd "$PROJECT_ROOT/src"

# 開発サーバーを起動
uvicorn main:app --reload --host 0.0.0.0 --port 3030
