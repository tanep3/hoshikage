# syntax=docker/dockerfile:1

# ベースイメージ（CUDA対応）
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 作業ディレクトリ
WORKDIR /app

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonのシンボリックリンク
RUN ln -s /usr/bin/python3 /usr/bin/python

# requirements.txtをコピー
COPY requirements.txt .

# 必要パッケージインストール（BuildKit cacheを使用）
# タイムアウトとリトライ設定でネットワーク問題に対応
# CMAKE_ARGSとFORCE_CMAKEを設定してCUDA対応版をビルド
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 3 --upgrade pip && \
    pip install --timeout 120 --retries 3 -r requirements.txt

# アプリケーションコードをコピー
COPY src/ ./src/
COPY .env.example .env

# ポート公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/status || exit 1

# 起動コマンド
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
