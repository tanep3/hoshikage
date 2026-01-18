# 星影 - Hoshikage インストール・運用ガイド

## 目次

1. [前提条件](#前提条件)
2. [開発環境セットアップ](#開発環境セットアップ)
3. [本番環境デプロイ](#本番環境デプロイ)
4. [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

### ハードウェア要件

| 項目 | 最小要件 | 推奨要件 |
|------|---------|---------|
| CPU | 8コア以上 | 16コア以上（Ryzen 7900相当） |
| メモリ | 16GB以上 | 32GB以上 |
| GPU | VRAM 8GB以上 | VRAM 12GB以上 |
| ストレージ | SSD 50GB以上 | NVMe SSD 100GB以上 |

### ソフトウェア要件

- **OS**: Linux（Ubuntu 20.04以降推奨）
- **Python**: 3.10以上
- **CUDA**: 11.8以上
- **Docker**: 20.10以上（本番環境のみ）
- **Docker Compose**: 2.0以上（本番環境のみ）
- **sudo権限**: RAMディスクマウント時に必要

---

## 開発環境セットアップ

開発中は**Dockerを使わず、venv環境で直接実行**します。

### 1. リポジトリのクローン

```bash
git clone /mnt/pluto/Programming/git/AI/hoshikage.git
cd hoshikage
```

### 2. ビルドツールのインストール

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake
```

### 3. CUDAツールキットの確認

```bash
# CUDAバージョンを確認
nvcc --version

# GPUが認識されているか確認
nvidia-smi
```

### 4. 環境変数の設定

```bash
# CUDAのパスを設定
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# llama-cpp-pythonのビルドオプションを設定（GGML_CUDAを使用）
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
```

**永続化する場合（推奨）:**
```bash
# ~/.bashrcに追加
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 設定を反映
source ~/.bashrc
```

### 5. Python仮想環境の作成

```bash
# venvを作成
python3 -m venv venv

# venvをアクティベート
source venv/bin/activate
```

### 6. 依存関係のインストール（CUDA対応版）

```bash
# 環境変数を設定（venv内で再度設定）
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

# 依存関係をインストール
pip install --upgrade pip
# PyTorchをCUDA 12.4対応版でインストール（システムのバージョンに合わせて調整）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

* llama-cpp-python ではKV Cacheや Flash Attention をデフォルトでサポートしていないため、llama.cppを自分でビルドする必要がある。

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA_F16=ON -DGGML_CUDA_PEER_ACCESS=ON -DCMAKE_CUDA_ARCHITECTURES="native"

  cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \                  # これでCUDA有効（旧: LLAMA_CUBLAS）
  -DGGML_FLASH_ATTN=ON \            # Flash Attention有効（最新のフラグ）
  -DLLAMA_BUILD_SERVER=ON \         # 必要ならサーバーもビルド
  -DLLAMA_BUILD_TESTS=OFF \          # テストはオフでOK
  -DGGML_CUDA_F16=ON \         # FP16計算を強化（RTX30/40系で速くなる）
  -DGGML_CUDA_PEER_ACCESS=ON \  # 複数GPUあるときに便利
  -DCMAKE_CUDA_ARCHITECTURES="native" # 今のGPU
cmake --build build --config Release -j 10  --target llama-python # -j で並列数（コア数くらい）
# プロジェクトルート直下に llama_cpp_local がある場合の例
export CMAKE_PREFIX_PATH="$(pwd) llama_cpp_local:$CMAKE_PREFIX_PATH"
# Flash Attention + CUDAを有効にするためのフラグ（ビルド時と同じ）
export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release"
# ソースビルドを強制（これがないとpre-built wheel使っちゃうかも）
export FORCE_CMAKE=1
export LLAMA_CPP_PYTHON_FORCE_CMAKE=1
# pip installを実行（これでローカルのllama.cppを参照するよ！）
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose --no-binary llama-cpp-python


### 8. 環境変数ファイルの設定

```bash
# .env.exampleをコピー
cp .env.example .env

# 必要に応じて編集
nano .env
```

### 9. 開発サーバーの起動

**方法1: 起動スクリプトを使用（推奨）**
```bash
./start-dev.sh
```

**方法2: 手動で起動**
```bash
# CUDAライブラリパスを設定
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# srcディレクトリから起動
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 3030
```

**開発サーバーのメリット:**
- ✅ コード変更が即座に反映（`--reload`）
- ✅ デバッグが簡単
- ✅ ビルド時間なし
- ✅ ログが見やすい

**アクセス:**
```bash
# ステータス確認
curl http://localhost:3030/v1/status

# チャット補完
curl -X POST http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "stream": false
  }'
```

### 10. モデルの登録（ストップシーケンス対応）

モデルを登録する際、そのモデル固有の**ストップトークン**（生成停止合図）を指定できます。特に **ChatML形式** や **Llama 3** などを使用する場合は重要です。

基本コマンド:
```bash
python hoshikage.py add [モデルパス] [モデル名] [ストップトークン(カンマ区切り)]
```

**LiquidAI LFM2.5 (ChatML形式) の例:**
ChatML形式では `<|im_end|>` が必須です。
```bash
python hoshikage.py add \
  /home/tane/datas/LLM/LFM/LFM2.5-1.2B-JP-Q8_0.gguf \
  LFM2.5_Q8 \
  "<|im_end|>,<|eot_id|>"
```

※ ストップトークンを省略した場合でも、主要なトークン（`<|im_end|>`, `</s>`など）が自動的にデフォルト値として設定されますが、モデル固有の特殊なタグがある場合は明示的に指定してください。
```

---

## 本番環境デプロイ

本番環境では**Docker Composeを使用**します。

### 方法1: 本番環境でビルド（推奨）

**メリット:** BuildKit cacheが効く、更新が高速

```bash
# デプロイスクリプトを実行
./deploy-production.sh /opt/hoshikage

# または、手動でデプロイ
mkdir -p /opt/hoshikage
cd /opt/hoshikage
git clone /mnt/pluto/Programming/git/AI/hoshikage.git .
cp .env.example .env
nano .env
DOCKER_BUILDKIT=1 docker-compose build
docker-compose up -d
```

**更新時:**
```bash
./deploy-production.sh /opt/hoshikage
```

### 方法2: デプロイパッケージ（インターネット接続なし）

**メリット:** インターネット接続不要、事前ビルド済み

```bash
# 開発環境でパッケージ作成
./create-deploy-package.sh

# 本番環境に転送
scp hoshikage-deploy-*.tar.gz server:/opt/

# 本番環境で展開
cd /opt
tar -xzf hoshikage-deploy-*.tar.gz
cd deploy-package
docker load -i ../hoshikage-image.tar
cp .env.example .env
nano .env
docker-compose up -d
```

### 運用コマンド

```bash
# ステータス確認
docker-compose ps

# ログ確認
docker-compose logs -f

# 再起動
docker-compose restart

# 停止
docker-compose down

# 完全削除（ボリュームも削除）
docker-compose down -v
```

---

## トラブルシューティング

### llama-cpp-pythonのインストールに失敗する

**症状:**
```
ERROR: Failed building wheel for llama-cpp-python
```

**解決方法:**

1. **環境変数が設定されているか確認:**
```bash
echo $CMAKE_ARGS
echo $FORCE_CMAKE
```

2. **CUDAツールキットがインストールされているか確認:**
```bash
nvcc --version
```

3. **ビルドツールがインストールされているか確認:**
```bash
cmake --version
gcc --version
```

4. **再度インストール:**
```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### CUDAが認識されない

**症状:**
```
CUDA not available
```

**解決方法:**

1. **nvidia-smiでGPUを確認:**
```bash
nvidia-smi
```

2. **CUDAドライバーを再インストール:**
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-xxx  # xxxはバージョン番号
```

3. **システムを再起動:**
```bash
sudo reboot
```

### Dockerコンテナが起動しない

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

### モデル管理ファイルパスの設定

環境変数（`.env`）で指定するパスが `./` で始まる場合、プログラムはそれを**プロジェクトルートからの相対パス**として解釈します。

例：
- `MODEL_MAP_FILE=./src/models/model_map.json` -> `{プロジェクトルート}/src/models/model_map.json`
- `CHROMA_PATH=./data/hoshikage_chroma_db` -> `{プロジェクトルート./data/hoshikage_chroma_db`

これにより、`src/`ディレクトリから起動しても、プロジェクトルートから起動しても、同じ設定ファイルを参照できます。

---

## 依存関係の詳細

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| fastapi | >=0.104.0 | Webフレームワーク |
| uvicorn | >=0.24.0 | ASGIサーバー |
| pydantic | >=2.5.0 | データバリデーション |
| python-dotenv | >=1.0.0 | 環境変数管理 |
| llama-cpp-python | >=0.2.0 | LLM推論エンジン（CUDA対応） |
| chromadb | >=0.4.0 | ベクトルデータベース |
| sentence-transformers | >=2.2.0 | 埋め込みモデル |
| sentencepiece | >=0.1.99 | トークン化 |
| fugashi | >=1.3.0 | 日本語形態素解析 |
| unidic-lite | >=1.0.8 | 日本語辞書 |
| scikit-learn | >=1.3.0 | K-Meansクラスタリング |
| numpy | >=1.24.0 | 数値計算 |
| requests | >=2.31.0 | HTTPリクエスト |

---

## 参考リンク

- [llama-cpp-python公式ドキュメント](https://github.com/abetlen/llama-cpp-python)
- [CUDA Toolkit ダウンロード](https://developer.nvidia.com/cuda-downloads)
- [FastAPI公式ドキュメント](https://fastapi.tiangolo.com/)
- [ChromaDB公式ドキュメント](https://docs.trychroma.com/)
