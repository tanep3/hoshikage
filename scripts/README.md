# 星影 - ライブラリ管理スクリプトの使用方法

## 概要

このスクリプトは、llama.cppの静的ライブラリをユーザーライブラリディレクトリにコピーするために使用できます。

## 使用方法

### 1. ユーザーライブラリの準備

```bash
# ライブラリ用ディレクトリを作成
mkdir -p ~/.local/cuda/lib

# システムCUDAライブラリから必要なファイルをコピー
cp /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so ~/.local/cuda/lib/
cp /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so ~/.local/cuda/lib/
cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so ~/.local/cuda/lib/
```

### 2. 環境変数の設定

```bash
# ユーザーライブラリを優先
export LD_LIBRARY_PATH=~/.local/cuda/lib:$LD_LIBRARY_PATH
```

### 3. 実行

```bash
cd /home/tane/dev/AI/hoshikage
./target/release/hoshikage
```

## 注意点

- システムCUDAライブラリのバージョンに合せてください
- `~/.config/hoshikage/lib` には何も配置しないでください
- 本番環境ではシステムCUDAライブラリを使用するのが推奨です

---

**作成日:** 2026-01-18
