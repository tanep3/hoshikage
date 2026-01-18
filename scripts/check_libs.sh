#!/bin/bash

# 星影 - ライブラリチェック・設定スクリプト
# llama.cppライブラリの場所を検出し、環境変数を設定します

set -e

CONFIG_DIR="$HOME/.config/hoshikage"
LIB_DIR="$CONFIG_DIR/lib"

echo "🔍 星影 - ライブラリチェック"

# ユーザーライブラリを確認
if [ -d "$LIB_DIR" ] && ls "$LIB_DIR"/*.{a,so} 1>/dev/null 2>&1; then
    echo "✅ ユーザーライブラリが見つかりました: $LIB_DIR"
    USER_LIB="$LIB_DIR"
else
    echo "⚠️  ユーザーライブラリは見つかりません"
    USER_LIB=""
fi

# システムCUDAライブラリを確認
CUDA_LIBS=("/usr/local/cuda/targets/x86_64-linux/lib" "/usr/local/cuda/lib64" "/usr/local/cuda/lib")

SYS_LIB=""
for lib_dir in "${CUDA_LIBS[@]}"; do
    if [ -d "$lib_dir" ] && ls "$lib_dir"/libcuda.so 1>/dev/null 2>&1; then
        SYS_LIB="$lib_dir"
        echo "✅ システムCUDAライブラリが見つかりました: $lib_dir"
        break
    fi
done

# 環境変数を設定（優先順位：ユーザーライブラリ > システムライブラリ > 既存設定）
if [ -n "$USER_LIB" ]; then
    export LD_LIBRARY_PATH="$USER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "✅ LD_LIBRARY_PATHを設定しました"
    echo "   優先: $USER_LIB"
    echo "   既存設定: ${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}"
elif [ -n "$SYS_LIB" ]; then
    export LD_LIBRARY_PATH="$SYS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "✅ システムCUDAライブラリを使用します"
    echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
else
    echo "⚠️  CUDAライブラリが見つかりません"
    echo "   llama.cppのビルドが必要です"
    exit 1
fi

# 動的リンク用ライブラリの確認
DYN_LIBS=("")
if [ -n "$USER_LIB" ]; then
    if ls "$USER_LIB"/*.so 1>/dev/null 2>&1; then
        DYN_LIBS="$(ls "$USER_LIB"/*.so)"
        echo "🔒 動的リンク用ライブラリ:"
        for lib in $DYN_LIBS; do
            echo "   - $lib"
    fi
fi

echo ""
echo "✅ ライブラリチェック完了"
echo ""
if [ -n "$USER_LIB" ]; then
    echo "ユーザーライブラリを使用します: $USER_LIB"
elif [ -n "$SYS_LIB" ]; then
    echo "システムCUDAライブラリを使用します: $SYS_LIB"
fi
echo ""
echo "環境変数:"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
