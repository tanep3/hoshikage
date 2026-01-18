# 星影 - ライブラリ管理スクリプトの使用方法

## 概要

星影は動的リンクでllama.cppを使用します。ライブラリの探索順位は以下の通りです：

1. **~/.config/hoshikage/lib** (ユーザーライブラリ - 最優先)
2. システムCUDAライブラリ
3. LD_LIBRARY_PATH環境変数

## ステップ1: ライブラリのセットアップ

カスタムビルドのllama.cppを使用する場合、まずセットアップしてください。

```bash
# 方法1: プロジェクトのスクリプトを使用
./scripts/setup_libs.sh

# 方法2: 手動でコピー
mkdir -p ~/.config/hoshikage/lib
cp llama_cpp_local/lib/static/*.{a,so} ~/.config/hoshikage/lib/
```

## ステップ2: 環境設定

```bash
# スクリプトを実行して、環境変数を設定
source ./scripts/check_libs.sh
```

または手動で設定：

```bash
# ユーザーライブラリがある場合
export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH

# システムCUDAライブラリを使用する場合
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## ステップ3: 実行

```bash
# hoshikageを実行
./target/release/hoshikage

# 設定された環境変数を確認
echo $LD_LIBRARY_PATH
```

## Windowsの場合

```powershell
# ユーザーライブラリを作成
mkdir -p $env:USERPROFILE\.config\hoshikage\lib

# ライブラリをコピー
# llama_cpp_local\lib\static\*.{a,dll} をコピー

# 環境変数を設定
$env:PATH = "$env:PATH;C:\Users\$env:USERNAME\.config\hoshikage\lib"

# 実行
.\target\release\hoshikage.exe
```

---

**作成日:** 2026-01-18
