# 星影 (Hoshikage) - ライブラリ運用ガイド

**作成日:** 2026-01-18  
**プロジェクト:** 星影 (Hoshikage)

---

## 1. 概要: 動的リンクと運用方針
星影は **動的ライブラリ (`libllama.so`)** を使用して動作します。
これにより、ユーザーが自分の環境に合わせてビルドしたライブラリを自由に差し替えたり、バージョンアップしたりすることが可能です。

### 探索順序 (優先度順)
1. **ユーザーライブラリ**: `~/.config/hoshikage/lib/`
2. **ログ**: 起動時にどのライブラリがロードされたかを表示します。

## 2. ライブラリの配置場所
ユーザーがライブラリを配置する標準ディレクトリは以下の通りです。ディレクトリが存在しない場合は作成してください。

```bash
mkdir -p ~/.config/hoshikage/lib
```

### 必要なファイル
ここに `libllama.so` (または `libllama.so.0` などの実体) を配置します。

## 3. セットアップ手順

### ケースA: 提供されたライブラリを使う場合
プロジェクトに含まれるビルド済みライブラリをコピーして使用します。

```bash
# プロジェクトルートで実行
cp llama_cpp_local/lib/libllama.so.0 ~/.config/hoshikage/lib/
cp llama_cpp_local/llama-cli ~/.config/hoshikage/lib/
```

### ケースB: 自分でビルドする場合 (上級者向け)
`llama.cpp` をご自身の環境でビルドし、生成された `libllama.so` を配置してください。

```bash
# 例: llama.cpp ビルド後
cp /path/to/llama.cpp/build/src/libllama.so ~/.config/hoshikage/lib/
```

## 4. 実行環境のセットアップ
`hoshikage` コマンドを端末から直接実行できるように、`LD_LIBRARY_PATH` を `.bashrc` (または `.zshrc`) に登録してください。

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.config/hoshikage/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 5. インストールと実行
Cargoを使ってシステムにインストールします。

```bash
cargo install --path .
```

これで、ターミナルから直接実行できます。

```bash
hoshikage
```

---

## 6. Windowsユーザーの方へ

Windows環境では、ライブラリの配置場所や環境変数が異なります。

### 配置場所 (Windows準拠)
`%APPDATA%\hoshikage\lib`
(通常は `C:\Users\ユーザー名\AppData\Roaming\hoshikage\lib`)

### セットアップ手順 (PowerShell)

1. **ディレクトリ作成**
   ```powershell
   mkdir -p "$env:APPDATA\hoshikage\lib"
   ```

2. **ライブラリ配置**
   ビルドした `llama.dll` (または `libllama.dll`) を上記フォルダにコピーします。

3. **環境変数 (PATH) の設定**
   コマンド実行時に読み込まれるよう、ユーザー環境変数の `Path` に追加します。

   ```powershell
   # 現在のセッションに追加
   $env:PATH += ";$env:APPDATA\hoshikage\lib"

   # 永続的に追加 (推奨)
   [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\hoshikage\lib", [System.EnvironmentVariableTarget]::User)
   ```

4. **インストールと実行**
   ```powershell
   cargo install --path .
   hoshikage
   ```
