# 星影 (Hoshikage) - llama.cpp runtime 運用ガイド

**作成日:** 2026-01-18  
**更新日:** 2026-07-23  
**プロジェクト:** 星影 (Hoshikage)

---

## 1. 基本方針

Hoshikage は llama.cpp 本体、ビルド済みバイナリ、GPU backend 用 shared library を配布しません。

利用者は、自分の OS と GPU に合う llama.cpp runtime を公式配布物または source build で用意し、Hoshikage の標準 runtime directory に配置します。

```text
~/.config/hoshikage/llama.cpp
```

Hoshikage はこの directory にある `llama-server` を managed runtime として起動・監視します。FFI 互換経路を使う場合も、同じ directory の `libllama` などの shared library を参照します。

初心者向けの OS 別導入手順は [llama.cpp Installation Guide for Hoshikage](llama-cpp-install-guide.md) を参照してください。

---

## 2. 必要なファイル

Linux:

```text
~/.config/hoshikage/llama.cpp/llama-server
~/.config/hoshikage/llama.cpp/llama-cli
~/.config/hoshikage/llama.cpp/libllama.so
~/.config/hoshikage/llama.cpp/libggml*.so
~/.config/hoshikage/llama.cpp/libmtmd.so
```

macOS:

```text
~/Library/Application Support/hoshikage/llama.cpp/llama-server
~/Library/Application Support/hoshikage/llama.cpp/llama-cli
~/Library/Application Support/hoshikage/llama.cpp/libllama.dylib
~/Library/Application Support/hoshikage/llama.cpp/libggml*.dylib
~/Library/Application Support/hoshikage/llama.cpp/libmtmd.dylib
```

Windows:

```text
%APPDATA%\hoshikage\llama.cpp\llama-server.exe
%APPDATA%\hoshikage\llama.cpp\llama-cli.exe
%APPDATA%\hoshikage\llama.cpp\llama.dll
%APPDATA%\hoshikage\llama.cpp\ggml*.dll
%APPDATA%\hoshikage\llama.cpp\mtmd.dll
```

---

## 3. 動作確認

runtime を配置したら、まず `llama-server` が起動できるか確認します。

Linux:

```bash
~/.config/hoshikage/llama.cpp/llama-server --version
```

macOS:

```bash
"$HOME/Library/Application Support/hoshikage/llama.cpp/llama-server" --version
```

Windows PowerShell:

```powershell
& "$env:APPDATA\hoshikage\llama.cpp\llama-server.exe" --version
```

Hoshikage 側の診断:

```bash
hoshikage doctor
```

モデル登録後の診断:

```bash
hoshikage doctor --model <model-label>
```

---

## 4. 探索順序

標準では、Hoshikage は次の runtime directory を使います。

```text
~/.config/hoshikage/llama.cpp
```

別の場所に置きたい場合は、環境変数で明示します。

```bash
HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR=/path/to/llama.cpp-runtime
```

Linux で loader が shared library を見つけられない場合は、同じ directory を `LD_LIBRARY_PATH` に追加します。

```bash
export LD_LIBRARY_PATH="$HOME/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH"
```

---

## 5. 開発者向けメモ

`llama_cpp_local/` はローカル検証用の作業 directory として使えますが、Git 管理対象にはしません。

Rust の FFI binding は通常、リポジトリに含まれる `src/ffi.rs` を使います。llama.cpp header から binding を再生成したい場合は、header directory を用意してから build します。

```bash
HOSHIKAGE_LLAMA_CPP_INCLUDE_DIR=/path/to/llama.cpp/include cargo build
```

`llama.cpp` の更新や CUDA build の具体手順は [llama.cpp Installation Guide for Hoshikage](llama-cpp-install-guide.md) を参照してください。
