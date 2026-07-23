# llama.cpp Installation Guide for Hoshikage

**作成日:** 2026-07-21  
**位置づけ:** 調査メモ兼導入ガイド  
**対象:** Linux / macOS / Windows

---

## 1. 概要

Hoshikage の今後の改訂では、MTP、Draft model、Vision など llama.cpp の新しい機能に追従する必要がある。

従来はユーザーが llama.cpp を自力でビルドして `libllama` を配置する運用だったが、現在は GitHub Releases から shared library 入りの prebuilt archive を取得できる。

- Linux CUDA では source build で `llama-server` と shared library 一式を作成する
- GitHub Releases から platform 別 prebuilt archive を取得する
- `llama.app` installer で prebuilt `llama` バイナリを導入する
- Homebrew / winget / conda-forge / nix などの package manager を使う
- 従来通り source build する

Hoshikage の標準 runtime は managed `llama-server` である。標準手順では、llama.cpp GitHub Releases の archive または source build の成果物から、`llama-server`、`llama-cli`、`libllama`、依存 shared library 一式を Hoshikage の runtime directory に配置する。

標準 runtime directory:

```text
~/.config/hoshikage/llama.cpp
```

Linux CUDA については、GitHub Releases で CUDA 用 Ubuntu archive が常に提供されるとは限らない。そのため、NVIDIA CUDA で使う場合は source build を正式な導入手順とする。

同 release の公式 prebuilt archive には、次の shared library が含まれることを確認済みである。

- Linux: `libllama.so`, `libllama.so.0`, `libggml*.so`, `libmtmd.so`
- macOS: `libllama.dylib`, `libllama.0.dylib`, `libggml*.dylib`, `libmtmd.dylib`
- Windows: `llama.dll`, `ggml*.dll`, `mtmd.dll`

---

## 2. 標準手順: GitHub Releases から runtime を入手する

llama.cpp Releases には、OS / architecture / GPU backend 別の archive が提供される。

最新 release は次で確認する。

```bash
curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest
```

ブラウザから確認する場合:

```text
https://github.com/ggml-org/llama.cpp/releases/latest
```

### 2.1 Archive 選択基準

| 環境 | 推奨 archive pattern | 備考 |
|------|----------------------|------|
| Linux x64 CPU | `llama-b*-bin-ubuntu-x64.tar.gz` | CPU 動作確認用 |
| Linux x64 NVIDIA CUDA | source build with `-DGGML_CUDA=ON` | CUDA archive が提供されていない release では source build |
| Linux x64 NVIDIA Vulkan | `llama-b*-bin-ubuntu-vulkan-x64.tar.gz` | CUDA を使わない場合の prebuilt 候補 |
| Linux x64 AMD | `llama-b*-bin-ubuntu-rocm-*-x64.tar.gz` | ROCm version に注意 |
| Linux x64 汎用 GPU | `llama-b*-bin-ubuntu-vulkan-x64.tar.gz` | NVIDIA / AMD / Intel で利用候補 |
| macOS Apple Silicon | `llama-b*-bin-macos-arm64.tar.gz` | Metal 対応 |
| macOS Intel | `llama-b*-bin-macos-x64.tar.gz` | CPU 中心 |
| Windows x64 CPU | `llama-b*-bin-win-cpu-x64.zip` | CPU 動作確認用 |
| Windows x64 NVIDIA | `llama-b*-bin-win-cuda-*-x64.zip` + `cudart-llama-bin-win-cuda-*-x64.zip` | CUDA DLL archive も同じ release から取得 |
| Windows x64 汎用 GPU | `llama-b*-bin-win-vulkan-x64.zip` | NVIDIA / AMD / Intel で利用候補 |

### 2.2 配置するファイル

Hoshikage では、`llama-server` と依存 shared library を同じ directory に置く。

Linux:

```text
llama-server
llama-cli
libllama.so
libllama.so.0
libggml.so
libggml-base.so
libggml-cpu*.so
libmtmd.so
backend-specific .so files
```

macOS:

```text
llama-server
llama-cli
libllama.dylib
libllama.0.dylib
libggml*.dylib
libmtmd.dylib
backend-specific .dylib files
```

Windows:

```text
llama-server.exe
llama-cli.exe
llama.dll
ggml.dll
ggml-base.dll
ggml-cpu*.dll
mtmd.dll
backend-specific .dll files
```

FFI 互換経路を使う場合も同じ directory の shared library を参照する。

---

## 3. llama.app installer

公式の installer repository は `ggml-org/llama-install.sh` である。

この installer は OS、CPU architecture、GPU backend を自動検出し、prebuilt `llama` 単一バイナリをダウンロードして配置する。

### 対応範囲

公式 README では次が示されている。

- Architecture: `x86_64`, `aarch64`
- OS: Linux, macOS, FreeBSD, Windows
- GPU backend: CUDA, ROCm, Vulkan, Metal

### 注意点

- POSIX 系では `~/.llama-app/llama` に実体を置き、`~/.local/bin/llama` にコピーする。
- `~/.local/bin` が `PATH` にない場合は、shell profile への追加が必要である。
- Windows では PowerShell 用の `install.ps1` を使う。
- installer は CLI バイナリ導入を主目的とする。Hoshikage の FFI に必要な `libllama` 入手には、GitHub Releases の prebuilt archive を使う。

---

## 4. Linux CUDA

### 4.1 推奨: CUDA easy build

NVIDIA CUDA で Hoshikage を使う場合は、この手順を標準とする。

事前条件:

- NVIDIA driver が入っている
- CUDA Toolkit が入っている
- `cmake`, `git`, C/C++ compiler が入っている

Ubuntu 例:

```bash
sudo apt update
sudo apt install -y git cmake build-essential
```

CUDA Toolkit は NVIDIA 公式手順または OS package manager で導入する。

確認:

```bash
nvidia-smi
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

build:

```bash
cd /tmp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# 必要に応じて利用したい release tag に固定する。
# 例: git checkout b10091
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON
cmake --build build --config Release -j "$(nproc)"
```

Hoshikage 用 runtime directory に server、CLI、shared library 一式を配置する。

```bash
mkdir -p ~/.config/hoshikage/llama.cpp
cp -a build/bin/llama-server ~/.config/hoshikage/llama.cpp/
cp -a build/bin/llama-cli ~/.config/hoshikage/llama.cpp/
find build -type f \( \
  -name 'libllama.so*' -o \
  -name 'libggml*.so*' -o \
  -name 'libmtmd.so*' \
\) -exec cp -a {} ~/.config/hoshikage/llama.cpp/ \;
```

確認:

```bash
~/.config/hoshikage/llama.cpp/llama-server --version
ls ~/.config/hoshikage/llama.cpp/libllama.so
ls ~/.config/hoshikage/llama.cpp/libggml-cuda.so
ldd ~/.config/hoshikage/llama.cpp/llama-server
```

library path:

```bash
echo 'export LD_LIBRARY_PATH="$HOME/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 4.2 CUDA architecture を明示する場合

`nvcc` が GPU を検出できない場合や、build 対象を固定したい場合は `CMAKE_CUDA_ARCHITECTURES` を指定する。

例:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"
cmake --build build --config Release -j "$(nproc)"
```

代表例:

| GPU | Compute Capability |
|-----|--------------------|
| RTX 30xx | 86 |
| RTX 40xx | 89 |
| RTX 50xx | 120 |

複数 GPU 世代で使い回す場合は、公式 build document の通り `-DGGML_NATIVE=OFF` も選択肢に入る。

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF
cmake --build build --config Release -j "$(nproc)"
```

### 4.3 CUDA runtime path

CUDA を標準以外の場所に入れている場合は、`CMAKE_CUDA_COMPILER` と install rpath を指定する。

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_INSTALL_RPATH="/usr/local/cuda/lib64;\$ORIGIN" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build build --config Release -j "$(nproc)"
```

### 4.4 CLI smoke test

Hoshikage FFI とは別に、llama.cpp 側の CUDA build が動くかを `llama-cli` または `llama-server` で確認できる。

```bash
./build/bin/llama-cli --version
./build/bin/llama-server --version
```

モデルを持っている場合:

```bash
./build/bin/llama-cli -m /path/to/model.gguf -ngl 99 -p "Hello"
```

---

## 5. Linux prebuilt archive

### 5.1 GitHub Releases archive

```bash
RELEASE_TAG=b10091
mkdir -p ~/.config/hoshikage/llama.cpp
curl -L -o /tmp/llama-bin.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/${RELEASE_TAG}/llama-${RELEASE_TAG}-bin-ubuntu-x64.tar.gz"
tar -xzf /tmp/llama-bin.tar.gz -C /tmp
cp /tmp/llama-${RELEASE_TAG}/llama-server ~/.config/hoshikage/llama.cpp/
cp /tmp/llama-${RELEASE_TAG}/llama-cli ~/.config/hoshikage/llama.cpp/
cp /tmp/llama-${RELEASE_TAG}/*.so* ~/.config/hoshikage/llama.cpp/
```

GPU backend を使う場合は、CPU archive の代わりに同じ release の Vulkan / ROCm / SYCL 対応 archive を選ぶ。CUDA で使う場合は、前章の CUDA easy build を使う。

### 5.2 library path

実行時に loader が見つけられるようにする。

```bash
echo 'export LD_LIBRARY_PATH="$HOME/.config/hoshikage/llama.cpp:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 5.3 確認

最低限、次のファイルが存在することを確認する。

```bash
~/.config/hoshikage/llama.cpp/llama-server --version
ls ~/.config/hoshikage/llama.cpp/libllama.so
ls ~/.config/hoshikage/llama.cpp/libggml.so
ls ~/.config/hoshikage/llama.cpp/libggml-base.so
```

`llama` CLI も入れたい場合は、別途 installer を使える。

```bash
curl https://llama.app/install.sh | sh
```

---

## 6. macOS

### 6.1 推奨: GitHub Releases archive

Apple Silicon:

```bash
RELEASE_TAG=b10091
mkdir -p "$HOME/Library/Application Support/hoshikage/llama.cpp"
curl -L -o /tmp/llama-bin.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/${RELEASE_TAG}/llama-${RELEASE_TAG}-bin-macos-arm64.tar.gz"
tar -xzf /tmp/llama-bin.tar.gz -C /tmp
cp /tmp/llama-${RELEASE_TAG}/llama-server "$HOME/Library/Application Support/hoshikage/llama.cpp/"
cp /tmp/llama-${RELEASE_TAG}/llama-cli "$HOME/Library/Application Support/hoshikage/llama.cpp/"
cp /tmp/llama-${RELEASE_TAG}/*.dylib "$HOME/Library/Application Support/hoshikage/llama.cpp/"
```

Intel Mac では `llama-${RELEASE_TAG}-bin-macos-x64.tar.gz` を使う。

### 6.2 確認

```bash
"$HOME/Library/Application Support/hoshikage/llama.cpp/llama-server" --version
ls "$HOME/Library/Application Support/hoshikage/llama.cpp/libllama.dylib"
ls "$HOME/Library/Application Support/hoshikage/llama.cpp/libggml.dylib"
ls "$HOME/Library/Application Support/hoshikage/llama.cpp/libggml-base.dylib"
```

macOS の loader 設定は実装時に Hoshikage 側の探索順序として明確化する。手動確認では次を使う。

```bash
export DYLD_LIBRARY_PATH="$HOME/Library/Application Support/hoshikage/llama.cpp:$DYLD_LIBRARY_PATH"
```

### 6.3 llama.app installer

CLI 動作確認用には installer を使える。

```bash
curl https://llama.app/install.sh | sh
```

---

## 7. Windows

### 7.1 推奨: GitHub Releases archive

PowerShell:

```powershell
$ReleaseTag = "b10091"
mkdir "$env:APPDATA\hoshikage\llama.cpp" -Force
Invoke-WebRequest `
  -Uri "https://github.com/ggml-org/llama.cpp/releases/download/$ReleaseTag/llama-$ReleaseTag-bin-win-cpu-x64.zip" `
  -OutFile "$env:TEMP\llama-bin.zip"
Expand-Archive "$env:TEMP\llama-bin.zip" "$env:TEMP\llama-bin" -Force
copy "$env:TEMP\llama-bin\*.exe" "$env:APPDATA\hoshikage\llama.cpp\"
copy "$env:TEMP\llama-bin\*.dll" "$env:APPDATA\hoshikage\llama.cpp\"
```

NVIDIA CUDA では、同じ release から次の両方を取得して同じ directory に展開する。

- `llama-$ReleaseTag-bin-win-cuda-12.4-x64.zip` または `llama-$ReleaseTag-bin-win-cuda-13.3-x64.zip`
- `cudart-llama-bin-win-cuda-12.4-x64.zip` または `cudart-llama-bin-win-cuda-13.3-x64.zip`

### 7.2 確認

```powershell
& "$env:APPDATA\hoshikage\llama.cpp\llama-server.exe" --version
dir "$env:APPDATA\hoshikage\llama.cpp\llama.dll"
dir "$env:APPDATA\hoshikage\llama.cpp\ggml.dll"
dir "$env:APPDATA\hoshikage\llama.cpp\ggml-base.dll"
```

### 7.3 PATH

```powershell
$env:Path += ";$env:APPDATA\hoshikage\llama.cpp"
[System.Environment]::SetEnvironmentVariable(
  "Path",
  $env:Path + ";$env:APPDATA\hoshikage\llama.cpp",
  [System.EnvironmentVariableTarget]::User
)
```

### 7.4 PowerShell installer

CLI 動作確認用には installer を使える。

```powershell
irm https://llama.app/install.ps1 | iex
```

---

## 8. Vision / MTP / Draft model の動作確認

### 8.1 Vision

llama.cpp の multimodal document では、`llama-cli`、`llama-server`、`llama-mtmd-cli` が multimodal input に対応するとされている。

Hugging Face repo 指定で対応モデルを読む場合:

```bash
llama-server -hf ggml-org/gemma-3-4b-it-GGUF
```

ローカルファイルで projector を指定する場合:

```bash
llama-server -m model.gguf --mmproj mmproj-model.gguf
```

projector の GPU offload を避ける場合:

```bash
llama-server -m model.gguf --mmproj mmproj-model.gguf --no-mmproj-offload
```

### 8.2 MTP

llama.cpp の speculative decoding document では、MTP は `draft-mtp` として指定される。

```bash
llama-server -m model.gguf --spec-type draft-mtp
```

実際に有効化できるかは、対象モデル、GGUF メタデータ、llama.cpp の version に依存する。

### 8.3 Draft model

別の draft model を使う場合:

```bash
llama-server -m target.gguf -md draft.gguf --spec-type draft-simple
```

Hugging Face repo から draft model を指定する場合:

```bash
llama-server -m target.gguf --spec-draft-hf user/model:quant --spec-type draft-simple
```

---

## 9. Hoshikage での採用方針

採用方針:

- Linux CUDA では source build で `llama-server`、`llama-cli`、`libllama.so`、依存 shared library 一式を作成・配置する。
- Linux CPU/Vulkan/ROCm/SYCL、macOS、Windows では GitHub Releases の prebuilt archive から `llama-server`、`llama-cli`、`libllama`、依存 shared library 一式を配置する。
- llama.cpp の CLI / server 動作確認には `llama.app` installer も併用できる。
- 自力ビルドは、Linux CUDA では標準手順、その他では必要な backend の prebuilt archive が存在しない場合、または Hoshikage が要求する C API symbol と prebuilt release が合わない場合の fallback とする。
- Hoshikage の主 runtime は managed `llama-server` とする。FFI 経路は既存互換用に残す。
- Vision / MTP / Draft model は `llama-server` / llama.cpp runtime の機能に委譲し、Hoshikage は bundle 管理、診断、起動 option 組み立て、OpenAI 互換 API を担当する。

---

## 10. 参考資料

- llama.cpp README: https://github.com/ggml-org/llama.cpp
- llama.cpp build document: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- NVIDIA CUDA GPU Compute Capability: https://developer.nvidia.com/cuda/gpus
- llama.cpp Releases: https://github.com/ggml-org/llama.cpp/releases
- llama.app installer repository: https://github.com/ggml-org/llama-install.sh
- llama.app install.sh: https://llama.app/install.sh
- llama.app install.ps1: https://llama.app/install.ps1
- llama.cpp speculative decoding: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- llama.cpp multimodal: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
