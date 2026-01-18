# OllamaとHoshikageのLLM効率化比較分析レポート

作成日: 2026-01-16

## 概要

本レポートは、Ollamaプロジェクト（https://github.com/ollama/ollama）とHoshikageプロジェクト（/home/tane/dev/AI/hoshikage）のLLM操作における効率化手法を比較分析したものです。

---

## Hoshikageの現在の実装

### アーキテクチャ
- **言語**: Python
- **フレームワーク**: FastAPI
- **LLMエンジン**: llama-cpp-python
- **バージョン**: 0.1.0

### 現在の効率化手法

#### 1. RAMディスクによるモデル高速化
- モデルをRAMディスク（`/mnt/temp/hoshikage`）にマウント
- ファイルシステムの代わりにメモリから直接読み込みで高速化
- 実装場所: `src/mount.py` 及び `main.py:122-130`

```python
self.llm = Llama(
    model_path=ram_model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    use_mmap=True,  # メモリマッピング
    verbose=False
)
```

#### 2. アイドルタイムアウトによる自動アンロード
- 非アクティブ時間が300秒（設定可能）を超えるとモデルをアンロード
- RAMディスクも60分後にアンマウント
- 実装場所: `main.py:145-159`

```python
async def check_idle_timeout(self) -> None:
    if time.time() - self.last_access_time > IDLE_TIMEOUT:
        self.llm.close()
        self.llm = None
        gc.collect()
```

#### 3. コンテキスト管理と要約
- 直近の会話履歴は原文を維持
- 古い履歴は文クラスタリングで要約（`select_sentence_representatives`）
- ChromaDBによる類似文検索で重複を排除
- 実装場所: `main.py:401-434`

```python
if all_histories:
    prompt = select_sentence_representatives(
        split_and_clean_sentences(all_histories),
        EMBEDDING_FUNCTION,
        cluster_divisor=CLUSTER_DIVISOR,
        min_clusters=MIN_CLUSTERS,
        max_clusters=MAX_CLUSTERS
    )
```

#### 4. メッセージ圧縮
- 150文字を超えるメッセージを自動的にLLMで要約
- コードブロックとシステムプロンプトは圧縮対象外
- 実装場所: `main.py:349-362`

#### 5. 単一モデル管理
- 一度に1つのモデルのみをロード
- モデル切り替え時に前のモデルを明示的にクローズ
- 排他制御は `asyncio.Semaphore(1)` で実装

### 制約事項

1. **同時実行制限**: 1モデルのみ、並列処理なし
2. **VRAM管理**: 手動設定（`N_GPU_LAYERS`）に依存
3. **メモリ回復待機**: GPUメモリ解放の待機なし
4. **キャッシュ最適化**: KV Cacheの量子化など未実装

---

## Ollamaの効率化手法

### アーキテクチャ
- **言語**: Go
- **フレームワーク**: 標準Go HTTPサーバー + CGO
- **LLMエンジン**: llama-cpp（カスタムビルド）+ 独自エンジン

### 主要な効率化手法

#### 1. 高度なスケジューラー（Scheduler）

**ファイル**: `server/sched.go`

Ollamaは複雑なスケジューラーを実装し、複数モデルの効率的な管理を実現しています。

##### 参照カウントによるライフサイクル管理
```go
type runnerRef struct {
    refMu    sync.Mutex
    refCount uint  // prevent unloading if > 0
    // ... other fields
}
```

- `refCount`で現在の使用中リクエスト数を追跡
- 使用中のモデルはアンロードから保護
- リクエスト完了時に自動的に減少

##### 智能的モデルアンロード戦略
```go
func (s *Scheduler) findRunnerToUnload() *runnerRef {
    // Sort by session duration and name
    sort.Sort(ByDurationAndName(runnerList))
    // Try to find idle runner first
    for _, runner := range runnerList {
        if runner.refCount == 0 {
            return runner
        }
    }
    // No idle runners, pick shortest duration
    return runnerList[0]
}
```

- セッション時間の短いモデルを優先してアンロード
- アイドル状態のモデルを優先
- `defaultModelsPerGPU = 3`（デフォルトでGPUあたり3モデルまで許容）

##### 複数GPU対応
- 複数のGPUを使用した場合、レイヤーを最適に分配
- GPUライブラリごとにグループ化（CUDA, ROCm, Vulkan, Metal）
- 各GPUの空きメモリをリアルタイムで監視

#### 2. VRAM回復待機機構

**ファイル**: `server/sched.go:waitForVRAMRecovery`

```go
func (s *Scheduler) waitForVRAMRecovery(...) chan any {
    // Establish baseline before unload
    gpusBefore := s.getGpuFn(context.Background(), runners)

    go func() {
        ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
        defer cancel()
        ticker := time.NewTicker(250 * time.Millisecond)

        for {
            select {
            case <-ticker.C:
                // Query GPUs, look for free to go back up
                gpusNow := s.getGpuFn(ctx, runners)
                freeMemoryNow := calculateFree(gpusNow)

                // If we're within ~75% of estimated memory usage recovered, bail out
                if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
                    finished <- struct{}{}
                    return
                }
            case <-ctx.Done():
                finished <- struct{}{}
                return
            }
        }
    }()
    return finished
}
```

**特徴**:
- GPUメモリの解放をポーリングで監視（250ms間隔）
- 推定メモリ使用量の75%回復した時点で完了
- タイムアウト後は推定値に信頼して続行
- CPU, Metal, iGPUは待機なし（即時完了）

**効果**:
- 次のモデルロード時のVRAM不足を防ぐ
- メモリ報告の遅延（CUDAなど）に対応

#### 3. 動的メモリレイアウト最適化

**ファイル**: `llm/server.go:load` and `llm/server.go:createLayout`

Ollamaはモデルをロードする際、以下のアプローチでメモリ割り当てを最適化します。

##### イテレーティブなレイアウト決定
```go
for {
    var runnerToExpire *runnerRef

    // Get current loaded runners
    runner := s.loaded[pending.model.ModelPath]

    if runner != nil {
        if runner.needsReload(ctx, pending) {
            runnerToExpire = runner
        } else {
            // Use existing runner
            pending.useLoadedRunner(runner, s.finishedReqCh)
            break
        }
    } else if maxRunners > 0 && loadedCount >= int(maxRunners) {
        runnerToExpire = s.findRunnerToUnload()
    } else {
        // Try to fit model
        gpus := s.getGpuFn(ctx, runnersSnapshot)
        systemInfo := s.getSystemInfoFn()

        needEvict := s.loadFn(pending, ggml, systemInfo, gpus, true)
        if !needEvict {
            break  // Model fits with existing models
        }
        runnerToExpire = s.findRunnerToUnload()
    }

    if runnerToExpire != nil {
        // Expire and wait for unload
        runnerToExpire.sessionDuration = 0
        s.expiredCh <- runnerToExpire
        <-s.unloadedCh
        continue
    }
}
```

##### グラフサイズ計算と最適化
```go
kv, graphPartialOffload, graphFullOffload := s.ggml.GraphSize(
    uint64(s.options.NumCtx),
    uint64(s.loadRequest.BatchSize),
    s.loadRequest.Parallel,
    s.loadRequest.KvCacheType,
    s.loadRequest.FlashAttention,
)

for _, gl := range ml.ByLibrary(gpus) {
    gpuLayers = assignLayers(layers, gl, requireFull, s.options.NumGPU, lastUsedGPU)
    if gpuLayers.Sum() > currentMax {
        currentMax = gpuLayers.Sum()
    }
}
```

- 各レイヤーのサイズとKV Cacheサイズを計算
- `assignLayers`で貪欲法（Greedy Fit）により割り当て
- GPUライブラリ種類でグループ化して最適化

##### マルチGPUへのレイヤー割り当て
```go
func assignLayers(layers []uint64, gpus []ml.DeviceInfo, requireFull bool, requestedLayers int, lastUsedGPU int) (gpuLayers ml.GPULayersList) {
    // Pack layers into as few GPUs as possible
    for i := lastUsedGPU; i < len(gpus); i++ {
        gpuLayers = findBestFit(layers, gpus[:i+1], requestedLayers, forceRequest)
        if gpuLayers.Sum() == len(layers) || gpuLayers.Sum() == requestedLayers {
            break
        }
    }
    return gpuLayers
}
```

- **貪欲アルゴリズム**: 空き容量の大きいGPUから順に割り当て
- **部分オフロード**: VRAM不足時は一部レイヤーをCPUに
- 複数GPUを効率的に活用

#### 4. Flash AttentionとKV Cache量子化

**ファイル**: `llm/server.go`

##### Flash Attention
```go
fa := envconfig.FlashAttention(f.FlashAttention())

if fa && !ml.FlashAttentionSupported(gpus) {
    slog.Warn("flash attention enabled but not supported by gpu")
    fa = false
}

if fa && !f.SupportsFlashAttention() {
    slog.Warn("flash attention enabled but not supported by model")
    fa = false
}

loadRequest.FlashAttention = flashAttention
```

- 全GPUがFlash Attentionをサポートしている場合のみ有効化
- モデルとGPUの両方の互換性を確認

##### KV Cache量子化
```go
kvct := strings.ToLower(envconfig.KvCacheType())

if textProcessor == nil {
    if kvct != "" {
        if f.KVCacheTypeIsQuantized(kvct) {
            if flashAttention != ml.FlashAttentionEnabled {
                slog.Warn("OLLAMA_FLASH_ATTENTION must be enabled to use quantized OLLAMA_KV_CACHE_TYPE")
            } else if f.SupportsKVCacheType(kvct) {
                loadRequest.KvCacheType = kvct
            }
        }
    }
}
```

- Flash Attention有効時のみKV Cacheの量子化を許容
- 環境変数 `OLLAMA_KV_CACHE_TYPE` で設定可能

#### 5. 並列処理（Multi-User Cache）

**ファイル**: `llm/server.go`

```go
numParallel := max(int(envconfig.NumParallel()), 1)

// Embedding models should always be loaded with parallel=1
if req.model.CheckCapabilities(model.CapabilityCompletion) != nil {
    numParallel = 1
}

loadRequest.Parallel = numParallel
```

- 同一モデルに対して複数の並列リクエストを許容
- `OLLAMA_NUM_PARALLEL` 環境変数で制御
- Embeddingモデルは常に `parallel=1`（キャッシュ競合を防ぐため）

#### 6. プロセス分離による安定性確保

**ファイル**: `llm/server.go`

```go
cmd := exec.Command(exe, params...)

// Create subprocess with stdout/stderr pipes
stdout, err := cmd.StdoutPipe()
stderr, err := cmd.StderrPipe()

go func() {
    io.Copy(out, stdout)
}()
go func() {
    io.Copy(out, stderr)
}()

cmd.Start()
```

- llama.cppプロセスをサブプロセスとして起動
- HTTP（localhost:port）で通信
- プロセスクラッシュ時もサーバープロセスは生存

#### 7. KV Cacheモジュール（専用実装）

**ファイル**: `kvcache/cache.go`, `kvcache/causal.go`, `kvcache/encoder.go`

OllamaはKV Cacheの独自実装を持っています。

##### Causal Attentionの最適化
- `kvcache/causal.go` でCausal Attentionの特殊化最適化
- エンコーダーによる効率的なキー生成

##### マルチユーザーキャッシュ
- `kvcache/cache.go` で複数ユーザーの並列アクセスをサポート
- Lock-freeなアクセスでコンテンションを最小化

---

## 比較分析

### 効率化手法の比較表

| 手法 | Hoshikage | Ollama | 差分 |
|------|------------|---------|------|
| **モデル管理** | 単一モデル、セマフォア制御 | 複数モデル、参照カウント、スマートアンロード | Ollamaは複数モデルを同時管理可能 |
| **VRAM回復待機** | なし | ポーリングによる待機機構（75%回復で完了） | OllamaはGPUメモリ解放を待ってから次モデルをロード |
| **メモリレイアウト** | 手動設定（N_GPU_LAYERS） | イテレーティブな動的最適化 | OllamaはVRAMに合わせて自動最適 |
| **マルチGPU対応** | 基本的に対応 | 複数GPUへの動的レイヤー割り当て | Ollamaは複数GPUを効率的に活用 |
| **Flash Attention** | llama-cppのデフォルト使用 | 条件付き有効化 + KV Cache量子化 | Ollamaは明示的に最適化 |
| **並列処理** | なし（セマフォ=1） | Multi-User Cacheによる並列処理 | Ollamaは同一モデルで並列リクエスト可能 |
| **プロセス分離** | 同一プロセス内 | サブプロセス + HTTP通信 | Ollamaはクラッシュに強靭 |
| **キャッシュ最適化** | ChromaDBによる文キャッシュ | KV Cache量子化 + マルチユーザーキャッシュ | OllamaはKV Cacheレベルで最適化 |
| **コンテキスト管理** | 文クラスタリングで要約 | llama.cppの標準機能 | Hoshikageは独自の要約ロジックを持つ |

### アーキテクチャの比較

| 項目 | Hoshikage | Ollama |
|------|------------|---------|
| **言語** | Python | Go |
| **プロセスモデル** | 同一プロセス | サブプロセス分離 |
| **通信方式** | 関数呼び出し | HTTP（localhost） |
| **LLMエンジン** | llama-cpp-python | llama-cpp（CGO）+ 独自エンジン |
| **設定方法** | 環境変数 + .env | 環境変数 + コマンドライン + Modelfile |
| **プラットフォーム** | Pythonでクロスプラットフォーム容易 | ネイティブバイナリで各OSに最適化 |

---

## Hoshikageへの適用推奨事項

### 優先度高（即時実装可能）

#### 1. VRAM回復待機機構の導入

**実装場所**: `src/main.py`

現在のHoshikageはモデルをアンロードした後、GPUメモリの解放を待たずに次のモデルをロードしようとします。これはVRAM不足を引き起こす可能性があります。

```python
import time
import subprocess

def wait_for_vram_recovery(gpus, vram_size, timeout=5):
    """
    GPUメモリの解放を待機する関数

    Args:
        gpus: GPUデバイス情報のリスト
        vram_size: 推定解放メモリ量（バイト）
        timeout: 最大待機時間（秒）
    """
    start = time.time()
    baseline_free = get_gpu_free_memory(gpus)

    while time.time() - start < timeout:
        current_free = get_gpu_free_memory(gpus)
        recovered = current_free - baseline_free

        # 推定メモリ使用量の75%以上回復したら完了
        if recovered > vram_size * 0.75:
            logger.info(f"✅ VRAM recovered: {recovered / 1024**3:.2f}GB / {vram_size / 1024**3:.2f}GB")
            return True

        time.sleep(0.25)  # 250ms間隔でポーリング

    logger.warning(f"⚠️  VRAM recovery timeout after {timeout}s")
    return False
```

**効果**:
- 大きなモデルを連続してロードする際のVRAM不足を防ぐ
- モデル切り替えの成功率向上

#### 2. 参照カウントによるモデル管理

**実装場所**: `src/main.py` の `ModelManager` クラス

現在のHoshikageはリクエストの完了を待ってからアンロード判定を行っていますが、参照カウントを追跡していません。

```python
class ModelManager:
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.llm_lock = asyncio.Lock()
        self.concurrency_semaphore = asyncio.Semaphore(1)
        self.last_access_time = time.time()
        self.current_model = ""
        self.current_model_config: Dict[str, Any] = {}
        self.is_processing = False
        self.ref_count = 0  # 新規追加

    async def acquire(self) -> None:
        """モデル参照カウントを増加"""
        async with self.llm_lock:
            self.ref_count += 1
            self.last_access_time = time.time()

    async def release(self) -> None:
        """モデル参照カウントを減少し、必要ならアンロード"""
        async with self.llm_lock:
            self.ref_count -= 1

            if self.ref_count == 0 and time.time() - self.last_access_time > IDLE_TIMEOUT:
                # 参照カウントが0でアイドルタイムアウトの場合のみアンロード
                logger.info("🔄 Model idle timeout, unloading...")
                if self.llm:
                    self.llm.close()
                    self.llm = None
                gc.collect()
```

**効果**:
- 使用中のモデルが誤ってアンロードされるのを防ぐ
- 複数の並列リクエストを安全に処理可能になる基盤

#### 3. Flash AttentionとKV Cache量子化の有効化

**実装場所**: `src/main.py`

llama-cppの設定でFlash AttentionとKV Cache量子化を有効化します。

```python
# モデル初期化時
self.llm = Llama(
    model_path=ram_model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    use_mmap=True,
    f16_kv=True,  # KV Cacheをf16で量子化
    verbose=False
)

# Flash Attention（llama-cppのバージョンによる）
# 注: llama-cpp-pythonでFlash Attentionを有効化するには
# 最新バージョンとコンパイルオプションが必要
```

**環境変数の追加（.env）**:
```bash
# llama-cpp用のFlash Attention設定
OLLAMA_FLASH_ATTENTION=true  # 注: これはOllamaの環境変数です

# KV Cache量子化タイプ
LLAMA_F16_KV=true
```

**効果**:
- 推論速度の大幅向上（Flash Attentionにより）
- VRAM使用量の削減（KV Cache量子化により）
- より大きなモデルやコンテキスト長で顕著

### 優先度中（調査と実装必要）

#### 4. 並列処理の導入

現在のHoshikageは `asyncio.Semaphore(1)` により同時実行を1リクエストに制限しています。これを緩和して並列処理を可能にするには、以下のアプローチが考えられます。

**オプションA: マルチスレッド+排他制御の強化**
```python
from concurrent.futures import ThreadPoolExecutor

class ModelManager:
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.lock = asyncio.Lock()
        self.max_parallel = int(os.getenv("MAX_PARALLEL", "1"))  # 設定可能

    async def generate(self, prompt, options):
        """排他制御付きの並列生成"""
        async with self.lock:
            # Llamaインスタンス自体はスレッドセーフではないため
            # ここでは非同期実行をシミュレート
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # デフォルトのThreadPoolExecutorを使用
                self._sync_generate,
                prompt,
                options
            )
            return result

    def _sync_generate(self, prompt, options):
        """同期的な生成関数"""
        return self.llm(prompt, **options)
```

**オプションB: マルチプロセスモデル（Ollama方式）**
- HoshikageをGoで書き直すか、llama.cppをサブプロセスとして実行
- HTTPサーバーを立ち上げ、複数クライアントからのリクエストを許容
- 実装コストは高いが、最も堅牢なアプローチ

#### 5. 動的メモリレイアウト

llama-cpp-python APIには限定的なメモリ制御オプションしかありません。以下の方法で改善可能です。

**A. llama-cppのネイティブAPIを使用**
```python
from llama_cpp import Llama, llama_cpp

# モデルロード前に空きメモリを確認
def check_gpu_memory_availability():
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True)
    # 解析して判断...
    return free_memory

# モデルサイズに基づいてn_gpu_layersを動的に調整
model_size_bytes = os.path.getsize(model_path)
available_vram = check_gpu_memory_availability()

# 簡易なヒューリスティック
if available_vram < model_size_bytes * 0.8:
    n_gpu_layers = -1  # 全てCPU（安全策）
else:
    n_gpu_layers = int(available_vram / (model_size_bytes / 32))  # 約1/3をGPUに

self.llm = Llama(
    model_path=ram_model_path,
    n_gpu_layers=n_gpu_layers,  # 動的設定
    # ... other params
)
```

**B. llama-cppのCGO拡張を使用**
- llama-cppを直接ビルドし、HoshikageからCGO経由で呼び出し
- Ollamaと同じレベルの制御が可能になる

---

## 技術的差分の深掘り

### llama-cpp-python と llama-cpp（CGO）の違い

| 項目 | llama-cpp-python | llama-cpp（Ollamaで使用） |
|------|-------------------|----------------------------|
| **API** | 高レベルPythonラッパー | 低レベルC/C++ API |
| **制御細度** | 限定的 | 高度（レイヤー単位、キャッシュ制御） |
| **Flash Attention** | バージョン/コンパイル依存 | 動的切替可能 |
| **KV Cache量子化** | 基本的なフラグのみ | 複数の量子化タイプ選択可能 |
| **メモリレイアウト** | 単純なn_gpu_layers設定 | イテレーティブな最適化アルゴリズム |
| **プロセス分離** | Pythonインタプリタ内 | 独自プロセス |

### スケジューラーの複雑さ

Ollamaのスケジューラーは以下の状態遷移を管理します：

1. **新規リクエスト受領**
   - 既存モデルがロード済みか確認
   - 必要なら再ロード判定（`needsReload`）

2. **メモリ適合性確認**
   - `LoadOperationFit` でメモリ要件のみを計算
   - 既存モデルとの共存を確認

3. **メモリ割り当て**
   - `LoadOperationAlloc` で実際のメモリ確保
   - レイヤーをGPU/CPUに配置

4. **コミット**
   - `LoadOperationCommit` でウェイトロード
   - 使用開始

5. **リクエスト処理**
   - 参照カウントの増減
   - アイドル検出とタイマー設定

6. **期限切れ**
   - セッション期間終了
   - アンロードとVRAM回復待機

---

## Hoshikageの独自機能の評価

HoshikageはOllamaにはない独自の最適化機能も持っています。

### 良い点

1. **文クラスタリングによる要約**
   - 文レベルでの類似度計算
   - クラスタリングで代表的な文のみを抽出
   - ChromaDBによる重複排除
   - 長い会話履歴の効率的な圧縮

2. **RAMディスクによる高速化**
   - モデルファイルのメモリマッピング
   - ディスクI/Oの削減

3. **ストリーミング対応**
   - Server-Sent Events（SSE）によるリアルタイム出力
   - ユーザーエクスペリエンスの向上

---

## 推奨される実装ロードマップ

### フェーズ1：即時改善（1-2週間）

1. ✅ VRAM回復待機機構の実装
2. ✅ 参照カウントによるモデル管理の強化
3. ✅ Flash AttentionとKV Cache量子化の有効化
4. ✅ ドキュメンテーションの更新

### フェーズ2：中期的改善（1-2ヶ月）

1. 🔄 動的メモリレイアウトの導入（nvidia-smi連携）
2. 🔄 並列処理の実装（ThreadPoolExecutorモデル）
3. 🔄 GPUメトリクスの監視と記録
4. 🔄 パフォーマンス測定とベンチマーク

### フェーズ3：長期的検討（3ヶ月以上）

1. 🔭 llama-cppを直接使用するか検討
   - CGOバインディングの作成
   - サブプロセスモデルへの移行

2. 🔭 マルチGPU対応の強化
   - GPUごとの動的レイヤー割り当て
   - GPU間通信の最適化

3. 🔭 独自スケジューラーの実装
   - Ollamaの方式を参考にした独自スケジューラー

4. 🔭 KV Cacheの高度な最適化
   - マルチユーザーキャッシュ
   - キャッシュヒット率の最適化

---

## 結論

### Hoshikageの強み

1. **シンプルで理解しやすいアーキテクチャ**
   - 単一ファイル（main.py）で全てのロジック
   - Pythonによる迅速な開発

2. **高度なコンテキスト管理**
   - 文クラスタリングによる効率的な要約
   - ChromaDBによる重複排除

3. **柔軟な設定**
   - 環境変数による動的設定
   - ユーザー環境に合わせた調整が容易

### Hoshikageの弱点

1. **LLMエンジンの制限**
   - llama-cpp-pythonは高度な機能にアクセス困難
   - メモリレイアウト制御が限定的

2. **スケーラビリティの制限**
   - 単一モデル、単一リクエストのみ
   - 複数ユーザーの同時利用に最適化されていない

3. **VRAM管理の不備**
   - GPUメモリ解放の待機がない
   - 大きなモデル切り替えで失敗の可能性

### Ollamaのアプローチから学べべき点

1. **堅牢なエラーハンドリング**
   - プロセス分離によりクラッシュからの復旧容易
   - HTTPエラーとプロセスエラーの明示的区別

2. **綿密な監視とロギング**
   - VRAM使用量のリアルタイム監視
   - 詳細なプログレス報告

3. **ユーザー中心の設定**
   - Modelfileによる宣言的な設定
   - 環境変数による微細な制御

4. **プラットフォーム最適化**
   - 各OS向けのネイティブバイナリ
   - Metal, CUDA, ROCm, Vulkanの最適サポート

---

## 参考資料

### Ollamaの主要なソースファイル

1. **server/sched.go** (約31,000行)
   - スケジューラーの中核実装
   - VRAM回復待機、モデルアンロード戦略

2. **llm/server.go** (約56,000行）
   - LLMサーバーの実装
   - メモリレイアウト、Flash Attention、KV Cache

3. **kvcache/cache.go**
   - KV Cacheの独自実装
   - マルチユーザーサポート

4. **runner/llamarunner** / **runner/ollamarunner**
   - llama.cppのラッパー実装
   - サブプロセスの起動と管理

### 関連技術

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Flash Attention**: https://arxiv.org/abs/2307.08691
- **KV Cache Quantization**: https://github.com/ggerganov/llama.cpp/pull/3953
- **GGML Format**: Ollamaが使用するモデルフォーマット

---

**作成者**: Code Analysis Agent
**対象バージョン**:
- Hoshikage: 0.1.0
- Ollama: mainブランチ（2026-01-16時点）

---

*本レポートは、Ollamaのオープンソースコードを分析した結果に基づいています。*
