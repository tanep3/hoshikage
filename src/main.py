import os
import asyncio
import time
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from llama_cpp import Llama
import mount as mt
from uuid import uuid4
from models.schema import ChatCompletionRequest
import json
import gc
import re
import chromadb
from chroma_embedding_function import ChromaEmbeddingFunction
from select_sentence_representatives import select_sentence_representatives, split_and_clean_sentences
from fastapi.exceptions import RequestValidationError
from typing import Optional, Dict, Any

VERSION = "0.1.0"

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# .env 読み込み（プロジェクトルートから）
import pathlib
project_root = pathlib.Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

def get_env_path(key, default_rel_path):
    val = os.getenv(key)
    if val:
        if val.startswith("./"):
            return str(project_root / val[2:])
        return val
    # デフォルトはプロジェクトルート直下の data ディレクトリ
    return str(project_root / default_rel_path)

# 環境変数から設定を読み込み
RAMDISK_PATH = os.getenv("RAMDISK_PATH", "/mnt/temp/hoshikage")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))
GREAT_TIMEOUT = int(os.getenv("GREAT_TIMEOUT", "60")) * 60
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "8"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
MAX_TOKENS_STREAMING = int(os.getenv("MAX_TOKENS_STREAMING", "2096"))
MAX_TOKENS_NON_STREAMING = int(os.getenv("MAX_TOKENS_NON_STREAMING", "1024"))
SEMAPHORE_TIMEOUT = int(os.getenv("SEMAPHORE_TIMEOUT", "180"))

# クラスタリング設定
CLUSTER_DIVISOR = int(os.getenv("CLUSTER_DIVISOR", "100"))
MIN_CLUSTERS = int(os.getenv("MIN_CLUSTERS", "1"))
MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS", "20"))

# ルートの data ディレクトリを基準にする
MODEL_MAP_FILE = get_env_path("MODEL_MAP_FILE", "data/model_map.json")
TAG_CACHE_FILE = get_env_path("TAG_CACHE_FILE", "data/tags_cache.json")

# JSONの存在確認（なければ作成）
for fpath in [MODEL_MAP_FILE, TAG_CACHE_FILE]:
    p = pathlib.Path(fpath)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            if "tags" in p.name:
                json.dump({"data": []} if "cache" in p.name else {"models": []}, f)
            else:
                json.dump({}, f)

# ChromaDBの初期化
CHROMA_PATH = get_env_path("CHROMA_PATH", "data/hoshikage_chroma_db")
SENTENCE_BERT_MODEL = os.getenv("SENTENCE_BERT_MODEL", "cl-nagoya/ruri-small-v2")
CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
EMBEDDING_FUNCTION = ChromaEmbeddingFunction(model_name=SENTENCE_BERT_MODEL)
CHROMA_SHORT_MEMORY_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="short_memory_db",
    embedding_function=EMBEDDING_FUNCTION
)

# モデル管理クラス
class ModelManager:
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.llm_lock = asyncio.Lock()
        self.concurrency_semaphore = asyncio.Semaphore(1)
        self.last_access_time = time.time()
        self.current_model = ""
        self.current_model_config: Dict[str, Any] = {}
        self.is_processing = False

    async def initialize_model(self, model_alias: str) -> None:
        """モデルの初期化・切り替え"""
        async with self.llm_lock:
            if model_alias == self.current_model and self.llm is not None:
                return
            
            self.current_model = model_alias
            
            if self.llm is not None:
                self.llm.close()
                self.llm = None
                gc.collect()
            
            try:
                ram_model_path, config = mt.get_model(self.current_model)
                self.current_model_config = config
                
                self.llm = Llama(
                    model_path=ram_model_path, 
                    n_ctx=N_CTX,      # コンテキスト長
                    n_threads=N_THREADS,     # スレッド数
                    n_gpu_layers=N_GPU_LAYERS, # GPUをMaxまで使う
                    n_batch=N_BATCH,     # バッチサイズ
                    use_mmap=True,   # モデルファイルを RAM や VRAM に全て読み込む代わりに、ファイルシステムから直接メモリにマッピングして利用しようとします。
                    verbose=False    # 出力を抑制
                )
                logger.info(f"✅ モデル '{model_alias}' を初期化しました")
            except Exception as e:
                logger.error(f"モデルの初期化に失敗しました: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "code": "model_load_failed",
                            "message": f"モデルのロードに失敗しました: {str(e)}",
                            "type": "internal_server_error"
                        }
                    }
                )

    async def check_idle_timeout(self) -> None:
        """非アクティブ時のモデルアンロード"""
        if time.time() - self.last_access_time > IDLE_TIMEOUT:
            async with self.llm_lock:
                if self.llm:
                    logger.info("⏳ 非アクティブ時間超過のためモデルをアンロードします")
                    self.llm.close()  # リソースを解放
                    self.llm = None  # ガベージコレクタが解放しやすくする
                    gc.collect()
        if time.time() - self.last_access_time > GREAT_TIMEOUT:
            try:
                mt.unmount_ramdisk(RAMDISK_PATH)
                logger.info("✅ RAMディスクをアンマウントしました")
            except Exception as e:
                logger.error(f"RAMディスクのアンマウントに失敗しました: {e}")

    async def acquire_processing_lock(self) -> None:
        """処理ロックの取得（タイムアウト付き）"""
        sleep_count = 0
        while self.is_processing:
            if sleep_count > SEMAPHORE_TIMEOUT * 10:  # タイムアウト
                self.is_processing = False
                if self.llm:
                    self.llm.close()
                    self.llm = None
                gc.collect()
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "code": "processing_timeout",
                            "message": "処理タイムアウトが発生しました",
                            "type": "internal_server_error"
                        }
                    }
                )
            await asyncio.sleep(0.1)
            sleep_count += 1
        self.is_processing = True

    def release_processing_lock(self) -> None:
        """処理ロックの解放"""
        self.is_processing = False

# モデルマネージャーのインスタンス化
model_manager = ModelManager()

@app.get("/v1/status")
async def status():
    return {"status": "ok"}

@app.get("/v1/models")
async def get_model_tags():
    try:
        if not os.path.exists(TAG_CACHE_FILE):
            raise FileNotFoundError(f"モデルキャッシュファイルが見つかりません: {TAG_CACHE_FILE}")
        with open(TAG_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"モデルキャッシュファイルが見つかりません: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "tags_file_not_found",
                    "message": str(e),
                    "type": "not_found"
                }
            }
        )
    except Exception as e:
        logger.error(f"/api/tags エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "tags_fetch_failed",
                    "message": str(e),
                    "type": "internal_server_error"
                }
            }
        )

@app.get("/v1/api/version")
async def get_version():
    return {"version": VERSION}

# ストリーミングジェネレータ定義
def stream_generator(current_model, prompt, session_id):
    try:
        partial_text = ""
        # 設定からstopシーケンスを取得
        stop_tokens = model_manager.current_model_config.get("stop", ["<|im_end|>", "</s>"])
        
        for chunk in model_manager.llm(
            prompt,
            max_tokens=MAX_TOKENS_STREAMING,
            stop=stop_tokens,
            stream=True
        ):
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if not delta:
                continue
            partial_text += delta
            # OpenAI互換フォーマット
            payload = {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": current_model,
                "choices": [{
                    "delta": {"content": delta},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(payload)}\n\n"
        # ストリームの終了通知
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

# 非ストリーミング用
async def non_streaming_generator(current_model, prompt, session_id):
    # 設定からstopシーケンスを取得
    stop_tokens = model_manager.current_model_config.get("stop", ["<|im_end|>", "</s>"])

    output = model_manager.llm(
        prompt, 
        max_tokens=MAX_TOKENS_NON_STREAMING, 
        stop=stop_tokens,
    )
    assistant_message = output["choices"][0]["text"]

    usage = output.get("usage", {})
    response = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": current_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_message
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    }

    return response

def save_chroma(role, message, is_compressed):
    if not message:
        return
    try:
        if read_chroma(message):
            # 既に存在する場合は保存しない
            return
        CHROMA_SHORT_MEMORY_COLLECTION.add(
            documents=[message],
            metadatas=[{
                "role": role,
                "create_date": datetime.now().strftime("%Y%m%d"),
                "create_time": datetime.now().strftime("%H:%M:%S"),
                "compressed": is_compressed
            }],
            ids=[uuid4().hex]
        )
    except Exception as e:
        logger.error(f"ChromaDBの保存エラー: {e}")

def read_chroma(message):
    if not message:
        return None
    try:
        query_results = CHROMA_SHORT_MEMORY_COLLECTION.query(
            query_texts=[message],
            n_results=1
        )
        logger.debug(f"Chroma検索クエリ: {message}")
        distances = query_results.get("distances", [[]])
        if distances and distances[0]:
            if distances[0][0]:
                if distances[0][0] >= 0.15:
                    logger.debug("ヒットしませんでした。")
                    logger.debug(f"距離: {distances[0][0]}")
                    logger.debug(query_results["documents"][0][0])
                    return None
        logger.debug(f"Chroma検索結果: {query_results}")
        return query_results["documents"][0][0] if query_results["documents"] else None
    except Exception as e:
        logger.error(f"ChromaDBのクエリエラー: {e}")
        return None

def message_compress(role: str, message: str) -> str:
    if "```python" in message or role == "system":
        return message  # コードやシステム指示はそのまま保持
    if len(message) <= 150:
        return message  # 150文字以下は無圧縮
    compress_prompt = f"system: 次の内容を150文字程度に親しみやすい口調で要約してください。文章の主たる意図を失わないように要約してください。語調や言い回しなど会話表現的なニュアンスが有る場合は、それを維持するよう心がけて下さい。回答は要約のみを出力し、他の情報は付与しないこと。Markdown表記は平文に直して下さい。制御コードは削除して、文字数削減に務めること。\nuser: {message}\nassistant: "
    output = model_manager.llm(
        compress_prompt, 
        max_tokens=256, 
        stop=["<|eot|>", "user:"],
    )
    # 文末のゴミを除去してから返す
    result = re.sub(r"(```+|[\[\(\{]*$)", "", output["choices"][0]["text"]).strip()
    return result

@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    logger.error(f"リクエストバリデーションエラー: {exc}")
    return JSONResponse(
        content={
            "error": {
                "code": "validation_error",
                "message": "リクエストのバリデーションに失敗しました",
                "type": "invalid_request",
                "details": exc.errors()
            }
        },
        status_code=422
    )

@app.post("/v1/chat/completions")
async def create_completion(completion_data: ChatCompletionRequest):
    async with model_manager.concurrency_semaphore:
        try:
            # 処理ロックの取得
            await model_manager.acquire_processing_lock()
            
            model_alias = completion_data.model
            await model_manager.initialize_model(model_alias)
            model_manager.last_access_time = time.time()
            
            messages = completion_data.messages
            session_id = messages[0].session_id if hasattr(messages[0], "session_id") else "default_session"
            
            prompt = ""
            system_prompt = ""
            user_prompt = ""
            all_histories = ""
            prompt_raw = ""
            raw_talks = 3
            talks_count = 0
            # 最後のraw_talks往復は「要約対象から除外」して原文のまま連結することで文脈を保持
            for msg in messages[::-1]:
                if msg.role == "system":
                    system_prompt += msg.content + "\n"
                    continue
                if user_prompt == "":
                    if msg.role == "user":
                        user_prompt = msg.content
                        continue
                if talks_count < raw_talks:
                    # 直近のraw_talks往復はそのまま連結
                    prompt_raw += f"{msg.role}: {msg.content}\n"
                    if msg.role == "user":
                        talks_count += 1
                    continue
                all_histories += msg.content + "\n"
            if all_histories:
                prompt = select_sentence_representatives(
                    split_and_clean_sentences(all_histories),
                    EMBEDDING_FUNCTION,
                    cluster_divisor=CLUSTER_DIVISOR,
                    min_clusters=MIN_CLUSTERS,
                    max_clusters=MAX_CLUSTERS
                )
            if prompt:
                prompt = f"## **(参考情報)会話のダイジェスト** \n{prompt}\n" 
            if prompt_raw:
                prompt += f"## **(参考情報)直近の会話履歴** \n{prompt_raw}\n"
            if prompt:
                prompt += "\n## **本題（以下に会話を続けて下さい。）** \n"
            if user_prompt:
                prompt += f"user: {user_prompt}\n"
            if system_prompt:
                prompt = f"system: {system_prompt}\n" + prompt
            prompt += "\nassistant: "
            logger.debug(f"プロンプト: {prompt}")
            
            if completion_data.stream:
                # ストリーミングの場合
                return StreamingResponse(stream_generator(model_alias, prompt, session_id), media_type="text/event-stream")
            else:
                response = await non_streaming_generator(model_alias, prompt, session_id)
                logger.debug(f"レスポンス: {response}")
                return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"チャット補完エラー: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": "inference_failed",
                        "message": f"推論に失敗しました: {str(e)}",
                        "type": "internal_server_error"
                    }
                }
            )
        finally:
            # 処理ロックの解放
            model_manager.release_processing_lock()
        
async def background_cleanup():
    while True:
        await asyncio.sleep(30)  # 30秒ごとにチェック
        await model_manager.check_idle_timeout()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())
