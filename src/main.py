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
from models.schema import ChatCompletionRequest, ChatSessionManager
import json
import gc
import re
import chromadb
from chroma_embedding_function import ChromaEmbeddingFunction
from select_sentence_representatives import select_sentence_representatives, split_and_clean_sentences
from fastapi.exceptions import RequestValidationError

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
    # デフォルトはプロジェクトルート直下の models ディレクトリ
    return str(project_root / default_rel_path)

RAMDISK_PATH = os.getenv("RAMDISK_PATH", "/mnt/temp/hoshikage")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))
GREAT_TIMEOUT = int(os.getenv("GREAT_TIMEOUT", "60")) * 60

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

llm = None
llm_lock = asyncio.Lock()
concurrency_semaphore = asyncio.Semaphore(1) # 同時実行数を1に制限
last_access_time = time.time()
# chat_session_manager = ChatSessionManager()
current_model = ""
IS_SEMAPHORE=False

async def initialize_model(model_alias):
    global llm, current_model
    if model_alias == current_model:
        if llm is not None:
            return
    current_model = model_alias
    if llm is not None:
        llm.close()
        llm = None
        gc.collect()
    ram_model_path = mt.get_model(current_model)
    llm = Llama(
        model_path=ram_model_path, 
        # n_ctx=20960,
        # n_ctx=12288,
        # n_ctx=10240,     # 文脈長：長めでもOK（4096が推奨最大）
        # n_ctx=9126,
        # n_ctx=8192,
        # n_ctx=5120,
        n_ctx=4096,
        n_threads=20,    # Ryzen 7900のスレッド数に応じて（上限は自動でも良い）
        n_gpu_layers=-1, # -1はGPUをMaxまで使う
        # n_gpu_layers=49, # -1はGPUをMaxまで使う
        # n_batch=1024,         # 一度に処理するトークン数（大きいと高速・ただしVRAMに注意）
        n_batch=512,
        use_mmap=True,   # モデルファイルを RAM や VRAM に全て読み込む代わりに、ファイルシステムから直接メモリにマッピングして利用しようとします。
        # type_k=7,     # デフォルトはf16
        # offload_kqv=False,   # Attention 計算の一部 (K, Q, V の射影) を CPU に担当させる
        verbose=False    # 👈 出力を抑制
    )

async def check_idle_timeout():
    global llm, last_access_time
    if time.time() - last_access_time > IDLE_TIMEOUT:
        async with llm_lock:
            if llm:
                logger.info("⏳ 非アクティブ時間超過のためモデルをアンロードします")
                llm.close()  # リソースを解放
                llm = None  # ガベージコレクタが解放しやすくする
                gc.collect()
    if time.time() - last_access_time > GREAT_TIMEOUT:
        mt.unmount_ramdisk(RAMDISK_PATH)

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

# ✅ ストリーミングジェネレータ定義
def stream_generator(current_model, prompt, session_id):
    global IS_SEMAPHORE
    try:
        partial_text = ""
        for chunk in llm(
            prompt,
            max_tokens=2096,
            stop=["<|eot|>", "user:", "<|user|>", "</|assistant|>", "<|endoftext|>", "Q:"],
            # stop=["<|eot|>", "user:", "User:", "Assistant:", "assistant:"],
            # "<|user|>","</|assistant|>"
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
        # ✅ ストリームの終了通知
        yield "data: [DONE]\n\n"
        # メッセージを要約してChromaに保存
        # history_message = message_compress("assistant", partial_text)
        # is_compressed = history_message != partial_text
        # save_chroma("assistant", history_message, is_compressed)
        IS_SEMAPHORE = False

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        IS_SEMAPHORE = False
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    finally:
        IS_SEMAPHORE = False

# 非ストリームイング用
async def non_streaming_generator(current_model, prompt, session_id):
    output = llm(
        prompt, 
        max_tokens=1024, 
        stop=["<|eot|>", "<|endoftext|>", "user:", "Q:"],
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
    # メッセージを要約してChromaに保存
    # history_message = message_compress("assistant", assistant_message)
    # is_compressed = history_message != assistant_message
    # save_chroma("assistant", history_message, is_compressed)

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
            # include=["distances", "documents"],
        )
        print(f"🔍 Chroma検索クエリ: {message}")
        distances = query_results.get("distances", [[]])
        if distances and distances[0]:
            if distances[0][0]:
                if distances[0][0] >= 0.15:
                    print("❌️ ヒットしませんでした。")
                    print(f"✈️ 距離: {distances[0][0]}")
                    print(query_results["documents"][0][0])
                    return None
        print(f"🔍 Chroma検索結果: {query_results}")
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
    output = llm(
        compress_prompt, 
        max_tokens=256, 
        stop=["<|eot|>", "user:"],
    )
    # 文末のゴミを除去してから返す
    result = re.sub(r"(```+|[\[\(\{]*$)", "", output["choices"][0]["text"]).strip()
    return result

@app.exception_handler(RequestValidationError)
async def handler(request:Request, exc:RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=422)

@app.post("/v1/chat/completions")
async def create_completion(completion_data: ChatCompletionRequest):
# async def create_completion(completion_data):
    global last_access_time, IS_SEMAPHORE  #, chat_session_manager
    sleep_count = 0
    while IS_SEMAPHORE:
        if sleep_count > 1800: # 180秒待っても解放されない場合は強制終了
            IS_SEMAPHORE = False
            LLM.close()
            LLM = None
            gc.collect()
            break
        await asyncio.sleep(0.1)
        sleep_count += 1
    IS_SEMAPHORE = True
    async with concurrency_semaphore:
        model_alias = completion_data.model
        await initialize_model(model_alias)
        last_access_time = time.time()
        # print("#########################")
        # print(f"model_alias: {model_alias}")
        # print(f"stream: {completion_data.stream}")
        # print(f"messages: {completion_data}")

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
            prompt = select_sentence_representatives(split_and_clean_sentences(all_histories), EMBEDDING_FUNCTION)
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
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(prompt)

        """
        prompt = ""
        system_prompt = ""
        user_prompt = ""
        for msg in messages[::-1]:
            if msg.role == "system":
                system_prompt += msg.content
                continue
            if user_prompt == "":
                if msg.role == "user":
                    user_prompt = msg.content
                    continue
            history_message = read_chroma(msg.content)
            if not history_message:
                # Chromaに無い場合は、メッセージをそのまま使用
                history_message = msg.content
            if msg.content != history_message:
                print("#########################")
                print(f"role: {msg.role}")
                print(f"original: {msg.content}")
                print("---")
                print(f"compress: {history_message}")
            prompt = f"{msg.role}: {history_message}\n" + prompt
            if len(prompt) > 5120:
                break
        if prompt:
            prompt = (
                "## **(参考情報)会話のダイジェスト** \n" 
                + prompt + 
                "\n## **本題（以下に会話を続けて下さい。）** \n"
            )
        if user_prompt:
            prompt += f"user: {user_prompt}\n"
            # 🧠 最新のuser発言だけを、RAG履歴として保持しておく
            compressed_message = message_compress("user", user_prompt)
            is_compressed = compressed_message != user_prompt
            save_chroma("user", compressed_message, is_compressed)
        if system_prompt:
            prompt = f"system: {system_prompt}\n" + prompt
        prompt += "assistant: "
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(prompt)
        """

        if completion_data.stream:
            # ストリーミングの場合
            return StreamingResponse(stream_generator(current_model, prompt, session_id), media_type="text/event-stream")
        else:
            response = await non_streaming_generator(current_model, prompt, session_id)
            print(response)
            IS_SEMAPHORE = False
            return response
        
async def background_cleanup():
    while True:
        await asyncio.sleep(30) #30秒ごとにチェック
        await check_idle_timeout()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())

