import os
import asyncio
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from llama_cpp import Llama
import mount as mt
from uuid import uuid4
from models.schema import ChatCompletionRequest, ChatSessionManager
import json
import gc
import re

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

# .env 読み込み
load_dotenv()
RAMDISK_PATH = os.getenv("RAMDISK_PATH", "/mnt/temp/hoshikage")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))
GREAT_TIMEOUT = int(os.getenv("GREAT_TIMEOUT", "60")) * 60
MODEL_MAP_FILE = os.getenv("MODEL_MAP_FILE", "./models/model_map.json")
TAG_CACHE_FILE = os.getenv("TAG_CACHE_FILE", "./models/tags_cache.json")

llm = None
llm_lock = asyncio.Lock()
concurrency_semaphore = asyncio.Semaphore(1) # 同時実行数を1に制限
last_access_time = time.time()
chat_session_manager = ChatSessionManager()
current_model = ""

async def initialize_model(model_alias):
    global llm, current_model
    if model_alias == current_model:
        if llm is not None:
            return
    current_model = model_alias
    ram_model_path = mt.get_model(current_model)
    if llm is not None:
        llm.close()
    llm = Llama(
        model_path=ram_model_path, 
        # n_ctx=2096,
        # n_ctx=122880,
        # n_ctx=61440,
        n_ctx=20960,
        # n_ctx=4096,     # 文脈長：長めでもOK（4096が推奨最大）
        n_threads=12,    # Ryzen 7900のスレッド数に応じて（上限は自動でも良い）
        n_gpu_layers=-1, # -1はGPUをMaxまで使う
        n_batch=1024,         # 一度に処理するトークン数（大きいと高速・ただしVRAMに注意）
        verbose=False    # 👈 出力を抑制
    )
    gc.collect()

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
async def stream_generator(current_model, prompt, session_id):
    try:
        partial_text = ""
        for chunk in llm(
            prompt,
            max_tokens=2096,
            stop=["<|eot|>", "user:"],
            # stop=["<|eot|>", "user:", "User:", "Assistant:", "assistant:"],
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
        # セッションに最終メッセージを追加
        chat_session_manager.add_message(session_id, "assistant", partial_text)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

# 非ストリームイング用
async def non_streaming_generator(current_model, prompt, session_id):
    output = llm(
        prompt, 
        max_tokens=1024, 
        stop=["<|eot|>", "user:"],
    )
    assistant_message = output["choices"][0]["text"]
    chat_session_manager.add_message(session_id, "assistant", assistant_message)

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

def message_compress(message: str, role: str) -> str:
    if "```python" in message or role == "system":
        return message  # コードやシステム指示はそのまま保持
    if len(message) <= 150:
        return message  # 150文字以下は無圧縮
    compress_prompt = f"system: 次の内容を150文字以内に親しみやすい口調で要約してください。文章の主たる意図を失わないように要約してください。語調や言い回しなど会話表現的なニュアンスが有る場合は、それを維持するよう心がけて下さい。回答は要約のみを出力し、文字数などの情報は付与しないこと。Markdown表記は平文に直して下さい。制御コードは削除して、文字数削減に務めること。\nuser: {message}\nassistant: "
    output = llm(
        compress_prompt, 
        max_tokens=256, 
        stop=["<|eot|>", "user:"],
    )
    # 文末のゴミを除去してから返す
    result = re.sub(r"(```+|[\[\(\{]*$)", "", output["choices"][0]["text"]).strip()
    return result

@app.post("/v1/chat/completions")
async def create_completion(completion_data: ChatCompletionRequest):
    global last_access_time, chat_session_manager
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

        # 🧠 最新のuser発言だけを、RAG履歴として保持しておく
        latest_user_msg = None
        # for i in range(len(messages)-1, -1, -1):
        #     if messages[i].role == "user":
        #         latest_user_msg = messages[i]
        #         break
        for msg in messages[::-1]:
            if msg.role == "user":
                latest_user_msg = msg
                break
        if latest_user_msg:
            chat_session_manager.add_message(session_id, "user", latest_user_msg.content)

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
            compressed_message = message_compress(msg.content, msg.role)
            if msg.content != compressed_message:
                print("#########################")
                print(f"role: {msg.role}")
                print(f"original: {msg.content}")
                print("---")
                print(f"compress: {compressed_message}")
            prompt = f"{msg.role}: {compressed_message}\n" + prompt
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
        if system_prompt:
            prompt = f"system: {system_prompt}\n" + prompt
        prompt += "assistant: "
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(prompt)

        if completion_data.stream:
            # ストリーミングの場合
            return StreamingResponse(stream_generator(current_model, prompt, session_id), media_type="text/event-stream")
        else:
            # 非ストリーミングの場合
            # pre_messages = chat_session_manager.get_messages(session_id)
            # print(f"pre_messages: {pre_messages}")
            # for i in range(len(pre_messages)-1, -1, -1):
            #     if pre_messages[i].get("role"):
            #         prompt = f"{pre_messages[i].get("role")}: {pre_messages[i].get("content")}\n" + prompt
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(prompt)
            response = await non_streaming_generator(current_model, prompt, session_id)
            print(response)
            return response
        
async def background_cleanup():
    while True:
        await asyncio.sleep(30) #30秒ごとにチェック
        await check_idle_timeout()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())

