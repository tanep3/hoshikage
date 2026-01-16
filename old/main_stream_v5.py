import os
import asyncio
import time
import logging
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.responses import Response
from dotenv import load_dotenv
from llama_cpp import Llama
# from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import mount as mt
from uuid import uuid4
from models.schema import ChatCompletionRequest, ChatSessionManager
import json
import gc

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
        n_ctx=10240,
        # n_ctx=4096,     # 文脈長：長めでもOK（4096が推奨最大）
        n_threads=12,    # Ryzen 7900のスレッド数に応じて（上限は自動でも良い）
        n_gpu_layers=-1, # -1はGPUをMaxまで使う
        n_batch=512,         # 一度に処理するトークン数（大きいと高速・ただしVRAMに注意）
        # low_vram=True,
        # use_mlock=False,
        # embedding=True, #これ絶対にだめ。落ちる。
        # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10), #これもだめ。エラー出る。回答が途切れる。
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

# @app.get("/api/tags")
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

# @app.post("/api/chat")
@app.post("/v1/chat/completions")
async def create_completion(completion_data: ChatCompletionRequest):
    global last_access_time, chat_session_manager
    async with llm_lock:
        model_alias = completion_data.model
        await initialize_model(model_alias)
        last_access_time = time.time()

        messages = completion_data.messages
        session_id = messages[0].session_id if hasattr(messages[0], "session_id") else "default_session"

        # 🧠 最新のuser発言だけを、RAG履歴として保持しておく
        latest_user_msg = None
        for i in range(len(messages)-1, -1, -1):
            if messages[i].role == "user":
                latest_user_msg = messages[i]
        if latest_user_msg:
            chat_session_manager.add_message(session_id, "user", latest_user_msg.content)

        prompt = ""
        for message in messages:
            prompt += f"{message.role}: {message.content}\n"
        prompt += "assistant: "

        # ✅ ストリーミングジェネレータ定義
        def stream_generator():
            try:
                partial_text = ""
                for chunk in llm(
                    prompt,
                    max_tokens=2096,
                    stop=["<|eot|>"],
                    # stop=["<|eot|>", "user:", "User:", "Assistant:", "assistant:"],
                    stream=True
                ):
                    delta = chunk.get("choices", [{}])[0].get("text", "")
                    # delta = chunk["choices"][0]["text"]
                    if not delta:
                        continue
                    partial_text += delta
                    # OpenAI互換フォーマット
                    payload = {
                        "id": f"chatcmpl-{uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_alias,
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

        # st =  StreamingResponse(stream_generator(), media_type="text/event-stream")
        # st =  stream_generator()
        # logger.info("#############")
        # return st
        # return Response(stream_generator(), media_type="text/event-stream")
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # def wrapped_stream():
        #     try:
        #         for chunk in stream_generator():
        #             yield chunk
        #             gc.collect()
        #     except GeneratorExit:
        #         logger.warning("クライアントが切断しました")
        #     except Exception as e:
        #         logger.error(f"ストリーム中に例外: {e}")
        #         yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        # return StreamingResponse(wrapped_stream(), media_type="text/event-stream")

        # queue = asyncio.Queue()

        # async def producer():
        #     for chunk in llm(
        #         prompt,
        #         max_tokens=2096,
        #         stop=["<|eot|>"], 
        #         stream=True):
        #         await queue.put(chunk)
        #     await queue.put("[DONE]")

        # async def consumer():
        #     while True:
        #         item = await queue.get()
        #         if item == "[DONE]":
        #             yield "data: [DONE]\n\n"
        #             break
        #         delta = item["choices"][0]["text"]
        #         payload = {
        #             "id": f"chatcmpl-{uuid4().hex}",
        #             "object": "chat.completion.chunk",
        #             "created": int(time.time()),
        #             "model": model_alias,
        #             "choices": [{
        #                 "delta": {"content": delta},
        #                 "index": 0,
        #                 "finish_reason": None
        #             }]
        #         }
        #         yield f"data: {json.dumps(payload)}\n\n"

        # asyncio.create_task(producer())
        # return StreamingResponse(consumer(), media_type="text/event-stream")

async def background_cleanup():
    while True:
        await asyncio.sleep(30) #30秒ごとにチェック
        await check_idle_timeout()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())

