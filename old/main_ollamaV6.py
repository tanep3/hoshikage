import os
import asyncio
import time
import logging
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from dotenv import load_dotenv
from llama_cpp import Llama
import mount as mt
from uuid import uuid4
from models.schema import ChatCompletionRequest, ChatSessionManager
import json
from datetime import datetime

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
TAG_OLLAMA_FILE = os.getenv("TAG_OLLAMA_FILE", "./models/tags_ollama.json")

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
    print(current_model)
    ram_model_path = mt.get_model(current_model)
    if llm is not None:
        llm.close()
    llm = Llama(
        model_path=ram_model_path, 
        # n_ctx=2096,
        # n_ctx=122880,
        # n_ctx=61440,
        n_ctx=20960,     # 文脈長：長めでもOK（4096が推奨最大）
        n_threads=12,    # Ryzen 7900のスレッド数に応じて（上限は自動でも良い）
        n_gpu_layers=-1, # GPUをMaxまで使う
        n_batch=1024,         # 一度に処理するトークン数（大きいと高速・ただしVRAMに注意）
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
    if time.time() - last_access_time > GREAT_TIMEOUT:
        mt.unmount_ramdisk(RAMDISK_PATH)

@app.get("/status")
@app.get("/v1/status")
async def status():
    return {"status": "ok"}

@app.get("/api/tags")
async def get_ollama_tags():
    try:
        if not os.path.exists(TAG_OLLAMA_FILE):
            raise FileNotFoundError(f"モデルキャッシュファイルが見つかりません: {TAG_OLLAMA_FILE}")
        with open(TAG_OLLAMA_FILE, "r", encoding="utf-8") as f:
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

@app.get("/v1/models")
async def get_models():
    try:
        if not os.path.exists(TAG_CACHE_FILE):
            raise FileNotFoundError(f"モデルキャッシュファイルが見つかりません: {TAG_CACHE_FILE}")
        with open(TAG_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"/v1/models エラー: {e}")
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

@app.get("/api/version")
@app.get("/v1/api/version")
async def get_version():
    return {"version": VERSION}

# 疑似ストリーミングジェネレータ
def ollama_stream(model_alias, prompt, session_id):
    partial_text = ""
    try:
        for chunk in llm(
            prompt,
            max_tokens=2096,
            stream=True,
            stop=["<|eot|>", "user:", "User:"]
        ):
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if not delta:
                continue
            print(delta)
            partial_text += delta
            response_chunk = {
                "model": model_alias,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": delta
                },
                "done": False
            }
            yield json.dumps(response_chunk, ensure_ascii=False) + "\n"

        # ✅ 終了時の最後のチャンク
        final_chunk = {
            "model": model_alias,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": True
        }
        yield json.dumps(final_chunk, ensure_ascii=False) + "\n"

        chat_session_manager.add_message(session_id, "assistant", partial_text)

    except Exception as e:
        logger.error(f"[/api/chat] stream error: {e}")
        yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"


@app.post("/api/chat")
async def ollama_chat(request: Request):
    global last_access_time, chat_session_manager
    try:
        body = await request.json()
        messages = body.get("messages", [])
        model_latest = body.get("model", "")
        model_alias = model_latest.split(":")[0]
        session_id = body.get("session_id", "default_session")
        stream = body.get("stream", False)
        print(body)
        prompt = ""
        for message in messages:
            prompt += f"{message['role']}: {message['content']}\n"
        prompt += "assistant: "

        async with llm_lock:
            await initialize_model(model_alias)
            last_access_time = time.time()

            # セッション履歴として保存（任意）
            chat_session_manager.add_message(session_id, "user", prompt)

            # return PlainTextResponse(content=ollama_stream(model_alias, prompt, session_id), media_type="application/jsonlines")
            return StreamingResponse(ollama_stream(model_alias, prompt, session_id), media_type="application/x-ndjson")
        
    except Exception as e:
        logger.error(f"/api/chat エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "ollama_stream_failed",
                    "message": str(e),
                    "type": "internal_server_error"
                }
            }
        )

@app.post("/v1/chat/completions")
async def create_completion(completion_data: ChatCompletionRequest):
    global last_access_time, chat_session_manager
    async with llm_lock:
        await initialize_model(completion_data.model)
        last_access_time = time.time()
        try:
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

            output = llm(
                prompt, 
                max_tokens=1024, 
                stop=["User:"],
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
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": "internal_server_error",
                        "message": str(e),
                        "type": "internal_server_error"
                    }
                }
            )

async def background_cleanup():
    while True:
        await asyncio.sleep(30) #30秒ごとにチェック
        await check_idle_timeout()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup())

