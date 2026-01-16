
import os
import time
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# .env 読み込み
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-12b-it-q4_K_M.gguf")
RAMDISK = os.getenv("RAMDISK_PATH", "/mnt/hoshikage")
MODEL_PATH = os.path.join(RAMDISK, MODEL_NAME)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))

app = FastAPI()
llama = None
llama_lock = asyncio.Lock()
last_access_time = time.time()

class ChatCompletionRequest(BaseModel):
    model: str = "Hoshikage"
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

async def initialize_model():
    global llama
    with await llama_lock:
        if llama is None:
            print(f"🚀 モデルをロードします: {MODEL_PATH}")
            llama = Llama(model_path=MODEL_PATH, n_ctx=2048)

async def check_idle_timeout():
    global llama, last_access_time
    if time.time() - last_access_time > IDLE_TIMEOUT:
        with await llama_lock:
            if llama is not None:
                print("⏳ 非アクティブ時間超過のためモデルをアンロードします")
                llama = None

@app.middleware("http")
async def update_last_access_time(request: Request, call_next):
    global last_access_time
    response = await call_next(request)
    last_access_time = time.time()
    return response

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        await initialize_model()
        await check_idle_timeout()

        prompt = ""
        for message in request.messages:
            role = message.get("role", "user")
            if role == "system":
                prompt += f"{message['content']}\n"
            elif role == "user":
                prompt += f"User: {message['content']}\n"
            elif role == "assistant":
                prompt += f"Assistant: {message['content']}\n"

        output = llama.create_completion(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False
        )
        text = output["choices"][0]["text"]

        return {
            "id": "chatcmpl-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": {
                "message": str(e),
                "type": "internal_error",
                "param": None,
                "code": "500"
            }
        })

@app.get("/v1/status")
async def status():
    return {
        "status": "running" if llama else "unloaded",
        "model": MODEL_NAME,
        "path": MODEL_PATH
    }

# Session管理の拡張構造（未実装）
class ChatSessionManager:
    def __init__(self):
        self.sessions = {}
    def get_context(self, session_id):
        return self.sessions.get(session_id, [])
    def update_context(self, session_id, messages):
        self.sessions[session_id] = messages
