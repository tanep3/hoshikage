import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_cpp import Llama
from mount import prepare_ram_model

# .env ファイルの読み込み
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-12b-it-q4_K_M.gguf")
RAMDISK = os.getenv("RAMDISK_PATH", "/mnt/hoshikage")
MODEL_PATH = os.path.join(RAMDISK, MODEL_NAME)

# FastAPI アプリケーションの初期化
app = FastAPI()

# グローバルな Llama インスタンス (シングルトン)
llama = None
llama_lock = asyncio.Lock()

# モデルの初期化 (初回起動時のみ実行)
async def initialize_model():
    global llama
    with llama_lock:
        if llama is None:
            print(f"Initializing Llama model from {MODEL_PATH}")
            llama = Llama(model_path=MODEL_PATH, n_ctx=2048) # n_ctxは文脈長

# OpenAI 互換の Chat Completion リクエストボディの定義
class ChatCompletionRequest(BaseModel):
    model: str = "Hoshikage"  # モデル名 (OpenAI 互換)
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

# エンドポイント定義
@app.post("/v1/chat/complections")
async def chat_completion(request: ChatCompletionRequest):
    """
    OpenAI 互換のチャットコンプリーションエンドポイント
    """
    try:
        # モデルの初期化 (必要な場合)
        await initialize_model()

        # プロンプトの構築
        prompt = ""
        for message in request.messages:
            if message["role"] == "system":
                prompt += f"{message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            else:
                prompt += f"Assistant: {message['content']}\n" # assistantメッセージを考慮

        # 推論の実行
        output = llama.create_completion(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False # stream: false が必須
        )

        text = output["choices"][0]["text"]

        # OpenAI 互換のレスポンス形式での返却
        response = {
            "id": "chatcmpl-example",
            "object": "chat.completion",
            "created": 1678888888,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        }

        return response

    except Exception as e:
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

