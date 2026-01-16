from pydantic import BaseModel
from typing import List, Literal, Optional

class ChatSessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id: str):
        self.sessions[session_id] = []

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.create_session(session_id)
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str):
        return self.sessions.get(session_id, [])
    
    def get_last_message(self, session_id: str):
        messages = self.get_messages(session_id)
        if messages:
            return messages[-1]
        return None
    
    def get_messages(self, session_id: str):
        return self.sessions.get(session_id, [])

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Hoshikage"  # 動的ロード用のキー名
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
