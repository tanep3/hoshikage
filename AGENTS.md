---
model: openrouter/glm-4.7   # 自分が使ってるprovider名
temperature: 0.6
max_tokens: 8192
---

# This guide is for agentic coding assistants working in the Hoshikage repository.

## プロジェクト特有の追加ルール：
- Python, Rustの使用を前提
- 常にエラーハンドリングを明示的に
- テスト駆動を意識（テストが書かれていない機能は追加提案）

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server (with hot reload)
./start-dev.sh
# or manually:
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 3030

# Start production server
cd src && uvicorn main:app --host 0.0.0.0 --port 3030 --workers 1

# Model management CLI
cd src && python hoshikage.py add <model_path> <alias> [stop_tokens]
cd src && python hoshikage.py remove <alias>
cd src && python hoshikage.py list

# Environment setup
cp .env.example .env  # then edit .env with your settings
```

Note: This project does not currently have tests or linting configured. Consider adding pytest, ruff, and mypy for better code quality.

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Local module imports last
- Group imports with blank lines between groups

Example:
```python
import os
import asyncio
from datetime import datetime

import logging
from fastapi import FastAPI, HTTPException
from llama_cpp import Llama

from models.schema import ChatCompletionRequest
import mount as mt
```

### Type Hints
Use typing module for type hints:
```python
from typing import List, Optional, Dict, Any, Literal

def process_messages(messages: List[str]) -> Optional[str]:
    return None
```

### Naming Conventions
- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Error Handling
```python
# File operations with proper exception handling
try:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    logger.error(f"JSON decode error in {filepath}: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError(f"Failed to load {filepath}") from e

# HTTP errors with structured response
raise HTTPException(
    status_code=500,
    detail={
        "error": {
            "code": "model_load_failed",
            "message": f"Model load failed: {str(e)}",
            "type": "internal_server_error"
        }
    }
)
```

### Logging
Use Python's logging module:
```python
import logging

logger = logging.getLogger(__name__)

logger.info("✅ Operation completed")
logger.warning("⚠️  Warning message")
logger.error("❌ Error occurred")
logger.debug("Debug info")
```

### Environment Variables
Use `python-dotenv` and resolve relative paths to project root:
```python
import pathlib
from dotenv import load_dotenv

project_root = pathlib.Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

def get_env_path(key: str, default_rel_path: str) -> str:
    val = os.getenv(key)
    if val:
        if val.startswith("./"):
            return str(project_root / val[2:])
        return val
    return str(project_root / default_rel_path)
```

### File I/O
Always specify UTF-8 encoding:
```python
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### Async Patterns
- Use `asyncio.Lock` for thread-safe operations
- Use `asyncio.Semaphore` for concurrency control
- Always release locks in finally blocks or use async context managers
```python
async with self.llm_lock:
    # critical section
```

### Pydantic Models
Define models in `src/models/schema.py`:
```python
from pydantic import BaseModel
from typing import List, Literal, Optional

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
```

### Documentation
- Use Japanese for docstrings and comments
- Follow Google-style docstrings where applicable
- Include emoji prefixes for status messages: ✅, ⚠️, ❌, 🔧, 🚀

### Project Structure
- `src/main.py` - FastAPI application, API endpoints
- `src/hoshikage.py` - CLI for model management
- `src/models/` - Pydantic schemas
- `src/mount.py` - RAM disk utilities
- `src/chroma_embedding_function.py` - ChromaDB embedding function
- `src/select_sentence_representatives.py` - Sentence clustering/summarization
- `data/` - Runtime data (gitignored, use .env.example to set paths)

### Security Notes
- Use shell=False in subprocess calls with list arguments (not shell strings)
- Never commit `.env` file (use .env.example)
- All subprocess commands should use list format to prevent injection

### Subprocess Safety
```python
# Good - safe from injection
subprocess.run(['sudo', 'umount', mount_point], check=True)

# Bad - vulnerable to injection
subprocess.run(f'sudo umount {mount_point}', shell=True)
```
