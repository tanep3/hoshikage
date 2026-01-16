'''
hoshikage.py は、星影システムの設定を行うためのスクリプトプログラムと位置づけます。
使い方は以下の通り。
python hoshikage.py [command] args...
commandは以下のコマンドが必要。
１．add: モデルを登録する。
使い方： python hoshikage.py add [登録用のモデル名] [モデルのパス]
処理：model_map.json、tags_cache.json にモデルを追加する。もし、モデル名が登録済みだったら、エラーにする。
２．remove: モデルを削除する。
使い方：python hoshikage.py remove [モデル名]
３．list: 登録されているモデルを一覧表示する。必要な情報は、モデル名、モデルパス、モデルのサイズ。
'''

import os
import json
import hashlib
import datetime
import sys
from llama_cpp import Llama
from dotenv import load_dotenv
import pathlib

# .env 読み込み（プロジェクトルートから）
project_root = pathlib.Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

def get_env_path(key, default_rel_path):
    val = os.getenv(key)
    if val:
        if val.startswith("./"):
            return str(project_root / val[2:])
        return val
    return str(project_root / default_rel_path)

MODEL_MAP_FILE = get_env_path("MODEL_MAP_FILE", "data/model_map.json")
TAG_CACHE_FILE = get_env_path("TAG_CACHE_FILE", "data/tags_cache.json")
TAG_OLLAMA_FILE = get_env_path("TAG_OLLAMA_FILE", "data/tags_ollama.json")


def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def format_openai_model(name):
    return {
      "id": name,
      "object": "model",
      "created": 1686935002,
      "owned_by": "tane"
    }

def get_file_metadata(full_path):
    stat = os.stat(full_path)
    size = stat.st_size
    modified_at = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    with open(full_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return size, modified_at, digest

def format_ollama_model(name, full_path):
    # llm = Llama(model_path=full_path, vocab_only=True, verbose=False)
    # meta = llm.metadata
    size, modified_at, digest = get_file_metadata(full_path)

    return {
        "name": name,
        "model": name + ":latest",
        "modified_at": modified_at,
        "size": size,
        "digest": digest,
        "details": {
            # "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": "null",
            "parameter_size": "12B",
            "quantization_level": "Q4_0"
        }
    }


def add_model(model_path, model_alias):
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが存在しません: {model_path}")
        return

    model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)

    model_map = load_json(MODEL_MAP_FILE)
    tags_cache = load_json(TAG_CACHE_FILE).get("data", [])
    tags_ollama = load_json(TAG_OLLAMA_FILE).get("models", [])

    if model_alias in model_map:
        print(f"❌ モデル名 '{model_alias}' はすでに登録されています。")
        return

    model_map[model_alias] = {"path": model_dir, "model": model_name}
    formatted = format_openai_model(model_alias)
    formatted_ollama = format_ollama_model(model_alias, model_path)
    tags_cache.append(formatted)
    tags_json = {
        "object": "list",
        "data": tags_cache,
    }
    tags_ollama.append(formatted_ollama)
    tags_ollama_json = {
        "models": tags_ollama,
    }
    save_json(MODEL_MAP_FILE, model_map)
    save_json(TAG_CACHE_FILE, tags_json)
    save_json(TAG_OLLAMA_FILE, tags_ollama_json)
    print(f"✅ モデル '{model_alias}' を追加しました。")


def remove_model(model_alias):
    model_map = load_json(MODEL_MAP_FILE)
    tags_cache = load_json(TAG_CACHE_FILE).get("data", [])
    tags_ollama = load_json(TAG_OLLAMA_FILE).get("models", [])

    if model_alias not in model_map:
        print(f"❌ モデル '{model_alias}' は登録されていません。")
        return

    del model_map[model_alias]
    tags_cache = [m for m in tags_cache if m["id"] != model_alias]
    tags_json = {
        "object": "list",
        "data": tags_cache,
    }
    tags_ollama = [m for m in tags_ollama if m["name"] != model_alias]
    tags_ollama_json = {
        "models": tags_ollama,
    }

    save_json(MODEL_MAP_FILE, model_map)
    save_json(TAG_CACHE_FILE, tags_json)
    save_json(TAG_OLLAMA_FILE, tags_ollama_json)
    print(f"🗑️ モデル '{model_alias}' を削除しました。")


def list_models():
    model_map = load_json(MODEL_MAP_FILE)
    print(f"📦 登録済みモデル一覧（{len(model_map)}件）:")
    for alias, conf in model_map.items():
        model_path = os.path.join(conf["path"], conf["model"])
        size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        print(f" - {alias}: {model_path} ({size / 1024 / 1024:.2f} MB)")

def usage():
    print("使い方: python hoshikage.py [add|remove|list] ...")
    print("add [モデルのフルパス] [alias]")
    print("remove [alias]")
    print("list")
    return

def main():
    if len(sys.argv) < 2:
        usage()
        return

    command = sys.argv[1]
    if command == "add" and len(sys.argv) == 4:
        add_model(sys.argv[2], sys.argv[3])
    elif command == "remove" and len(sys.argv) == 3:
        remove_model(sys.argv[2])
    elif command == "list":
        list_models()
    else:
        print("❌ コマンドが正しくありません。")
        usage()

main()