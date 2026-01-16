import os
import subprocess
import json
from dotenv import load_dotenv
import shutil
import logging
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# 設定値の読み込み
RAMDISK_PATH = os.getenv("RAMDISK_PATH", "/mnt/temp/hoshikage")
MODEL_MAP_FILE = get_env_path("MODEL_MAP_FILE", "data/model_map.json")
RAMDISK_SIZE = int(os.getenv("RAMDISK_SIZE", 12))  # デフォルトは12GB

def get_model_info(model_alias):
    if not os.path.exists(MODEL_MAP_FILE):
        raise FileNotFoundError(f"モデルマップファイルが見つかりません: {MODEL_MAP_FILE}")
    with open(MODEL_MAP_FILE, "r") as f:
        model_maps = json.load(f)
        model_data = model_maps.get(model_alias, {})
        model = model_data.get("model", None)
        path = model_data.get("path", None)
        # 設定情報も含めて返す
        config = model_data
        if model is None or path is None:
            return None, {}
        source_model_path = os.path.join(path, model)
        ramdisk_model_path = os.path.join(RAMDISK_PATH, model)
        return source_model_path, ramdisk_model_path, config

def is_mounted(path):
    """指定パスがマウントされているか確認"""
    return os.path.ismount(path)

def mount_ramdisk(size_gb=10):
    """
    指定されたマウントポイントに指定サイズのRAMディスクをマウントします。

    :param mount_point: RAMディスクのマウントポイント
    :param size_gb: RAMディスクのサイズ（GB単位）
    """
    size_mb = size_gb * 1024
    try:
        """tmpfsとしてRamdiskをマウント"""
        if not os.path.exists(RAMDISK_PATH):
            os.makedirs(RAMDISK_PATH)
        if not is_mounted(RAMDISK_PATH):
            logger.info("🔧 Ramdiskをマウントします...")
            subprocess.run(["sudo", "mount", "-t", "tmpfs", "-o", f"size={size_mb}M", "tmpfs", RAMDISK_PATH], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"マウント中にエラーが発生しました: {e}")

def unmount_ramdisk(mount_point=RAMDISK_PATH):
    """
    指定されたマウントポイントのRAMディスクをアンマウントします。

    :param mount_point: アンマウントするRAMディスクのマウントポイント
    """
    if is_mounted(mount_point):
        try:
            # コマンドインジェクションに注意すること！！！
            subprocess.run(['sudo', 'umount', mount_point], check=True)
            logger.info(f"{mount_point} のRAMディスクがアンマウントされました。")
        except subprocess.CalledProcessError as e:
            logger.error(f"アンマウント中にエラーが発生しました: {e}")

def copy_model(source_model_path):
    """Ramdiskにモデルをコピー"""
    logger.info("🚀 モデルをRamdiskへコピー中...")
    # shutil.copy(source_model_path, RAMDISK_PATH)
    # cp コマンドを使用してファイルをコピー
    # コマンドインジェクションに注意すること！！！
    # コマンドインジェクション対策: shell=False, 引数をリストで渡す
    command = ["cp", source_model_path, RAMDISK_PATH]
    subprocess.run(command, check=True) # `cp` コマンドを実行

def prepare_ram_model(source_model_path):
    """全体処理：マウントして、モデルをコピー"""
    unmount_ramdisk()
    mount_ramdisk()
    if source_model_path is not None:
        copy_model(source_model_path)

def get_model(model_alias):
    source_model_path, ramdisk_model_path, config = get_model_info(model_alias)
    if is_mounted(RAMDISK_PATH):
        if os.path.exists(ramdisk_model_path):
            logger.info("✅ Ramdisk上にモデルが既に存在しています")
            return ramdisk_model_path, config
    prepare_ram_model(source_model_path)
    return ramdisk_model_path, config
    
# prepare_ram_model(None)
# unmount_ramdisk()