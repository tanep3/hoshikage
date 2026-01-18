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
            return None, None, {}
        source_model_path = os.path.join(path, model)
        ramdisk_model_path = os.path.join(RAMDISK_PATH, model)
        return source_model_path, ramdisk_model_path, config

def is_mounted(path):
    """指定パスがマウントされているか確認"""
    return os.path.ismount(path)

def mount_ramdisk(size_gb=None):
    """
    指定されたマウントポイントに指定サイズのRAMディスクをマウントします。

    :param size_gb: RAMディスクのサイズ（GB単位）。Noneの場合は環境変数RAMDISK_SIZEを使用
    :raises RuntimeError: マウントに失敗した場合
    """
    if size_gb is None:
        size_gb = RAMDISK_SIZE
    
    size_mb = size_gb * 1024
    try:
        """tmpfsとしてRamdiskをマウント"""
        if not os.path.exists(RAMDISK_PATH):
            os.makedirs(RAMDISK_PATH)
        if not is_mounted(RAMDISK_PATH):
            logger.info("🔧 Ramdiskをマウントします...")
            result = subprocess.run(
                ["sudo", "mount", "-t", "tmpfs", "-o", f"size={size_mb}M", "tmpfs", RAMDISK_PATH],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ RAMディスクをマウントしました（サイズ: {size_gb}GB）")
    except subprocess.CalledProcessError as e:
        error_msg = f"マウント中にエラーが発生しました: {e}"
        if e.stderr:
            error_msg += f"\nエラー出力: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"マウント中に予期しないエラーが発生しました: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def unmount_ramdisk(mount_point=RAMDISK_PATH):
    """
    指定されたマウントポイントのRAMディスクをアンマウントします。

    :param mount_point: アンマウントするRAMディスクのマウントポイント
    :raises RuntimeError: アンマウントに失敗した場合
    """
    if is_mounted(mount_point):
        try:
            # コマンドインジェクションに注意すること！！！
            result = subprocess.run(
                ['sudo', 'umount', mount_point],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ {mount_point} のRAMディスクがアンマウントされました。")
        except subprocess.CalledProcessError as e:
            error_msg = f"アンマウント中にエラーが発生しました: {e}"
            if e.stderr:
                error_msg += f"\nエラー出力: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"アンマウント中に予期しないエラーが発生しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    else:
        logger.info(f"{mount_point} はマウントされていません")

def copy_model(source_model_path):
    """
    Ramdiskにモデルをコピーする
    
    :param source_model_path: コピー元のモデルパス
    :raises RuntimeError: コピーに失敗した場合
    """
    logger.info("🚀 モデルをRamdiskへコピー中...")
    try:
        # cp コマンドを使用してファイルをコピー
        # コマンドインジェクションに注意すること！！！
        # コマンドインジェクション対策: shell=False, 引数をリストで渡す
        command = ["cp", source_model_path, RAMDISK_PATH]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ モデルをRAMディスクにコピーしました: {source_model_path}")
    except subprocess.CalledProcessError as e:
        error_msg = f"モデルのコピー中にエラーが発生しました: {e}"
        if e.stderr:
            error_msg += f"\nエラー出力: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"モデルのコピー中に予期しないエラーが発生しました: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def prepare_ram_model(source_model_path):
    """
    全体処理：マウントして、モデルをコピー
    
    :param source_model_path: コピー元のモデルパス
    :raises RuntimeError: 準備に失敗した場合
    """
    try:
        unmount_ramdisk()
        mount_ramdisk()
        if source_model_path is not None:
            copy_model(source_model_path)
    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"RAMモデルの準備中に予期しないエラーが発生しました: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

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