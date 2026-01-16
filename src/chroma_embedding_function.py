from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)

class ChromaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB用の埋め込み関数クラス"""
    
    def __init__(self, model_name: str = "cl-nagoya/ruri-small-v2"):
        """
        埋め込みモデルを初期化する
        
        :param model_name: 使用する埋め込みモデル名
        :raises RuntimeError: モデルのロードに失敗した場合
        """
        try:
            logger.info(f"埋め込みモデル '{model_name}' をロード中...")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            logger.info(f"✅ 埋め込みモデル '{model_name}' をロードしました")
        except Exception as e:
            error_msg = f"埋め込みモデルのロードに失敗しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        テキストリストを埋め込みベクトルに変換する
        
        :param input: 埋め込み対象のテキストリスト
        :return: 埋め込みベクトルのリスト
        :raises RuntimeError: 埋め込み処理に失敗した場合
        """
        try:
            if not input:
                return []
            
            model_vecs = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
            vecs = normalize(model_vecs, norm="l2")  # L2正規化
            return vecs.tolist()
        except Exception as e:
            error_msg = f"埋め込み処理に失敗しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e