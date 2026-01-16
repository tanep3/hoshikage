from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class ChromaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "cl-nagoya/ruri-small-v2"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: list[str]) -> list[list[float]]:
        model_vecs = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        vecs = normalize(model_vecs, norm="l2")  # 🔧 L2正規化！
        return vecs.tolist()