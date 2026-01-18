import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
import logging

logger = logging.getLogger(__name__)

def split_and_clean_sentences(text: str) -> list[str]:
    """
    星影要約用：コードブロック保護 + 英文と日本語の分割 + ノイズ除去処理

    Parameters:
    - text (str): 会話履歴などのプレーンテキスト

    Returns:
    - List[str]: クリーンな文単位のリスト（末尾句点なし・英字ノイズ除去済）
    """

    # === ① コードブロックの抽出 ===
    code_blocks = {}
    def extract_code_block(match):
        key = f"<CODE_BLOCK_{len(code_blocks)}_CODE_BLOCK>"
        code_blocks[key] = match.group(0)
        return key

    text = re.sub(r"```(?:python)?\n.*?```", extract_code_block, text, flags=re.DOTALL)

    # === ② 改行で一次分割 ===
    lines = text.splitlines()

    cleaned_sentences = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # コードブロックプレースホルダはそのまま
        if re.fullmatch(r"<CODE_BLOCK_\d+_CODE_BLOCK>", line):
            cleaned_sentences.append(code_blocks[line])
            continue

        # === ③ 英文判定：.?!で終わり & 日本語なし ===
        if re.search(r'[.?!]$', line) and not re.search(r'[ぁ-んァ-ン一-龥]', line):
            # 英文 → .?! + スペースで分割（自然な分割）
            english_fragments = re.split(r'(?<=[.?!])\s+', line)
            for frag in english_fragments:
                frag = frag.strip()
                if frag and not re.fullmatch(r'[^\w]+', frag):
                    cleaned_sentences.append(frag)
            continue

        # === ④ 日本語処理：。または改行で分割（句点除去）===
        jp_fragments = re.split(r'[。]', line)
        for frag in jp_fragments:
            frag = frag.strip()
            # ノイズ英字行もここで除去
            if frag and not re.fullmatch(r'[^\wぁ-んァ-ン一-龥a-zA-Z0-9]+', frag):
                # 英字のみ（記号除外）の場合は削除
                if re.fullmatch(r'[a-zA-Z0-9_()[\]{}\-+=:;\"\'*.,<>/@\\\s]+', frag):
                    continue  # ← ノイズなのでスキップ！

                # 🔥 意味不明防止：15文字以下の文は除外
                if len(frag) <= 15:
                    continue
                cleaned_sentences.append(frag)

    return cleaned_sentences

def is_english_line(line: str) -> bool:
    """
    たねちゃん式：英文判定ロジック（末尾が .!? のみ / 日本語文字を含まない）
    """
    line = line.strip()
    return (
        re.search(r'[.?!]$', line) is not None and   # ← 文末が .!? のいずれか
        not re.search(r'[ぁ-んァ-ン一-龥]', line)     # ← 日本語文字が含まれていない
    )

def format_clustered_representatives(representatives: list[tuple[int, int, str]]) -> str:
    """
    クラスタ代表文リストを Markdown 形式に整形（クラスタ間に --- を挿入）

    Parameters:
    - representatives: (cluster_id, original_index, sentence) のリスト（クラスタ順でソート済み）

    Returns:
    - str: Markdown形式の要約文字列
    """
    markdown = []
    current_cluster = None

    for cluster_id, _, sentence in representatives:
        if cluster_id != current_cluster:
            if current_cluster is not None:
                markdown.append('---')
            current_cluster = cluster_id
        markdown.append(f"- {sentence.strip()}")

    return '\n'.join(markdown)

def select_sentence_representatives(
    sentences: list[str],
    embedder,
    cluster_divisor: int = 100,
    min_clusters: int = 1,
    max_clusters: int = 20
) -> list[str]:
    """
    意味クラスタリングによる代表文抽出（等間隔 + 意味順ソート）

    Parameters:
    - sentences (List[str]): 分割・クレンジング済みの文リスト
    - embedder (Callable): __call__ で文リストをベクトル化できる関数（ruriインスタンス）
    - cluster_divisor (int): クラスタ数を計算するための除数（デフォルト: 100）
    - min_clusters (int): 最小クラスタ数（デフォルト: 1）
    - max_clusters (int): 最大クラスタ数（デフォルト: 20）

    Returns:
    - List[str]: クラスタ単位で意味順に並べられた代表文リスト
    """
    try:
        if not sentences:
            return []

        # ベクトル化
        vecs = np.array(embedder(sentences))

        # クラスタ数決定
        k = max(min_clusters, min(max_clusters, len(sentences) // cluster_divisor))
        logger.info(f"クラスタ数: {k}（文数: {len(sentences)}）")

        # クラスタリング
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_ids = kmeans.fit_predict(vecs)

        # 各クラスタごとに代表文を抽出（意味順出力のためクラスタ順にまとめる）
        representatives = []
        for cluster_id in range(k):
            indices = [i for i, cid in enumerate(cluster_ids) if cid == cluster_id]
            if not indices:
                continue

            cluster_vecs = vecs[indices]
            centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
            sims = cosine_similarity(cluster_vecs, centroid).flatten()
            sorted_pairs = sorted(zip(sims, indices), reverse=True)

            n_extract = min(math.ceil(len(indices) / 10), 5)
            logger.debug(f"クラスタ {cluster_id} の文数: {len(indices)}、代表文数: {n_extract}")
            step = max(1, len(sorted_pairs) // n_extract)

            selected = [sorted_pairs[i * step][1] for i in range(n_extract)]
            # (クラスタID, 元文インデックス, 文) を保存
            representatives.extend((cluster_id, i, sentences[i]) for i in selected)

        # クラスタ順 → インデックス順 で並べ替え
        representatives.sort(key=lambda x: (x[0], x[1]))

        return format_clustered_representatives(representatives)
    
    except ValueError as e:
        logger.error(f"クラスタリング中に値エラーが発生しました: {e}")
        # クラスタリング失敗時は、等間隔で代表文を抽出
        logger.info("クラスタリングに失敗したため、等間隔で代表文を抽出します")
        step = max(1, len(sentences) // 10)
        selected_indices = list(range(0, len(sentences), step))[:10]
        representatives = [(0, i, sentences[i]) for i in selected_indices]
        return format_clustered_representatives(representatives)
    except Exception as e:
        logger.error(f"代表文抽出中に予期しないエラーが発生しました: {e}")
        raise
