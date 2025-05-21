# modules/reranking.py
from sentence_transformers import CrossEncoder


def cross_encoder_rerank(query: str, candidates: list, top_k=3):
    """返回排序后的索引列表（整数）"""
    model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    pairs = [[query, doc] for doc in candidates]
    scores = model.predict(pairs)

    # 生成索引的排序列表（从高到低）
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # 返回前 top_k 的索引（整数）
    return ranked_indices[:top_k]  # 例如 [2, 0, 1]