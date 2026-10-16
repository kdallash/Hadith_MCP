import numpy as np


 

def bm25_retrieve(query, bm25, preprocess_query, top_k=50):
    tokenized_query = preprocess_query(query)
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]

def hybrid_search_fixed(
    query,
    df,
    bm25,
    preprocess_query,
    model,
    retriever,
    top_k=5,
    top=50,
    alpha_semantic=0.35,
    alpha_bm25=0.25,
):
    """
    Hybrid BM25 + ColBERT/PyLate search.

    BM25 and ColBERT scores are independently normalized by their
    maximum retrieved score, preserving their relative strength within
    the candidate pool.

    Parameters
    ----------
    query : str
        User search query.

    df : pandas.DataFrame
        Hadith corpus. Its index must correspond to the document IDs
        used when building the ColBERT/PyLate index.

    bm25 : object
        Initialized BM25 retriever.

    preprocess_query : callable
        Query preprocessing function used by BM25.

    model : object
        PyLate/ColBERT model used to encode the query.

    retriever : object
        PyLate retriever.

    top_k : int, default=5
        Number of final results to return.

    top : int, default=50
        Number of candidates retrieved from each retriever.

    alpha_semantic : float, default=0.35
        Weight given to the normalized ColBERT score.

    alpha_bm25 : float, default=0.25
        Weight given to the normalized BM25 score.

    Returns
    -------
    results_df : pandas.DataFrame
        Top-k results ranked by the hybrid score.

    debug : dict
        Normalized BM25/ColBERT scores and final hybrid scores.
    """

    # ---------------------------------------------------------
    # 1. Validate parameters
    # ---------------------------------------------------------

    if alpha_semantic < 0 or alpha_bm25 < 0:
        raise ValueError(
            "alpha_semantic and alpha_bm25 must be >= 0"
        )

    if alpha_semantic == 0 and alpha_bm25 == 0:
        raise ValueError(
            "At least one alpha weight must be > 0"
        )

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if top <= 0:
        raise ValueError("top must be > 0")

    # ---------------------------------------------------------
    # 2. BM25 retrieval
    # ---------------------------------------------------------

    bm25_indices, bm25_scores = bm25_retrieve(
        query,
        bm25,
        preprocess_query,
        top_k=top,
    )

    bm25_indices = [int(idx) for idx in bm25_indices]
    bm25_scores = [float(score) for score in bm25_scores]

    bm25_map = {
        idx: score
        for idx, score in zip(bm25_indices, bm25_scores)
    }

    # ---------------------------------------------------------
    # 3. ColBERT query encoding
    # ---------------------------------------------------------

    query_embeddings = model.encode(
        [query],
        is_query=True,
    )

    # ---------------------------------------------------------
    # 4. ColBERT retrieval
    # ---------------------------------------------------------

    colbert_results = retriever.retrieve(
        queries_embeddings=query_embeddings,
        k=top,
    )

    # One query → first result list
    colbert_results = colbert_results[0]

    # ---------------------------------------------------------
    # 5. Extract ColBERT IDs + scores
    # ---------------------------------------------------------

    colbert_map = {}

    for result in colbert_results:
        doc_id = int(result["id"])
        score = float(result["score"])

        colbert_map[doc_id] = score

    # ---------------------------------------------------------
    # 6. Normalize by maximum score
    # ---------------------------------------------------------

    def normalize_by_max(score_map):
        """
        Normalize scores by the maximum score.

        The strongest retrieved result becomes 1.0 and all other
        scores retain their magnitude relative to that strongest
        result.
        """

        if not score_map:
            return {}

        max_score = max(score_map.values())

        if max_score <= 0:
            return {
                idx: 0.0
                for idx in score_map
            }

        return {
            idx: float(score / max_score)
            for idx, score in score_map.items()
        }

    bm25_norm = normalize_by_max(bm25_map)
    colbert_norm = normalize_by_max(colbert_map)

    # ---------------------------------------------------------
    # 7. Union of candidates
    # ---------------------------------------------------------

    union_indices = (
        set(bm25_norm.keys())
        | set(colbert_norm.keys())
    )

    # ---------------------------------------------------------
    # 8. Hybrid score
    # ---------------------------------------------------------

    hybrid_scores = {}

    for idx in union_indices:

        bm25_score = bm25_norm.get(idx, 0.0)
        colbert_score = colbert_norm.get(idx, 0.0)

        hybrid_scores[idx] = (
            alpha_semantic * colbert_score
            + alpha_bm25 * bm25_score
        )

    # ---------------------------------------------------------
    # 9. Rank
    # ---------------------------------------------------------

    ranked_indices = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True,
    )[:top_k]

    # ---------------------------------------------------------
    # 10. Results
    # ---------------------------------------------------------

    results_df = df.loc[ranked_indices].copy()

    results_df["hybrid_score"] = [
        hybrid_scores[idx]
        for idx in ranked_indices
    ]

    results_df["colbert_score"] = [
        colbert_norm.get(idx, 0.0)
        for idx in ranked_indices
    ]

    results_df["bm25_score"] = [
        bm25_norm.get(idx, 0.0)
        for idx in ranked_indices
    ]

    # ---------------------------------------------------------
    # 11. Debug
    # ---------------------------------------------------------

    debug = {
        "final_scores": hybrid_scores,
        "colbert_scores": colbert_norm,
        "bm25_scores": bm25_norm,
        "union_size": len(union_indices),
        "bm25_candidates": len(bm25_norm),
        "colbert_candidates": len(colbert_norm),
    }

    return results_df, debug