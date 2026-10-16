import sys
import pickle
from pathlib import Path

import pandas as pd
from pylate import indexes, models, retrieve

from retrieval import hybrid_search_fixed
from utils import preprocess_query

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ALPHA_SEMANTIC = 0.65
ALPHA_BM25 = 0.35
MAX_TOP_K = 20

df = pd.read_csv(DATA_DIR / "hadith.csv")

with open(DATA_DIR / "bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

# guard against the bug from earlier: df and bm25 must match
if len(df) != bm25.corpus_size:
    raise ValueError(f"df has {len(df)} rows but bm25 has {bm25.corpus_size}")
if not df.index.equals(pd.RangeIndex(len(df))):
    raise ValueError("df index must be 0..N-1, run reset_index(drop=True)")

model = models.ColBERT(
    model_name_or_path=str(BASE_DIR / "model" ),
    query_prefix="[QueryMarker]",
    document_prefix="[DocumentMarker]",
    attend_to_expansion_tokens=True,
    trust_remote_code=True,
)

index = indexes.PLAID(
    index_folder=str(DATA_DIR / "hadith_colbert_index"),
    index_name="hadith_bukhari",
    override=False,
)
retriever = retrieve.ColBERT(index=index)

print(f"loaded {len(df)} hadiths", file=sys.stderr)


def search_df(query, top_k=5):
    if query is None or not str(query).strip():
        raise ValueError("query must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    top_k = min(top_k, MAX_TOP_K)

    return hybrid_search_fixed(
        query=str(query).strip(),
        df=df,
        bm25=bm25,
        preprocess_query=preprocess_query,
        model=model,
        retriever=retriever,
        top_k=top_k,
        alpha_semantic=ALPHA_SEMANTIC,
        alpha_bm25=ALPHA_BM25,
    )