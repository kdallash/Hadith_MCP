---
title: Hadith Semantic Search
emoji: 📖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.4.0
app_file: app.py
pinned: false
hugginface space :[space](https://huggingface.co/spaces/kdallash/Hadith_Semantic_Search)
---

# 📖 Hadith Semantic Search Engine

Semantic search over **Sahih al-Bukhari** (6,373 hadiths). Describe a topic or a
meaning in Arabic and get the most relevant hadiths back, even when your wording
differs from the text. The corpus is Sahih al-Bukhari only, so every result is
already authentic by construction.

The engine is available two ways: a Gradio web UI (this Space) and an
[MCP](https://modelcontextprotocol.io) server so AI agents can call it as a tool.

## How it works

```
query ──┬─► BM25 (lexical, top 50) ────────┐
        │                                   ├─► weighted fusion ─► top-k hadiths
        └─► ColBERT (late interaction, top 50) ┘
```

- **ColBERT** (`jinaai/jina-colbert-v2` via [PyLate](https://github.com/lightonai/pylate)):
  keeps one vector per token instead of one per document and scores with MaxSim,
  so short or abstract hadiths are not squeezed into a single averaged vector.
- **BM25**: exact-term matching with Arabic normalization (tashkeel removal,
  alef/ya/ta marbuta unification).
- **Fusion**: both score lists are min-max normalized and combined with
  weights 0.65 (ColBERT) / 0.35 (BM25), chosen from a sweep on LLM-judged relevance.

## Evaluation

Relevance was judged strictly by an LLM (shared words without a shared topic
count as irrelevant) on sets of 20 Arabic queries.

| Retriever | Judged relevance |
|---|---|
| Single-vector bi-encoder (STS Matryoshka) | about 26% |
| ColBERT | about 49% |

Treat the comparison as approximate: the two runs used different result windows
(top 20 vs. top 10) and an LLM judge, not human labels.

What changed along the way, from per-signal diagnostics:
- found and fixed an 84-row duplicate mismatch between the corpus, BM25 and the embeddings
- removed a topic-anchor signal that correlated 0.61 with the dense retriever and added no independent evidence
- replaced the bi-encoder with ColBERT

Typical latency is about 1 to 1.5 seconds per query on a laptop CPU.

## Known limitations

- **Homograph confusion.** Token-level matching still confuses words with a shared
  root or spelling: كبر (arrogance) vs. تكبير (saying "Allahu akbar"), خدم (servants)
  vs. its "anklets" sense, صدق (truthfulness) vs. صدقة (charity).
- **Abstract queries** (envy, humility, reliance on God) score lower than queries
  about concrete acts or well-known hadiths.
- Retrieval only. It returns hadith text and sources, not rulings or interpretation.

## Use it from an AI agent (MCP)

The same engine runs as an MCP server exposing one tool:

| | |
|---|---|
| Tool | `search_hadith` |
| `query` | string, required, non-empty, Arabic recommended |
| `top_k` | integer, optional, default 5, values above 20 are limited to 20 |
| Returns | list of `{ "hadith", "isnad_text", "url" }`, best match first |
| Errors | empty `query` or `top_k` below 1 returns an MCP error (`isError`) |

Server endpoint: `https://kdallash-hadith-semantic-search.hf.space/gradio_api/mcp/`

Example client configuration :

```json

{
  "mcpServers": {
    "hadith-search": {
      "url": "https://kdallash-hadith-semantic-search.hf.space/gradio_api/mcp/"
    }
  }
}

```

## Run locally

```bash
pip install -r requirements.txt
python app.py                     # Gradio UI
python mcp_server.py              # MCP server over stdio
npx @modelcontextprotocol/inspector python mcp_server.py   # inspect the MCP tool
```

Needs `data/hadith.csv`, `data/bm25.pkl` and the `data/hadith_colbert_index/` folder.

## Project layout

```
app.py              Gradio UI
mcp_server.py       MCP server
search_service.py   loads data and models once, shared by both
retrieval.py        BM25 + ColBERT fusion
utils.py            Arabic normalization and tokenization
```

## Credits

- ColBERT model: [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2)
- Retrieval library: [PyLate](https://github.com/lightonai/pylate)
- Hadith source links point to islamweb.net
