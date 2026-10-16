from mcp.server.fastmcp import FastMCP
from search_service import search_df

mcp = FastMCP("hadithMCP")


@mcp.tool()
def search_hadith(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """
    Search Sahih al-Bukhari for hadiths relevant to a query.

    Uses hybrid retrieval: ColBERT late-interaction semantic search combined
    with BM25 keyword search. The query can be a topic, question, or
    description of a meaning, in Arabic. It does not need to match the
    hadith wording exactly.

    Parameters:
        query: Non-empty search text, preferably Arabic.
        top_k: Number of hadiths to return, 1 to 20 (default 5).
            Values above 20 are limited to 20.

    Returns:
        A list of hadiths, best match first. Each item has:
        - "hadith": the hadith text (matn)
        - "isnad_text": the chain of narration
        - "url": source link for further reading

    Errors:
        Fails if query is empty or top_k is below 1.
    """
    results_df, _ = search_df(query, top_k)
    return [
        {
            "hadith": str(row["matn_text"]),
            "isnad_text": str(row["isnad_text"]),
            "url": str(row["url"]),
        }
        for _, row in results_df.iterrows()
    ]


if __name__ == "__main__":
    mcp.run()