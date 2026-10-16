import html
import sys

import gradio as gr
import numpy as np

from search_service import search_df


# ===============================
# Helper functions
# ===============================
def safe_get(row, col):
    try:
        v = row.get(col, "")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).strip()
    except Exception:
        return ""


def first_nonempty(row, cols):
    for c in cols:
        val = safe_get(row, c)
        if val:
            return val
    return ""


def short_preview(text, length=300):
    text = (text or "").strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."


# ===============================
# Search function (UI layer only)
# ===============================
def search_hadith(query, top_k):
    try:
        results_df, _ = search_df(query, int(top_k))
    except ValueError:
        # empty query or invalid top_k: friendly message for humans
        return "<p class='empty'>الرجاء إدخال استعلام بحث</p>"
    except Exception as e:
        print(f"[search_hadith] internal error: {e!r}", file=sys.stderr)
        return "<p class='empty'>حدث خطأ أثناء البحث، حاول مرة أخرى</p>"

    html_parts = ["<div class='results'>"]

    for rank, (hadith_idx, row) in enumerate(results_df.iterrows(), start=1):
        title_val = (
            first_nonempty(row, ["isnad_text", "hadith_title", "title", "main_subj"])
            or short_preview(safe_get(row, "matn_text"), 60)
            or f"حديث #{hadith_idx}"
        )
        clean_text = safe_get(row, "matn_text")
        clean_text_escaped = html.escape(clean_text).replace("\n", "<br>")
        topic = html.escape(safe_get(row, "main_subj"))
        source_url = safe_get(row, "url")
        score_display = f"{float(row['hybrid_score']):.3f}"
        preview_escaped = html.escape(short_preview(clean_text, 360)).replace("\n", "<br>")

        link_html = (
            f"<a href='{html.escape(source_url)}' target='_blank' rel='noopener noreferrer'>📖 عرض المصدر</a>"
            if source_url.startswith("http")
            else ""
        )

        card_html = f"""
        <div class="card">
            <div class="card-header">
                <div class="left">
                    <div class="index">#{rank}</div>
                    <div class="title">{html.escape(title_val)}</div>
                    <div class="topic">الموضوع: {topic}</div>
                </div>
                <div class="right-meta">
                    <div class="meta">الرقم: <span class="meta-val">{hadith_idx}</span></div>
                    <div class="meta">الدرجة: <span class="meta-val">{score_display}</span></div>
                </div>
            </div>
            <div class="text-rtl">
                <details>
                    <summary class="summary">{preview_escaped}</summary>
                    <div class="full-text">{clean_text_escaped}</div>
                </details>
            </div>
            <div class="footer">{link_html}</div>
        </div>
        """
        html_parts.append(card_html)

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ===============================
# Styling
# ===============================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Noto+Sans+Arabic:wght@400;500;700&display=swap');

:root {
    --body-background-fill: #020617 !important;
    --background-fill-primary: #020617 !important;
    --background-fill-secondary: #0f172a !important;
    --border-color-primary: #334155 !important;
    --color-text-primary: #f1f5f9 !important;
    --color-text-secondary: #cbd5e1 !important;
    --button-primary-background-fill: #3b82f6 !important;
    --button-primary-background-fill-hover: #2563eb !important;
}

body, .gradio-container, .gr-panel, .gr-box, .gr-form, .wrap, .panel, .block, footer {
    background-color: #020617 !important;
    color: #f1f5f9 !important;
    font-family: 'Cairo', 'Noto Sans Arabic', system-ui, sans-serif !important;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }

.text-rtl, .full-text, .summary, .title, .topic {
    font-family: 'Cairo', 'Noto Sans Arabic', sans-serif !important;
    line-height: 2.05 !important;
}

label span, input, textarea, .gr-input, .gr-textarea {
    color: #f1f5f9 !important;
    background: #1e2937 !important;
    border-color: #475569 !important;
}
button.primary {
    background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
    font-weight: 600 !important;
}

.card {
    background: #1e2937 !important;
    border-radius: 18px !important;
    padding: 26px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
    border: 1px solid #334155 !important;
    transition: all 0.35s ease !important;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 25px 50px rgba(59, 130, 246, 0.2) !important;
}
.card-header { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.index {
    display: inline-block;
    background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
    color: #fff !important;
    padding: 7px 16px !important;
    border-radius: 9999px !important;
    font-weight: 700 !important;
}
.title { font-size: 19px !important; font-weight: 700 !important; }
.topic { font-size: 14px !important; color: #94a3b8 !important; }
.meta { font-size: 13px; color: #94a3b8; }

.text-rtl {
    direction: rtl;
    text-align: right;
    background: #0f172a !important;
    border-radius: 12px !important;
    padding: 18px !important;
    border-right: 6px solid #3b82f6 !important;
    font-size: 17px !important;
    color: #e2e8f0 !important;
}

.results { margin-top: 24px; }
.footer a { color: #60a5fa !important; font-weight: 600 !important; }
.empty { color: #94a3b8 !important; text-align: center; padding: 50px 20px; font-size: 17px; }

@media (max-width: 768px) {
    .card-header { flex-direction: column !important; }
}
"""

# ===============================
# Interface
# ===============================
demo = gr.Interface(
    fn=search_hadith,
    inputs=[
        gr.Textbox(
            label="🔍 استعلام البحث",
            placeholder="مثال: أهمية النية في الإسلام",
            lines=2,
            max_lines=3,
        ),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="📌 عدد النتائج"),
    ],
    outputs=gr.HTML(),
    title="📖 Intelligent Hadith Search Engine",
    description=(
        "Semantic search over Sahih al-Bukhari.<br>"
        "Combines ColBERT late-interaction retrieval with BM25 lexical search."
    ),
    examples=[
        ["أهمية النية وأثرها في قبول الأعمال", 5],
        ["حقوق الوالدين في الإسلام", 5],
        ["فضل طلب العلم", 5],
    ],
    cache_examples=False,  # don't run searches at startup
    flagging_mode="never",
)

# one search at a time: the model runs on CPU, parallel calls would just slow each other
demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    # Gradio 6: theme and css are passed to launch(), not the constructor
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
