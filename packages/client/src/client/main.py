from typing import Any

import gradio as gr
from model import BM25Model, BoWModel, LSAModel

custom_css = """
#main-container {
    max-width: 98vw !important;
    margin: 0 auto;
    overflow: visible !important;
}

#search-panel {
    position: sticky !important;
    top: 0 !important;
    z-index: 9999 !important;
    background: var(--body-background-fill, white) !important;
    padding-top: 20px !important;
    padding-bottom: 20px !important;
    width: 100% !important;
    border-bottom: 1px solid #eee !important;
    align-self: flex-start !important;
}

#search-panel .form,
#search-panel .block,
#search-panel > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#query-input {
    overflow: visible !important;
}

#query-input textarea {
    border-radius: 24px !important;
    padding: 12px 20px !important;
    border: 1px solid #dfe1e5 !important;
    resize: none !important;
    outline: none !important;
    background-color: white !important;
}

#query-input textarea:focus {
    border: 1px solid #dfe1e5 !important;
    box-shadow: 0 1px 6px rgba(32,33,36,.28) !important;
}

#search-row {
    max-width: 1000px !important;
    margin: 0 auto !important;
    align-items: center !important;
    gap: 12px !important;
    overflow: visible !important;
}

#submit-btn {
    border-radius: 24px !important;
    height: 48px !important;
    margin: 0 !important;
}

.result-group {
    border: 1px solid #eaeaea !important;
    border-radius: 8px !important;
}

.result-group > * {
    background: white !important;
}

.result-group * {
    border-color: #f2f2f2 !important;
}

.chunk-text textarea {
    overflow-y: auto !important;
}
"""


def get_color_for_score(score: float) -> str:
    clamped = max(0.0, min(1.0, float(score)))
    hue = int(clamped * 120)

    return f"hsl({hue}, 80%, 40%)"


def format_results(matched_documents) -> list[dict[str, Any]]:
    results = {}

    for document in matched_documents:
        if document.title in results:
            results[document.title]["documents"].append(
                {"text": document.text, "score": document.score}
            )
        else:
            results[document.title] = {
                "title": document.title,
                "url": document.url,
                "documents": [{"text": document.text, "score": document.score}],
            }

    return list(results.values())


def main():
    def predict(query) -> dict[str, list[dict[str, Any]]]:
        if not query:
            return {}

        top_n = 50

        return {
            "bm25": format_results(BM25Model.query(phrase=query, top_n=top_n)),
            "lsa": format_results(LSAModel.query(phrase=query, top_n=top_n)),
            "bow": format_results(BoWModel.query(phrase=query, top_n=top_n)),
        }

    with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
        search_results = gr.State({})

        with gr.Column(elem_id="main-container"):
            with gr.Column(elem_id="search-panel"):
                with gr.Row(elem_id="search-row"):
                    query_input = gr.Textbox(
                        placeholder="Search articles...",
                        show_label=False,
                        lines=1,
                        elem_id="query-input",
                        container=False,
                        scale=5,
                    )
                    submit_btn = gr.Button(
                        "Submit", variant="primary", elem_id="submit-btn", scale=1
                    )

            @gr.render(inputs=search_results)
            def render_articles(all_results):
                if not all_results:
                    return

                with gr.Row():
                    for display_name, dict_key in [
                        ("BM25", "bm25"),
                        ("LSA", "lsa"),
                        ("BoW", "bow"),
                    ]:
                        with gr.Column():
                            gr.Markdown(f"### {display_name} Results")
                            articles = all_results.get(dict_key, [])

                            if not articles:
                                gr.Markdown("*No results found.*")
                                continue

                            for article in articles:
                                best_score = max(
                                    chunk["score"] for chunk in article["documents"]
                                )

                                if dict_key in ["lsa", "bow"]:
                                    best_score_display = f"{best_score * 100:.1f}%"
                                    color = get_color_for_score(best_score)
                                else:
                                    best_score_display = f"{best_score:.4f}"
                                    color = "inherit"

                                with gr.Group(elem_classes="result-group"):
                                    header_html = f"""
                                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; border-bottom: 1px solid #f2f2f2; padding-bottom: 8px;">
                                        <h2 style="margin: 0; font-size: 1.1em; line-height: 1.2;">
                                            <a href="{article["url"]}" style="text-decoration: none; color: #1a0dab;" target="_blank">{article["title"]}</a>
                                        </h2>
                                        <div style="font-size: 1em; font-weight: bold; color: {color}; margin-left: 12px; flex-shrink: 0;">
                                            {best_score_display}
                                        </div>
                                    </div>
                                    """
                                    gr.HTML(header_html)

                                    with gr.Accordion(
                                        f"View {len(article['documents'])} matched chunks",
                                        open=False,
                                    ):
                                        for i, chunk in enumerate(article["documents"]):
                                            chunk_score_fmt = (
                                                f"{chunk['score'] * 100:.1f}%"
                                                if dict_key in ["lsa", "bow"]
                                                else f"{chunk['score']:.4f}"
                                            )

                                            gr.Textbox(
                                                value=chunk["text"],
                                                label=f"Chunk {i + 1} (Score: {chunk_score_fmt})",
                                                interactive=False,
                                                lines=5,
                                                max_lines=10,
                                                elem_classes="chunk-text",
                                            )

        submit_btn.click(
            fn=predict,
            inputs=[query_input],
            outputs=[search_results],
        )

        query_input.submit(
            fn=predict,
            inputs=[query_input],
            outputs=[search_results],
        )

    demo.launch()


if __name__ == "__main__":
    main()
