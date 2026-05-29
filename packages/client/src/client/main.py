from typing import Any

import gradio as gr
from model import BM25Model, BoWModel, LSAModel


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
    def predict(query, model) -> list[dict[str, Any]]:
        if not query:
            return []

        top_n = 50

        if model == "bm25":
            return format_results(BM25Model.query(phrase=query, top_n=top_n))
        elif model == "lsa":
            return format_results(LSAModel.query(phrase=query, top_n=top_n))
        elif model == "bow":
            return format_results(BoWModel.query(phrase=query, top_n=top_n))

        return []

    with gr.Blocks(theme=gr.themes.Default()) as demo:
        search_results = gr.State({})

        with gr.Column():
            with gr.Column():
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Search articles...",
                        show_label=False,
                        lines=1,
                        max_lines=1,
                        scale=5,
                        container=False,
                        min_width=100,
                    )

                    submit_btn = gr.Button("Submit", variant="primary")
                with gr.Row():
                    model = gr.Radio(choices=["bm25", "lsa", "bow"])

            @gr.render(inputs=[search_results, model])
            def render_articles(results, model):
                if not results:
                    return

                with gr.Column():
                    if not results:
                        gr.Markdown("\\(* _ *)/")

                    for article in results:
                        best_score = max(
                            chunk["score"] for chunk in article["documents"]
                        )

                        if model in ["lsa", "bow"]:
                            best_score_display = f"{best_score * 100:.1f}%"
                            color = get_color_for_score(best_score)
                        else:
                            best_score_display = f"{best_score:.4f}"
                            color = "inherit"

                        with gr.Row():
                            with gr.Column():
                                gr.HTML(f"""
                                    <div style="display: flex; align-items: center; justify-content: space-between; padding-top: 32px;">
                                        <h2 style="margin: 0;"><a href="{article["url"]}" target="_blank" style="padding: 0;">{article["title"]}</a></h2>
                                        <div style="color: {color};">
                                            {best_score_display}
                                        </div>
                                    </div>
                                """)

                                gr.HTML(
                                    f"<p>{article['documents'][0]['text'][:512] + '...'}</p>"
                                )

                                with gr.Accordion(
                                    f"View {len(article['documents'])} matched chunks",
                                    open=False,
                                ):
                                    for i, chunk in enumerate(article["documents"]):
                                        chunk_score_fmt = (
                                            f"{chunk['score'] * 100:.1f}%"
                                            if model in ["lsa", "bow"]
                                            else f"{chunk['score']:.4f}"
                                        )

                                        gr.HTML(
                                            f"<div><div style='padding: 32px 0;'>chunk {i}, score {chunk_score_fmt}</div><p>{chunk['text'][:512] + '...'}</p></div>",
                                        )

        submit_btn.click(
            fn=predict,
            inputs=[query_input, model],
            outputs=[search_results],
        )

        query_input.submit(
            fn=predict,
            inputs=[query_input, model],
            outputs=[search_results],
        )

    demo.launch()


if __name__ == "__main__":
    main()
