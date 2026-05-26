import gradio as gr
from model import BM25Model, BoWModel, LSAModel


def main():
    def predict(query, model) -> list[list]:
        results = []
        top_n = 50

        if model == "bm25":
            results = BM25Model.query(phrase=query, top_n=top_n)
        elif model == "lsa":
            results = LSAModel.query(phrase=query, top_n=top_n)
        elif model == "bow":
            results = BoWModel.query(phrase=query, top_n=top_n)

        return [
            [
                f'<a href="{response.url}" target="_blank">{response.title}</a>',
                response.score,
            ]
            for response in results
        ]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Query")
                model_input = gr.Radio(
                    label="Search approach", choices=["bm25", "lsa", "bow"]
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            output_df = gr.Dataframe(
                label="Found articles",
                headers=["title", "score"],
                datatype=["html", "number"],
            )

        submit_btn.click(
            fn=predict, inputs=[query_input, model_input], outputs=output_df
        )

        query_input.submit(
            fn=predict, inputs=[query_input, model_input], outputs=output_df
        )

        clear_btn.click(
            fn=lambda: ("", None, None),
            inputs=[],
            outputs=[query_input, model_input, output_df],
        )

    demo.launch()


if __name__ == "__main__":
    main()
