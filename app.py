import gradio as gr
from quiz import start_quiz, answer_question, format_question, parse_quiz, create_quiz
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from functools import partial

# Load model
model_path = hf_hub_download(
    repo_id="ebbalg/llama-finetome",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf"
)
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False, chat_format="llama-3")

# Generate and parse quiz
raw_quiz = create_quiz(llm)
parsed_quiz = parse_quiz(raw_quiz)

with gr.Blocks(title="TAI: AI Teacher Assistant") as demo:
    gr.Markdown("# TAI: Your AI Teacher Assistant\nTest your ML knowledge with a quiz!")

    start_btn = gr.Button("Start Quiz", variant="primary")
    question_md = gr.Markdown("")
    feedback_md = gr.Markdown("")
    progress_md = gr.Markdown("")

    # Answer buttons
    with gr.Row():
        btn_A = gr.Button("A", visible=False)
        btn_B = gr.Button("B", visible=False)
        btn_C = gr.Button("C", visible=False)
        btn_D = gr.Button("D", visible=False)
    retry_btn = gr.Button("Retry", visible=False)

    # States
    idx_state = gr.State(0)
    score_state = gr.State(0)

    # Start quiz
    def start_quiz_ui():
        q_text, feedback, progress, idx, score = start_quiz(parsed_quiz)
        return (
            q_text, feedback, progress, idx, score,
            gr.update(visible=False),  # hide start
            gr.update(visible=True),   # show buttons
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True)    # show retry
        )

    start_btn.click(
        fn=start_quiz_ui,
        inputs=[],
        outputs=[question_md, feedback_md, progress_md, idx_state, score_state,
                 start_btn, btn_A, btn_B, btn_C, btn_D, retry_btn]
    )

    # Answer buttons
    for letter, btn in zip(["A", "B", "C", "D"], [btn_A, btn_B, btn_C, btn_D]):
        btn_fn = partial(answer_question, parsed_quiz, letter)
        btn.click(
            fn=btn_fn,
            inputs=[idx_state, score_state],
            outputs=[question_md, idx_state, score_state, feedback_md, progress_md]
        )

    # Retry
    def retry_quiz_ui():
        q_text, feedback, progress, idx, score = start_quiz(parsed_quiz)
        return (
            q_text, feedback, progress, idx, score,
            gr.update(visible=True),   # show start
            gr.update(visible=False),  # hide buttons
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)   # hide retry
        )

    retry_btn.click(
        fn=retry_quiz_ui,
        inputs=[],
        outputs=[question_md, feedback_md, progress_md, idx_state, score_state,
                 start_btn, btn_A, btn_B, btn_C, btn_D, retry_btn]
    )

demo.launch()
