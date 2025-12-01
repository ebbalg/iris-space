import re
import gradio as gr
#from quiz import create_quiz, parse_quiz, start_quiz, submit_and_next, restart_quiz
from quiz import *
from chatbot import chat
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from functools import partial

print("Downloading model...")
model_path = hf_hub_download(
    repo_id="ebbalg/llama-finetome",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf"
)

print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    verbose=False,
    chat_format="llama-3"
)

print("Creating quiz...")
raw_quiz = create_quiz(llm)
parsed_quiz = parse_quiz(raw_quiz)



with gr.Blocks(title="TAI: AI Teacher Assistant") as demo:
    gr.Markdown("""
    # TAI: Your AI Teacher Assistant
    Ask questions about AI and Machine Learning! I can help you better understand the
    theoretical and practical skills to succeed in this field. Test your understanding with a quiz!
                
    This Llama 3.2 1B model was fine-tuned on the FineTome-100k instruction dataset.
    """)

    initial_quiz = create_quiz(llm)
    quiz_state = gr.State(initial_quiz)


    with gr.Row():

        # Left column: chat
        with gr.Column(scale=1):
            chatbot = gr.ChatInterface(
                fn=lambda message, history: chat(llm, message, history),
                examples=[
                    "Which tasks can recurrent neural networks address?",
                    "Explain backpropagation in simple terms.",
                    "What are the benefits of regularization during training?",
                    "Write a short summary of advancements that allowed for deep neural networks to work."
                ]
            )
        
        # Right column: quiz
        with gr.Column(scale=2):
            gr.Markdown("## Test Yourself")
            
            start_btn = gr.Button("Start Quiz", variant="primary")

            question_md = gr.Markdown("")
            feedback_md = gr.Markdown("")
            progress_md = gr.Markdown("")

            # Answer buttons
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
                q_data = parsed_quiz[0]
                q_text = q_data["q"]
                options = q_data["options"]  # list of 4 strings for A-D

                return (
                    q_text,
                    "",  # feedback
                    f"1/{len(parsed_quiz)}",  # progress
                    0,  # idx
                    0,  # score
                    gr.update(visible=False),   # hide start button
                    gr.update(visible=True, value=f"A: {options[0]}"),  # btn_A
                    gr.update(visible=True, value=f"B: {options[1]}"),  # btn_B
                    gr.update(visible=True, value=f"C: {options[2]}"),  # btn_C
                    gr.update(visible=True, value=f"D: {options[3]}"),  # btn_D
                    gr.update(visible=True)  # retry
                )

            start_btn.click(
                fn=start_quiz_ui,
                inputs=[],
                outputs=[question_md, feedback_md, progress_md, idx_state, score_state,
                         start_btn, btn_A, btn_B, btn_C, btn_D, retry_btn]
            )

            # Answer button clicks
            for letter, btn in zip(["A", "B", "C", "D"], [btn_A, btn_B, btn_C, btn_D]):
                btn_fn = partial(answer_question, parsed_quiz, letter)
                btn.click(
                    fn=btn_fn,
                    inputs=[idx_state, score_state],
                    outputs=[question_md, idx_state, score_state, feedback_md, progress_md,
                            btn_A, btn_B, btn_C, btn_D]
                )


            # Retry button
            def retry_quiz_ui():
                q_text, feedback, progress, idx, score, _ = start_quiz(parsed_quiz)
                return (
                    q_text, feedback, progress, idx, score,
                    gr.update(visible=True),    # show start
                    gr.update(visible=False),   # hide A
                    gr.update(visible=False),   # hide B
                    gr.update(visible=False),   # hide C
                    gr.update(visible=False),   # hide D
                    gr.update(visible=False)    # hide retry
                )

            retry_btn.click(
                fn=retry_quiz_ui,
                inputs=[],
                outputs=[question_md, feedback_md, progress_md, idx_state, score_state,
                         start_btn, btn_A, btn_B, btn_C, btn_D, retry_btn]
            )


demo.launch()
