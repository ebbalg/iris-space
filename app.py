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
        with gr.Column(scale=2):
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
        with gr.Column(scale=1):
            gr.Markdown("## Test Yourself")
            
            start_btn = gr.Button("Start Quiz", variant="primary")

            question_md = gr.Markdown("")
            feedback_md = gr.Markdown("")
            progress_md = gr.Markdown("")

            # Answer buttons
            btn_A = gr.Button("A")
            btn_B = gr.Button("B")
            btn_C = gr.Button("C")
            btn_D = gr.Button("D")
            retry_btn = gr.Button("Retry")

            # States
            idx_state = gr.State(0)
            score_state = gr.State(0)

            # Start quiz
            start_fn = partial(start_quiz, parsed_quiz)

            start_btn.click(
                fn=start_fn,
                inputs=[],
                outputs=[question_md, idx_state, score_state, feedback_md, progress_md, start_btn]
            )

            # Answer button clicks
            answer_fn_A = partial(answer_question, parsed_quiz, "A")
            answer_fn_B = partial(answer_question, parsed_quiz, "B")
            answer_fn_C = partial(answer_question, parsed_quiz, "C")
            answer_fn_D = partial(answer_question, parsed_quiz, "D")

            btn_A.click(fn=answer_fn_A, inputs=[idx_state, score_state],
                        outputs=[question_md, idx_state, score_state, feedback_md, progress_md])
            btn_B.click(fn=answer_fn_B, inputs=[idx_state, score_state],
                        outputs=[question_md, idx_state, score_state, feedback_md, progress_md])
            btn_C.click(fn=answer_fn_C, inputs=[idx_state, score_state],
                        outputs=[question_md, idx_state, score_state, feedback_md, progress_md])
            btn_D.click(fn=answer_fn_D, inputs=[idx_state, score_state],
                        outputs=[question_md, idx_state, score_state, feedback_md, progress_md])

            # Retry button
            retry_btn.click(
                fn=retry_quiz,
                inputs=[],
                outputs=[question_md, idx_state, score_state, feedback_md, progress_md, start_btn]
            )


demo.launch()
