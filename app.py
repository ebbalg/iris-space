import re
import gradio as gr
from quiz import create_quiz, parse_quiz, start_quiz, submit_and_next, restart_quiz
from chatbot import chat
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

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
             # Visible elements
            progress = gr.Markdown(f"**Progress:** 0/0", elem_id="quiz-progress")
            question_md = gr.Markdown("", elem_id="quiz-question")
            options = gr.Radio(choices=[], label="Choose an answer", type="value")
            next_btn = gr.Button("Next")
            start_btn = gr.Button("Start Quiz", variant="primary")
            restart_btn = gr.Button("Restart Quiz")
            result_md = gr.Markdown("", visible=False)

            # State holders
            quiz_raw_state = gr.State(raw_quiz)        # raw quiz text
            quiz_parsed_state = gr.State(parsed_quiz)       # parsed list of questions
            idx_state = gr.State(0)                         # current index
            score_state = gr.State(0)                       # current score
            finished_state = gr.State(False)                # finished flag

            # Start Quiz -> show first question (no model call)
            def on_start(quiz_raw, quiz_parsed):
                payload, idx, score = start_quiz(quiz_raw, quiz_parsed)
                # return question_md, options choices, idx, score, progress, result_md_visible
                return payload["question_md"], payload["options"], gr.update(value=idx), gr.update(value=score), payload["progress"], gr.update(visible=False), gr.update(visible=True)
                # Note: last two are for hiding/showing result_md (kept simple)

            start_btn.click(
                fn=lambda qr, qp: start_quiz(qr, qp)[0],  # we only want the payload (question data) but mapping below
                inputs=[quiz_raw_state, quiz_parsed_state],
                outputs=[question_md]
            )

            # After start, we also need to set options, progress, idx, score and ensure result is hidden.
            def start_full(quiz_raw, quiz_parsed):
                payload, idx, score = start_quiz(quiz_raw, quiz_parsed)
                return payload["question_md"], payload["options"], idx, score, payload["progress"], "", False

            start_btn.click(
                fn=start_full,
                inputs=[quiz_raw_state, quiz_parsed_state],
                outputs=[question_md, options, idx_state, score_state, progress, result_md, finished_state]
            )

            # Next button: check selected and move forward
            def on_next(selected, quiz_parsed, idx, score):
                md_or_q, opts, new_idx, new_score, prog, finished = submit_and_next(selected, quiz_parsed, idx, score)
                if finished:
                    # show final result
                    return md_or_q, [], new_idx, new_score, prog, True
                else:
                    return md_or_q, opts, new_idx, new_score, prog, False

            next_btn.click(
                fn=on_next,
                inputs=[options, quiz_parsed_state, idx_state, score_state],
                outputs=[question_md, options, idx_state, score_state, progress, finished_state]
            )

            # When finished_state becomes True, show the result markdown and hide options
            def show_result_if_finished(finished_flag, current_question_md):
                if finished_flag:
                    # current_question_md will be the result text returned earlier
                    return gr.update(value=current_question_md), gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(value=""), gr.update(visible=False), gr.update(visible=True)

            finished_state.change(
                fn=show_result_if_finished,
                inputs=[finished_state, question_md],
                outputs=[result_md, result_md, options]  # show result_md & hide options when finished
            )

            # Restart quiz (without regenerating)
            def on_restart():
                q_md, q_opts, idx, score, prog = restart_quiz(raw_quiz, parsed_quiz)
                return q_md, q_opts, idx, score, prog, "", False

            restart_btn.click(
                fn=on_restart,
                inputs=[],
                outputs=[question_md, options, idx_state, score_state, progress, result_md, finished_state]
            )


demo.launch()
