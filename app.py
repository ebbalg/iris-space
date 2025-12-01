import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

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

def chat(message, history):
    """Simple chat function"""
    messages = []
    
    # Handle history - it might be a list of tuples or list of dicts
    if history:
        for h in history:
            if isinstance(h, (list, tuple)):
                # Format: [user_msg, assistant_msg]
                messages.append({"role": "user", "content": str(h[0])})
                messages.append({"role": "assistant", "content": str(h[1])})
            elif isinstance(h, dict):
                # Format: {"role": "...", "content": "..."}
                messages.append(h)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    
    return response["choices"][0]["message"]["content"]

def quiz():
    '''Generate a multiple choice quiz with 10 questions (created by the llm)'''
    system_prompt = {
        "role": "system",
        "content": (
            "Generate a set of 10 multiple-choice questions about machine learning for a student."
            "Each question should have 4 answer options (A–D) with a single correct answer.\n\n"
            "Format exactly like this:\n\n"
            "1. Question text...\n"
            "A) ...\nB) ...\nC) ...\nD) ...\n**Correct Answer: X**\n\n"
        )
    }

    response = llm.create_chat_completion(
        messages=[system_prompt],
        max_tokens=512,
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]




with gr.Blocks(title="TAI: AI Teacher Assistant") as demo:
    gr.Markdown("""
    # TAI: Your AI Teacher Assistant
    Ask questions about AI and Machine Learning! I can help you better understand the
    theoretical and practical skills to succeed in this field. Test your understanding with a quiz!
                
    This Llama 3.2 1B model was fine-tuned on the FineTome-100k instruction dataset.
    """)
    with gr.Row():

        # Left column: chat
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                chat,
                examples=[
                    "Which tasks can recurrent neural networks address?",
                    "Explain backpropagation in simple terms.",
                    "What are the benefits of regularization during training?",
                    "Write a short summary of advancements that allowed for deep neural networks to work."
                ]
            )
        
        # Right column: quiz
        with gr.Column(scale=1):
            gr.Markdown("## Machine Learning Quiz")
            quiz_btn = gr.Button("Start Quiz", variant="primary")
            quiz_output = gr.Markdown()

            quiz_btn.click(fn=quiz, outputs=quiz_output)


demo.launch()
