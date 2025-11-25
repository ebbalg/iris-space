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
    # Convert history to the format llama-cpp-python expects
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    
    return response["choices"][0]["message"]["content"]

demo = gr.ChatInterface(
    chat,
    title="🦙 Llama 3.2 Fine-tuned Chatbot",
    description="Llama 3.2 1B fine-tuned on FineTome-100k",
)

demo.launch()
