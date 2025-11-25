import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Downloing model from Huggingface
print("Downloading model...")
model_path = hf_hub_download(
    repo_id="ebbalg/llama-finetome",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf"
)

# Loading model
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        # Context window
    n_threads=2,       # CPU threads
    verbose=False
)

def chat(message, history):
    """
    Chat function that maintains conversation history
    
    Args:
        message: The user's current message
        history: List of [user_msg, bot_msg] pairs from previous turns
    
    Returns:
        The model's response
    """
    # Build the conversation history in the format the model expects
    messages = []
    
    # Add previous conversation turns
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the current user message
    messages.append({"role": "user", "content": message})
    
    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    
    return response["choices"][0]["message"]["content"]

# Creating interface
demo = gr.ChatInterface(
    chat,
    title="Llama 3.2 Fine-tuned Chatbot",
    description="""
    This is a Llama 3.2 1B model fine-tuned on the FineTome-100k instruction dataset.
    Try asking questions, requesting explanations, or having a conversation!
    """,
    examples=[
        "Explain what machine learning is in simple terms",
        "Write a short poem about autumn",
        "What are the benefits of exercise?",
        "Help me understand recursion in programming"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
