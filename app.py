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
        history: List of previous conversation turns
    
    Returns:
        The model's response
    """

    messages = []
    
    # Add previous conversation turns
    # pass history as a list of messages
    if history:
        for turn in history:
            try:
                # Try different unpacking formats
                if isinstance(turn, (list, tuple)):
                    if len(turn) >= 2:
                        user_msg, assistant_msg = turn[0], turn[1]
                        messages.append({"role": "user", "content": str(user_msg)})
                        messages.append({"role": "assistant", "content": str(assistant_msg)})
                elif isinstance(turn, dict):
                    messages.append(turn)

    # Adding current user message
    messages.append({"role": "user", "content": str(message)})
    
    # Generating response
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
