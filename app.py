import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ebbalg/llama-finetome")
model = AutoModelForCausalLM.from_pretrained(
    "ebbalg/llama-finetome",
    torch_dtype=torch.float32,
    device_map="cpu",
)

def chat(message, history):
    """Chat function with conversation history"""

    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Generate
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    # Decode only new tokens
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

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

demo.launch()
