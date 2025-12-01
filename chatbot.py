
def chat(llm, message, history):
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