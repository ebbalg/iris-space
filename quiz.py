import re
import gradio as gr

### CREATE AND PARSE QUIZ

def create_quiz(llm):
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

def parse_quiz(text):
    question_blocks = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)', text, flags=re.S)
    parsed = []
    for idx, block_text in question_blocks:
        block = block_text.strip()
        # Extract the question line before options
        # Find A) position
        m_a = re.search(r'\nA[\)\.]', block)
        if m_a:
            question_text = block[:m_a.start()].strip()
            rest = block[m_a.start():].strip()
        else:
            # if no A) found, entire block as question
            question_text = block
            rest = ""

        # Extract options A-D
        opts = re.findall(r'([A-D])[\)\.]\s*(.+?)(?=\n[A-D][\)\.]|\n\*\*Correct|\n\d+\.|\Z)', rest, flags=re.S)
        options = []
        # sort options by letter just in case
        opts_sorted = sorted(opts, key=lambda x: x[0]) if opts else []
        for letter, opt_text in opts_sorted:
            options.append(opt_text.strip().replace("\n", " "))

        # Extract correct answer
        ans = None
        m_corr = re.search(r'\*\*\s*Correct Answer\s*:\s*([A-D])\s*\*\*', block, flags=re.I)
        if not m_corr:
            m_corr = re.search(r'Correct Answer\s*:\s*([A-D])', block, flags=re.I)
        if not m_corr:
            # possible "Correct: A" or "Answer: A"
            m_corr = re.search(r'(Correct|Answer)\s*[:\-]\s*([A-D])', block, flags=re.I)
            if m_corr:
                ans = m_corr.group(2).upper()
        else:
            ans = m_corr.group(1).upper()

        parsed.append({
            "q": question_text,
            "options": options,
            "answer": ans
        })

    return parsed

def start_quiz(quiz_parsed):
    """Initialize quiz state, return first question display."""
    idx = 0
    q = quiz_parsed[idx]
    return q["q"], "", f"{idx+1}/{len(quiz_parsed)}", idx, 0, False

def answer_question(parsed_quiz, selected, idx, score):
    current = parsed_quiz[idx]
    correct = current["answer"]
    if selected.upper() == correct:
        score += 1
        idx += 1
        if idx >= len(parsed_quiz):
            return (
                "🎉 Quiz Complete!",
                idx,
                score,
                "✅ Correct!",
                f"{len(parsed_quiz)}/{len(parsed_quiz)}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        # next question
        next_q = parsed_quiz[idx]
        options = next_q["options"]
        return (
            next_q["q"],
            idx,
            score,
            "✅ Correct!",
            f"{idx+1}/{len(parsed_quiz)}",
            gr.update(value=f"A: {options[0]}"),
            gr.update(value=f"B: {options[1]}"),
            gr.update(value=f"C: {options[2]}"),
            gr.update(value=f"D: {options[3]}")
        )
    else:
        # incorrect → keep question and options the same
        options = current["options"]
        return (
            current["q"],
            idx,
            score,
            "❌ Incorrect, try again.",
            f"{idx+1}/{len(parsed_quiz)}",
            gr.update(value=f"A: {options[0]}"),
            gr.update(value=f"B: {options[1]}"),
            gr.update(value=f"C: {options[2]}"),
            gr.update(value=f"D: {options[3]}")
        )

def retry_quiz():
    return start_quiz()

