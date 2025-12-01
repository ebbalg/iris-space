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

def format_question(q_dict):
    """Return question text + options as a string for display"""
    options_text = "\n".join([f"{letter}: {text}" for letter, text in zip(["A","B","C","D"], q_dict["options"])])
    return f"**Q:** {q_dict['q']}\n\n{options_text}"

def start_quiz(quiz_parsed):
    idx = 0
    q = quiz_parsed[idx]
    q_text = format_question(q)
    return q_text, "", f"{idx+1}/{len(quiz_parsed)}", idx, 0, False

def answer_question(parsed_quiz, selected, idx, score):
    current = parsed_quiz[idx]
    correct = current["answer"]
    if selected.upper() == correct:
        score += 1
        feedback = "✅ Correct!"
        # Optionally: move to next question here or keep same question
        idx += 1 if idx + 1 < len(parsed_quiz) else idx
    else:
        feedback = "❌ Incorrect, try again."

    q_text = format_question(current if idx < len(parsed_quiz) else {"q": "Quiz Complete!", "options": ["","","",""]})
    progress = f"{min(idx+1,len(parsed_quiz))}/{len(parsed_quiz)}"
    return q_text, idx, score, feedback, progress


def retry_quiz():
    return start_quiz()

