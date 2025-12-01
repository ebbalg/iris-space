import re
from chatbot import llm

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


def start_quiz(quiz_raw, quiz_parsed):
    """Return first question payload and initialize index & score."""
    # If parse failed or zero questions, show raw text
    if not quiz_parsed:
        return {
            "question_md": "⚠️ Could not parse generated quiz. Showing raw output below:\n\n" + quiz_raw,
            "options": [],
            "progress": "0/0"
        }, 0, 0  # index, score

    idx = 0
    q = quiz_parsed[idx]
    opts = q["options"]
    # If options are empty, present the whole block as markdown
    if not opts:
        return {
            "question_md": f"**Q {idx+1}.** {q['q']}",
            "options": [],
            "progress": f"{idx+1}/{len(quiz_parsed)}"
        }, idx, 0

    # present options as list of strings
    return {
        "question_md": f"**Q {idx+1}.** {q['q']}",
        "options": [f"{letter}) {opt}" for letter, opt in zip(['A','B','C','D'], opts)],
        "progress": f"{idx+1}/{len(quiz_parsed)}"
    }, idx, 0

def show_question(quiz_parsed, idx):
    """Return the question (md), options list, progress for a given index."""
    if not quiz_parsed or idx < 0 or idx >= len(quiz_parsed):
        return "No question", [], "0/0"
    q = quiz_parsed[idx]
    opts = q["options"]
    md = f"**Q {idx+1}.** {q['q']}"
    if not opts:
        return md, [], f"{idx+1}/{len(quiz_parsed)}"
    return md, [f"{letter}) {opt}" for letter, opt in zip(['A','B','C','D'], opts)], f"{idx+1}/{len(quiz_parsed)}"

def submit_and_next(selected, quiz_parsed, idx, score):
    """
    Process user's selected option for question idx, return:
    - question text for next idx or results message if finished
    - options list for next question
    - updated index, updated score
    - status/progress string
    """
    # Validate
    if not quiz_parsed:
        return "No quiz parsed.", [], idx, score, "0/0", False

    # Check current answer
    if 0 <= idx < len(quiz_parsed):
        current = quiz_parsed[idx]
        correct = current.get("answer")
        # normalize selected like "A) text" -> "A"
        sel_letter = None
        if selected:
            m = re.match(r'\s*([A-D])[\)\.]', selected)
            if m:
                sel_letter = m.group(1).upper()
        # If there is a correct letter available, compare
        if correct and sel_letter:
            if sel_letter == correct:
                score += 1

    # Move to next question
    idx += 1
    if idx >= len(quiz_parsed):
        # Quiz finished
        result_md = f"### 🧾 Quiz complete!\n\nYour score: **{score} / {len(quiz_parsed)}**"
        # Optionally show correct answers summary
        answers_lines = []
        for i, q in enumerate(quiz_parsed):
            ans = q.get("answer") or "?"
            answers_lines.append(f"{i+1}. {ans}")
        result_md += "\n\n**Correct answers:**\n\n" + "\n".join(answers_lines)
        return result_md, [], idx, score, f"{len(quiz_parsed)}/{len(quiz_parsed)}", True

    # Otherwise return next question
    md, opts, progress = show_question(quiz_parsed, idx)
    return md, opts, idx, score, progress, False

def restart_quiz(raw_quiz, parsed_quiz):
    """Reset to the first question, keep the same parsed quiz."""
    payload, idx, score = start_quiz(raw_quiz, parsed_quiz)
    return payload["question_md"], payload["options"], idx, score, payload["progress"]