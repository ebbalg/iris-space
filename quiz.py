import re
import gradio as gr
import time

### CREATE AND PARSE QUIZ

def create_quiz(llm):
    system_prompt = {
        "role": "system",
        "content": (
            "Generate exactly 10 machine-learning multiple choice questions.\n"
            "FOLLOW THIS FORMAT EXACTLY. DO NOT add markdown, bullets, quotes, headings, or commentary.\n"
            "Spaces and punctuation MUST match exactly.\n\n"

            "QUESTION 1\n"
            "Question text.\n"
            "OPTION A: Text\n"
            "OPTION B: Text\n"
            "OPTION C: Text\n"
            "OPTION D: Text\n"
            "ANSWER: A\n"
            "END\n\n"

            "Repeat for QUESTION 2 through QUESTION 10.\n"
            "DO NOT add blank lines except exactly where shown.\n"
            "DO NOT change OPTION labels.\n"
        )
    }

    response = llm.create_chat_completion(
        messages=[system_prompt],
        max_tokens=1024,
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]



def parse_quiz(text):
    # Normalize newlines
    text = text.replace("\r", "")

    # Split per question block using regex
    blocks = re.split(r"QUESTION\s+\d+\s*", text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    parsed = []

    for block in blocks:
        # Extract question (first non-option, non-answer line)
        lines = [l.strip() for l in block.splitlines() if l.strip()]

        question_text = None
        for l in lines:
            if not l.upper().startswith("OPTION") and not l.upper().startswith("ANSWER"):
                question_text = l
                break

        # Extract OPTIONS (fallback to empty string if missing)
        def find_opt(prefix):
            m = re.search(prefix + r"\s*:\s*(.*)", block, flags=re.IGNORECASE)
            return m.group(1).strip() if m else ""

        options = [
            find_opt(r"OPTION A"),
            find_opt(r"OPTION B"),
            find_opt(r"OPTION C"),
            find_opt(r"OPTION D"),
        ]

        # Extract ANSWER
        m = re.search(r"ANSWER\s*:\s*([A-D])", block, flags=re.IGNORECASE)
        correct = m.group(1).upper() if m else "A"

        parsed.append({
            "q": question_text or "Question missing",
            "options": options,
            "answer": correct
        })

    return parsed



def format_question(q_dict):
    lines = [q_dict["q"], ""]  # blank line
    for letter, option in zip(["A","B","C","D"], q_dict["options"]):
        lines.append(f"{letter}: {option}")
    return "<br>".join(lines)

def start_quiz(parsed_quiz):
    """Initialize quiz state, return first question display."""
    q_data = parsed_quiz[0]
    q_text = format_question(q_data)
    return q_text, "", f"{1}/{len(parsed_quiz)}", 0, 0  # question_md, feedback, progress, idx, score


def answer_question(parsed_quiz, selected, idx, score):
    current = parsed_quiz[idx]
    correct = current["answer"]
    
    if selected.upper() == correct:
        score += 1
        feedback = "✅ Correct!"
        # Move to next question after 1 second
        time.sleep(1)
        idx += 1
        if idx >= len(parsed_quiz):
            question_text = "🎉 Quiz Complete!"
            progress = f"{len(parsed_quiz)}/{len(parsed_quiz)}"
            return question_text, idx, score, feedback, progress
        current = parsed_quiz[idx]
    else:
        feedback = "❌ Incorrect, try again."

    question_text = format_question(current)
    progress = f"{idx+1}/{len(parsed_quiz)}"

    return question_text, idx, score, feedback, progress


def retry_quiz():
    return start_quiz()

