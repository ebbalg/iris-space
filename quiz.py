import re
import gradio as gr
import time

### CREATE AND PARSE QUIZ

def create_quiz(llm):
    """Generate a multiple-choice quiz with 10 questions in a strict parseable format"""
    system_prompt = {
        "role": "system",
        "content": (
            "Generate 10 multiple-choice questions about machine learning for a student.\n"
            "Each question must have 4 options (A-D) and exactly one correct answer.\n\n"
            "Use the exact format below, no extra text, no markdown:\n\n"
            "QUESTION 1\n"
            "Question text...\n\n"
            "OPTION A: ...\n"
            "OPTION B: ...\n"
            "OPTION C: ...\n"
            "OPTION D: ...\n"
            "ANSWER: X\n"
            "END\n\n"
            "Repeat for QUESTIONS 2 through 10."
        )
    }

    response = llm.create_chat_completion(
        messages=[system_prompt],
        max_tokens=1024,
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]


def parse_quiz(text):
    parsed = []

    # Split the text by 'QUESTION ' to get each question block
    blocks = [b.strip() for b in text.split("QUESTION ") if b.strip()]
    for block in blocks:
        # First line is the question number, we can ignore it
        lines = block.splitlines()
        # Remove empty lines
        lines = [l.strip() for l in lines if l.strip()]
        if not lines:
            continue

        # Question text is the first line after number
        question_text = lines[1] if lines[0].isdigit() else lines[0]

        # Parse options
        options = []
        for line in lines:
            if line.startswith("OPTION "):
                # Example: "OPTION A: To learn from labeled data"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    options.append(parts[1].strip())

        # Parse correct answer
        correct_line = next((l for l in lines if l.startswith("ANSWER:")), None)
        if correct_line:
            correct = correct_line.split(":", 1)[1].strip().upper()
        else:
            correct = None

        parsed.append({
            "q": question_text,
            "options": options,
            "answer": correct
        })

    return parsed


def format_question(q_dict):
    """Return question text + options as a string for display in Markdown"""
    lines = [q_dict["q"] + "<br>"]
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

