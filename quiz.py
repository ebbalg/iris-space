import re
import gradio as gr
import time

### CREATE AND PARSE QUIZ

def create_quiz(llm):
    # system_prompt = {
    # "role": "system",
    # "content": (
    #     "Generate exactly 10 multiple-choice questions about machine learning.\n"
    #     "Each question must have 4 options (A-D) and exactly one correct answer.\n\n"
    #     "Start each question like 'QUESTION 1' and then continue with the four options like this:"
    #     "A: <option A>, B: <option B>, C: <option C>, D: <option D> but replace each <option> "
    #     "with the generated possible answers. At the end write 'ANSWER:' and the answer (A, B, C or D)."
    #     "Repeat the same format for QUESTIONS 2 through 10."
    # )

    system_prompt = {
        "role": "system",
        "content": (
            "Generate exactly 10 multiple-choice questions about machine learning.\n"
            "Each question must have 4 possible answers (A-D), where only one is the correct answer.\n\n"
            "Use this format strictly, with line breaks:\n\n"
            "QUESTION 1\n"
            "<question>\n"
            "OPTION A: <option>\n"
            "OPTION B: <option>\n"
            "OPTION C: <option>\n"
            "OPTION D: <option>\n"
            "ANSWER: <letter>\n"
            "END\n\n"
            "Repeat the same format for QUESTIONS 2 through 10, replacing placeholders with generated questions and answers."
        )
}


    response = llm.create_chat_completion(
        messages=[system_prompt],
        max_tokens=1024,
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]



def parse_quiz(text):
    """
    Parse quiz in inline format:
    QUESTION N <question text> A) <option> B) <option> C) <option> D) <option> ANSWER: <letter> END
    """
    parsed = []

    # Split by QUESTION N
    blocks = re.split(r'QUESTION \d+', text)
    blocks = [b.strip() for b in blocks if b.strip()]

    for block in blocks:
        # Extract question text (everything before 'A)')
        q_match = re.match(r'(.+?)\s+A\)', block)
        question_text = q_match.group(1).strip() if q_match else "Question missing"

        # Extract options using regex for A)-D)
        options = []
        for letter in ['A', 'B', 'C', 'D']:
            opt_match = re.search(rf'{letter}\)\s*(.+?)(?=\s+[A-D]\)|\s+ANSWER:)', block)
            options.append(opt_match.group(1).strip() if opt_match else "")

        # Extract correct answer
        answer_match = re.search(r'ANSWER:\s*([A-D])', block)
        correct = answer_match.group(1).upper() if answer_match else None

        parsed.append({
            "q": question_text,
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
            return question_text, idx, score, "", progress  # feedback cleared at end
        current = parsed_quiz[idx]
        # Clear feedback for the next question
        feedback = ""
    else:
        feedback = "❌ Incorrect, try again."
        time.sleep(1)
        feedback = ""

    question_text = format_question(current)
    progress = f"{idx+1}/{len(parsed_quiz)}"

    return question_text, idx, score, feedback, progress



def retry_quiz():
    return start_quiz()

