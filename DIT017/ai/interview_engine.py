import os
import json
from google import genai

from .interview_questions import get_fallback_question


def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as error:
        print("Gemini client error:", error)
        return None


def create_interview(interview_id, interview_type, details):
    return {
        "id": interview_id,
        "type": interview_type,
        "details": details,
        "questions": [],
        "max_questions": 15,
        "completed": False,
        "result": None,
        "audio_file": None,
        "question_pool": [],
        "gemini_failures": 0
    }


def get_candidate_details(session):
    details = session.get("details", {})

    cleaned = {}

    for key, value in details.items():
        if value is not None:
            cleaned[key] = str(value).strip()

    return cleaned


def build_gemini_prompt(session):
    interview_type = session.get("type", "technical")
    details = get_candidate_details(session)

    previous_questions = [
        item.get("question", "")
        for item in session.get("questions", [])
        if item.get("question")
    ]

    details_text = json.dumps(
        details,
        indent=2,
        ensure_ascii=False
    )

    previous_text = "\n".join(
        f"- {question}"
        for question in previous_questions
    )

    if not previous_text:
        previous_text = "No questions have been asked yet."

    if interview_type == "technical":
        focus = """
Generate a technical interview question.

The question should be related to the candidate's technical
skills, programming languages, projects, databases, technologies,
experience, or the role they entered.

If a project is available, prefer asking about that project.
If a programming language or technology is available, you may
ask about its practical use.
"""
    else:
        focus = """
Generate an HR interview question.

The question should be relevant to the candidate's role,
career goal, strengths, improvement areas, teamwork, leadership,
achievements, interests, or experience.

Prefer questions that make the candidate explain their real
experience rather than generic textbook questions.
"""

    return f"""
You are an AI interviewer for CareerForge AI.

Interview type:
{interview_type}

Candidate details:
{details_text}

{focus}

Rules:
1. Generate exactly ONE interview question.
2. Do not answer the question.
3. Do not include numbering.
4. Do not include labels such as "Question:".
5. Do not repeat any previous question.
6. Keep the question natural and suitable for a real interview.
7. Personalize the question using the candidate details whenever possible.
8. Do not invent candidate information.
9. The question should be different from previous questions.

Previous questions:
{previous_text}

Return ONLY the question text.
""".strip()


def clean_gemini_question(text):
    if not text:
        return None

    question = str(text).strip()

    question = question.replace("Question:", "")
    question = question.replace("QUESTION:", "")
    question = question.strip()

    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1].strip()

    if not question:
        return None

    return question


def ask_gemini(session, client, model_name):
    if client is None:
        raise RuntimeError("Gemini client is not available.")

    prompt = build_gemini_prompt(session)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    text = getattr(response, "text", None)

    question = clean_gemini_question(text)

    if not question:
        raise RuntimeError("Gemini returned an empty question.")

    previous_questions = {
        item.get("question")
        for item in session.get("questions", [])
    }

    if question in previous_questions:
        raise RuntimeError("Gemini generated a repeated question.")

    return question


def generate_question(session, client, model_name):
    """
    Gemini is the primary question generator.

    If Gemini fails for any reason, the predefined question bank
    automatically provides the next question.
    """

    try:
        question = ask_gemini(
            session,
            client,
            model_name
        )

        print("Question source: Gemini")
        print("Question:", question)

        return {
            "question": question,
            "source": "gemini"
        }

    except Exception as error:
        session["gemini_failures"] = session.get(
            "gemini_failures",
            0
        ) + 1

        print("========================================")
        print("GEMINI QUESTION GENERATION FAILED")
        print("Error:", repr(error))
        print("========================================")

        try:
            fallback = get_fallback_question(session)

            if fallback:
                print("Question source: Predefined fallback")
                print("Question:", fallback)

                return {
                    "question": fallback,
                    "source": "fallback"
                }

            print("ERROR: No fallback question available.")

        except Exception as fallback_error:
            print("========================================")
            print("FALLBACK QUESTION GENERATION FAILED")
            print("Error:", repr(fallback_error))
            print("========================================")

        raise RuntimeError(
            "Gemini failed and no fallback question is available."
        )