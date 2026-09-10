import json
import os
import re
import uuid

import pymupdf
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "careerforge-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# INTERVIEW STORAGE
# =========================================================

INTERVIEWS = {}

MAX_INTERVIEW_QUESTIONS = 6


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# AI INTERVIEW ENTRY
# =========================================================

@app.route("/interview")
def interview():
    return redirect(url_for("interview_upload"))


@app.route("/interview-setup")
def interview_setup():
    return redirect(url_for("interview_upload"))


# =========================================================
# INTERVIEW RESUME UPLOAD
# =========================================================

@app.route("/interview-upload", methods=["GET", "POST"])
def interview_upload():
    if request.method == "GET":
        return render_template("interview_upload.html")

    resume_file = request.files.get("resume")
    target_role = request.form.get("target_role", "").strip()

    if not resume_file:
        return render_template(
            "interview_upload.html",
            error="Please upload your resume."
        )

    if not resume_file.filename:
        return render_template(
            "interview_upload.html",
            error="Please select a PDF resume."
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "interview_upload.html",
            error="Only PDF resumes are supported."
        )

    if not gemini_client:
        return render_template(
            "interview_upload.html",
            error="Gemini API key is not configured. Add GEMINI_API_KEY to your .env file."
        )

    safe_filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        resume_file.filename
    )

    unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    pdf_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    resume_file.save(pdf_path)

    try:
        resume_text = extract_pdf_text(pdf_path)
    except Exception as error:
        print("PDF extraction error:", error)

        return render_template(
            "interview_upload.html",
            error="Unable to read the uploaded PDF."
        )

    if not resume_text.strip():
        return render_template(
            "interview_upload.html",
            error=(
                "No readable text was found in the PDF. "
                "Please upload a text-based PDF resume."
            )
        )

    try:
        interview_data = create_interview_with_ai(
            resume_text,
            target_role
        )
    except Exception as error:
        print("Gemini interview error:", error)

        return render_template(
            "interview_upload.html",
            error=(
                "AI interview could not be started. "
                f"Please check your Gemini API configuration. Error: {error}"
            )
        )

    interview_id = uuid.uuid4().hex

    INTERVIEWS[interview_id] = {
        "resume_text": resume_text[:20000],
        "candidate": interview_data.get("candidate", {}),
        "target_role": target_role,
        "question_number": 1,
        "max_questions": MAX_INTERVIEW_QUESTIONS,
        "current_question": interview_data.get(
            "first_question",
            "Tell me about yourself."
        ),
        "history": [],
        "interaction_id": interview_data.get("interaction_id"),
        "completed": False,
        "feedback": None
    }

    return redirect(
        url_for(
            "interview_session",
            interview_id=interview_id
        )
    )


# =========================================================
# INTERVIEW SESSION PAGE
# =========================================================

@app.route("/interview/session/<interview_id>")
def interview_session(interview_id):
    interview_data = INTERVIEWS.get(interview_id)

    if not interview_data:
        return redirect(url_for("interview_upload"))

    return render_template(
        "interview.html",
        interview_id=interview_id,
        candidate=interview_data["candidate"],
        target_role=interview_data["target_role"],
        question=interview_data["current_question"],
        question_number=interview_data["question_number"],
        max_questions=interview_data["max_questions"]
    )


# =========================================================
# GENERATE FIRST INTERVIEW QUESTION
# =========================================================

def create_interview_with_ai(resume_text, target_role):
    prompt = f"""
You are CareerForge AI, an expert technical and HR interviewer.

You are starting a realistic job interview for a student or job candidate.

Analyze the candidate's resume and create a personalized interview.

Target role:
{target_role if target_role else "Infer the most suitable role from the resume."}

Resume:
----------------
{resume_text[:20000]}
----------------

Your task:

1. Identify the candidate's name if available.
2. Identify their likely professional/technical profile.
3. Identify their strongest skills.
4. Identify their important projects.
5. Identify their experience level.
6. Start the interview with ONE natural question.

The interview must NOT be a generic fixed interview.

The first question should be relevant to the candidate's resume.

For example, if the resume contains a project, you may ask the candidate to explain that project.

Return ONLY valid JSON.

Required format:

{{
    "candidate": {{
        "name": "",
        "email": "",
        "phone": "",
        "profile": "",
        "skills": [],
        "projects": [],
        "experience": ""
    }},
    "first_question": ""
}}
"""

    response = gemini_client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    output = response.output_text

    data = parse_json_response(output)

    data["interaction_id"] = response.id

    return data


# =========================================================
# ANSWER INTERVIEW QUESTION
# =========================================================

@app.route(
    "/api/interview/<interview_id>/answer",
    methods=["POST"]
)
def interview_answer(interview_id):
    interview_data = INTERVIEWS.get(interview_id)

    if not interview_data:
        return jsonify({
            "success": False,
            "error": "Interview session not found."
        }), 404

    if interview_data["completed"]:
        return jsonify({
            "success": False,
            "error": "Interview has already been completed."
        }), 400

    data = request.get_json(silent=True) or {}

    answer = data.get("answer", "").strip()

    if not answer:
        return jsonify({
            "success": False,
            "error": "No answer was received."
        }), 400

    current_question = interview_data["current_question"]

    interview_data["history"].append({
        "question": current_question,
        "answer": answer
    })

    question_number = interview_data["question_number"]

    if question_number >= interview_data["max_questions"]:
        try:
            feedback = generate_final_feedback(
                interview_data
            )
        except Exception as error:
            print("Final feedback error:", error)

            feedback = {
                "overall_score": 0,
                "communication_score": 0,
                "technical_score": 0,
                "relevance_score": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": [
                    "Unable to generate AI feedback."
                ],
                "summary": "Interview completed."
            }

        interview_data["completed"] = True
        interview_data["feedback"] = feedback

        return jsonify({
            "success": True,
            "done": True,
            "feedback": feedback
        })

    try:
        next_question = generate_next_question(
            interview_data,
            answer
        )
    except Exception as error:
        print("Next question error:", error)

        return jsonify({
            "success": False,
            "error": (
                "Unable to generate the next AI question. "
                f"{error}"
            )
        }), 500

    interview_data["question_number"] += 1
    interview_data["current_question"] = next_question

    return jsonify({
        "success": True,
        "done": False,
        "next_question": next_question,
        "question_number": interview_data["question_number"],
        "max_questions": interview_data["max_questions"]
    })


# =========================================================
# GENERATE ADAPTIVE NEXT QUESTION
# =========================================================

def generate_next_question(interview_data, answer):
    candidate = interview_data["candidate"]
    current_question = interview_data["current_question"]

    history_text = ""

    for item in interview_data["history"]:
        history_text += (
            f"\nInterviewer: {item['question']}\n"
            f"Candidate: {item['answer']}\n"
        )

    prompt = f"""
You are CareerForge AI conducting a real job interview.

Candidate:
{json.dumps(candidate, indent=2)}

Target role:
{interview_data["target_role"] or "Infer from resume"}

Current question:
{current_question}

Candidate's latest answer:
{answer}

Previous interview:
{history_text}

Generate the NEXT interview question.

Rules:

- Make the question adaptive.
- Use information from the candidate's resume.
- Use information from their previous answer.
- Do not repeat previous questions.
- Mix technical, project, behavioral and situational questions.
- If the candidate mentioned a project or technology, you may ask a deeper follow-up.
- Keep the question natural and conversational.
- Ask exactly ONE question.
- Do not give feedback yet.
- Do not provide multiple questions.
- Do not number the question.

Return only the question text.
"""

    response = gemini_client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        previous_interaction_id=interview_data["interaction_id"]
    )

    interview_data["interaction_id"] = response.id

    return response.output_text.strip()


# =========================================================
# FINAL INTERVIEW FEEDBACK
# =========================================================

def generate_final_feedback(interview_data):
    candidate = interview_data["candidate"]

    history_text = ""

    for index, item in enumerate(
        interview_data["history"],
        start=1
    ):
        history_text += (
            f"\nQuestion {index}: {item['question']}\n"
            f"Answer {index}: {item['answer']}\n"
        )

    prompt = f"""
You are CareerForge AI.

The candidate has completed a job interview.

Candidate:
{json.dumps(candidate, indent=2)}

Target role:
{interview_data["target_role"] or "Not specified"}

Interview transcript:
{history_text}

Evaluate the candidate professionally.

Evaluate:

1. Overall performance
2. Communication quality
3. Technical understanding
4. Relevance of answers
5. Clarity
6. Confidence based only on the content and structure of answers
7. Strengths
8. Weaknesses
9. Improvement recommendations

Do not judge the candidate's accent, gender, appearance, race,
religion, disability or other personal characteristics.

Return ONLY valid JSON.

Format:

{{
    "overall_score": 0,
    "communication_score": 0,
    "technical_score": 0,
    "relevance_score": 0,
    "clarity_score": 0,
    "strengths": [],
    "weaknesses": [],
    "recommendations": [],
    "summary": ""
}}

All scores must be between 0 and 100.
"""

    response = gemini_client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        previous_interaction_id=interview_data["interaction_id"]
    )

    return parse_json_response(
        response.output_text
    )


# =========================================================
# INTERVIEW RESULT
# =========================================================

@app.route("/interview/<interview_id>/result")
def interview_result(interview_id):
    interview_data = INTERVIEWS.get(interview_id)

    if not interview_data:
        return redirect(url_for("interview_upload"))

    if not interview_data["completed"]:
        return redirect(
            url_for(
                "interview_session",
                interview_id=interview_id
            )
        )

    return render_template(
        "interview_result.html",
        interview_id=interview_id,
        candidate=interview_data["candidate"],
        feedback=interview_data["feedback"]
    )


# =========================================================
# INTERVIEW COMPLETE
# =========================================================

@app.route("/interview-complete")
def interview_complete():
    session.pop("candidate", None)
    return redirect(url_for("index"))


# =========================================================
# GROUP DISCUSSION SETUP
# =========================================================

@app.route(
    "/group-discussion-setup",
    methods=["GET", "POST"]
)
def group_discussion_setup():
    if request.method == "POST":
        discussion = {
            "topic": request.form.get(
                "topic",
                ""
            ).strip(),
            "duration": request.form.get(
                "duration",
                "10"
            ).strip(),
            "human_name": request.form.get(
                "human_name",
                "You"
            ).strip(),
            "participants": []
        }

        for i in range(1, 8):
            name = request.form.get(
                f"ai_name_{i}",
                f"Participant {i}"
            ).strip()

            if not name:
                name = f"Participant {i}"

            discussion["participants"].append(name)

        session["discussion"] = discussion

        return redirect(
            url_for("group_discussion")
        )

    return render_template(
        "group_discussion_setup.html"
    )


# =========================================================
# GROUP DISCUSSION
# =========================================================

@app.route(
    "/group-discussion",
    methods=["GET", "POST"]
)
def group_discussion():
    discussion = session.get(
        "discussion",
        {}
    )

    return render_template(
        "group_discussion.html",
        discussion=discussion
    )


# =========================================================
# COMMUNICATION
# =========================================================

@app.route("/communication")
def communication():
    return render_template(
        "communication.html"
    )


# =========================================================
# RESUME TESTER
# =========================================================

@app.route(
    "/resume",
    methods=["GET", "POST"]
)
def resume():
    if request.method == "GET":
        return render_template(
            "resume.html"
        )

    resume_file = request.files.get(
        "resume"
    )

    job_role = request.form.get(
        "job_role",
        ""
    ).strip()

    job_description = request.form.get(
        "job_description",
        ""
    ).strip()

    if not resume_file:
        return render_template(
            "resume.html",
            error="Please upload your resume."
        )

    if not resume_file.filename:
        return render_template(
            "resume.html",
            error="Please select a PDF resume."
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "resume.html",
            error="Only PDF files are supported."
        )

    if not job_role:
        return render_template(
            "resume.html",
            error="Please select a target job role."
        )

    if not job_description:
        return render_template(
            "resume.html",
            error="Please enter the job description."
        )

    original_filename = resume_file.filename

    safe_filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        original_filename
    )

    unique_filename = (
        f"{uuid.uuid4().hex}_{safe_filename}"
    )

    pdf_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    resume_file.save(pdf_path)

    try:
        resume_text = extract_pdf_text(
            pdf_path
        )
    except Exception as error:
        print(
            "PDF extraction error:",
            error
        )

        return render_template(
            "resume.html",
            error="Unable to read the uploaded PDF."
        )

    if not resume_text.strip():
        return render_template(
            "resume.html",
            error=(
                "No readable text was found in this PDF. "
                "If it is a scanned/image-only PDF, OCR is required."
            )
        )

    match_score = calculate_match_score(
        resume_text,
        job_description
    )

    skills_found = extract_skills(
        resume_text
    )

    jd_skills = extract_skills(
        job_description
    )

    resume_skill_lower = {
        skill.lower()
        for skill in skills_found
    }

    skill_gaps = [
        skill
        for skill in jd_skills
        if skill.lower() not in resume_skill_lower
    ]

    sections = analyze_resume_sections(
        resume_text
    )

    candidate_info = extract_candidate_information(
        resume_text
    )

    recommendations = generate_recommendations(
        skills_found,
        skill_gaps,
        resume_text,
        job_role,
        sections
    )

    analysis = {
        "job_role": job_role,
        "match_score": match_score,
        "resume_text": resume_text,
        "skills_found": skills_found,
        "skill_gaps": skill_gaps,
        "recommendations": recommendations,
        "sections": sections,
        "candidate_info": candidate_info,
        "filename": original_filename
    }

    session["resume_analysis"] = analysis

    return redirect(
        url_for(
            "resume_analytics_dashboard"
        )
    )


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_path):
    document = pymupdf.open(
        pdf_path
    )

    text_parts = []

    for page in document:
        page_text = page.get_text()

        if page_text:
            text_parts.append(
                page_text
            )

    document.close()

    return "\n".join(
        text_parts
    ).strip()


# =========================================================
# RESUME MATCH SCORE
# =========================================================

def calculate_match_score(
    resume_text,
    job_description
):
    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    score = int(
        round(
            similarity * 100
        )
    )

    return max(
        0,
        min(
            100,
            score
        )
    )


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(text):
    skill_database = [
        "Python",
        "Java",
        "C",
        "C++",
        "C#",
        "Dart",
        "JavaScript",
        "TypeScript",
        "HTML",
        "CSS",
        "Bootstrap",
        "Tailwind CSS",
        "React",
        "Angular",
        "Vue",
        "Node.js",
        "Express",
        "Flask",
        "Django",
        "PHP",
        "Laravel",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "SQLite",
        "Oracle",
        "REST API",
        "REST APIs",
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "NLP",
        "Computer Vision",
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "Keras",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "AWS",
        "Azure",
        "Google Cloud",
        "GCP",
        "Docker",
        "Kubernetes",
        "Jenkins",
        "Git",
        "GitHub",
        "GitLab",
        "VS Code",
        "Flutter",
        "Android",
        "Power BI",
        "Tableau",
        "Figma",
        "Data Structures",
        "Algorithms",
        "System Design",
        "SQL"
    ]

    found = []
    text_lower = text.lower()

    for skill in skill_database:
        skill_lower = skill.lower()

        if len(skill_lower) <= 3:
            pattern = (
                r"\b"
                + re.escape(skill_lower)
                + r"\b"
            )

            exists = re.search(
                pattern,
                text_lower
            )
        else:
            exists = skill_lower in text_lower

        if exists and skill not in found:
            found.append(skill)

    return found


# =========================================================
# RESUME SECTION ANALYSIS
# =========================================================

def analyze_resume_sections(text):
    text_lower = text.lower()

    sections = {
        "contact": False,
        "summary": False,
        "objective": False,
        "education": False,
        "experience": False,
        "projects": False,
        "skills": False,
        "certifications": False,
        "achievements": False,
        "languages": False
    }

    section_keywords = {
        "summary": [
            "summary",
            "professional summary",
            "profile"
        ],
        "objective": [
            "objective",
            "career objective"
        ],
        "education": [
            "education",
            "academic background",
            "qualification"
        ],
        "experience": [
            "experience",
            "work experience",
            "professional experience",
            "employment"
        ],
        "projects": [
            "projects",
            "academic projects",
            "personal projects"
        ],
        "skills": [
            "skills",
            "technical skills",
            "technical skill"
        ],
        "certifications": [
            "certification",
            "certifications",
            "courses"
        ],
        "achievements": [
            "achievement",
            "achievements",
            "awards",
            "honors"
        ],
        "languages": [
            "languages",
            "language skills"
        ]
    }

    for section, keywords in section_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                sections[section] = True
                break

    email_pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    phone_pattern = (
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"
    )

    sections["contact"] = bool(
        re.search(email_pattern, text)
        or re.search(phone_pattern, text)
    )

    return sections


# =========================================================
# CANDIDATE INFORMATION EXTRACTION
# =========================================================

def extract_candidate_information(text):
    information = {
        "email": "",
        "phone": "",
        "name": ""
    }

    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text
    )

    if email_match:
        information["email"] = (
            email_match.group(0)
        )

    phone_match = re.search(
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
        text
    )

    if phone_match:
        information["phone"] = (
            phone_match.group(0)
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:10]:
        if (
            len(line.split()) <= 5
            and "@" not in line
            and not re.search(
                r"\d{5,}",
                line
            )
        ):
            information["name"] = line
            break

    return information


# =========================================================
# RESUME RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    skills_found,
    skill_gaps,
    resume_text,
    job_role,
    sections
):
    recommendations = []

    if skill_gaps:
        missing = ", ".join(
            skill_gaps[:6]
        )

        recommendations.append(
            "Consider adding or developing "
            + missing
            + " because these skills appear relevant "
              "to the target job."
        )

    word_count = len(
        resume_text.split()
    )

    if word_count < 250:
        recommendations.append(
            "Your resume appears relatively short. "
            "Consider adding more detail about your "
            "projects, responsibilities and achievements."
        )

    if not sections["projects"]:
        recommendations.append(
            "Add a dedicated Projects section with "
            "technologies used, your contribution and "
            "the outcome of each project."
        )

    if not sections["experience"]:
        recommendations.append(
            "If you have internship, freelance or "
            "professional experience, add it with "
            "clear responsibilities and results."
        )

    if not sections["achievements"]:
        recommendations.append(
            "Add measurable achievements such as "
            "awards, competition results, performance "
            "improvements or project outcomes."
        )

    if not sections["certifications"]:
        recommendations.append(
            "Consider adding relevant certifications "
            "or professional courses that support "
            "your target role."
        )

    recommendations.append(
        "Practice technical and behavioral interview "
        "questions related to "
        + job_role
        + " to improve your placement readiness."
    )

    return recommendations


# =========================================================
# RESUME ANALYTICS DASHBOARD
# =========================================================

@app.route(
    "/resume-analytics-dashboard"
)
def resume_analytics_dashboard():
    analysis = session.get(
        "resume_analysis"
    )

    if not analysis:
        return redirect(
            url_for("resume")
        )

    return render_template(
        "resume_analytics_dashboard.html",
        analysis=analysis
    )


# =========================================================
# CLEAR RESUME ANALYSIS
# =========================================================

@app.route(
    "/clear-resume-analysis"
)
def clear_resume_analysis():
    session.pop(
        "resume_analysis",
        None
    )

    return redirect(
        url_for("resume")
    )


# =========================================================
# GEMINI JSON PARSER
# =========================================================

def parse_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "Gemini did not return valid JSON."
            )

        return json.loads(
            match.group(0)
        )


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )