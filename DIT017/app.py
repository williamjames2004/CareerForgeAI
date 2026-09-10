import json
import os
import re
import uuid

import pymupdf
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for
)
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "careerforge-secret-key"
)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini client initialized."
        )

    except Exception as error:

        print(
            "Gemini client initialization failed:",
            error
        )

        gemini_client = None

else:

    print(
        "WARNING: GEMINI_API_KEY is not configured."
    )

    print(
        "CareerForge will use fallback content where available."
    )


# =========================================================
# INTERVIEW STORAGE
# =========================================================

INTERVIEWS = {}

MAX_INTERVIEW_QUESTIONS = 6


# =========================================================
# GROUP DISCUSSION STORAGE
# =========================================================

GROUP_DISCUSSIONS = {}

MAX_GD_BOTS = 7


# =========================================================
# PREDEFINED INTERVIEW QUESTION BANK
# =========================================================

PREDEFINED_QUESTIONS = {

    "Software Developer": [

        "Tell me about yourself and your background.",

        "Which programming language are you most comfortable with and why?",

        "Tell me about an important project you have worked on.",

        "How do you approach debugging a program when you encounter an error?",

        "Explain a challenging technical problem you faced and how you solved it.",

        "Why do you want to work as a Software Developer?"
    ],

    "Full Stack Developer": [

        "Tell me about yourself and your experience with web development.",

        "Which frontend and backend technologies are you most comfortable with?",

        "Tell me about a full-stack project you have developed.",

        "How does a frontend application communicate with a backend server?",

        "How would you design a simple authentication system for a web application?",

        "Why do you want to work as a Full Stack Developer?"
    ],

    "Frontend Developer": [

        "Tell me about yourself and your interest in frontend development.",

        "What is the difference between HTML, CSS and JavaScript?",

        "Tell me about a frontend project you have worked on.",

        "How do you make a website responsive for different screen sizes?",

        "How do you improve the performance and user experience of a web page?",

        "Why do you want to become a Frontend Developer?"
    ],

    "Backend Developer": [

        "Tell me about yourself and your backend development experience.",

        "Which backend programming language or framework are you most comfortable with?",

        "Tell me about a backend project you have developed.",

        "What is an API and why is it useful in application development?",

        "How would you design a secure login system?",

        "Why do you want to work as a Backend Developer?"
    ],

    "Python Developer": [

        "Tell me about yourself and your experience with Python.",

        "What do you like about Python compared with other programming languages?",

        "Tell me about a project where you used Python.",

        "What is the difference between a list, tuple and dictionary in Python?",

        "How do you handle errors and exceptions in Python?",

        "Why do you want to work as a Python Developer?"
    ],

    "Java Developer": [

        "Tell me about yourself and your experience with Java.",

        "What are the main features of object-oriented programming?",

        "Tell me about a Java project you have worked on.",

        "What is the difference between an interface and an abstract class?",

        "How do you handle exceptions in Java?",

        "Why do you want to work as a Java Developer?"
    ],

    "AI/ML Engineer": [

        "Tell me about yourself and your interest in Artificial Intelligence and Machine Learning.",

        "Explain one machine learning project you have worked on.",

        "What is the difference between supervised and unsupervised learning?",

        "How do you determine whether a machine learning model is performing well?",

        "Tell me about a machine learning problem you faced and how you solved it.",

        "Why do you want to work as an AI/ML Engineer?"
    ],

    "Data Scientist": [

        "Tell me about yourself and your interest in data science.",

        "Tell me about a data science project you have worked on.",

        "What steps would you follow when starting a machine learning project?",

        "What is overfitting and how can you reduce it?",

        "How do you evaluate a machine learning model?",

        "Why do you want to work as a Data Scientist?"
    ],

    "Data Analyst": [

        "Tell me about yourself and your interest in data analysis.",

        "Which data analysis tools or technologies are you comfortable with?",

        "Tell me about a data analysis project you have worked on.",

        "What is the difference between mean, median and mode?",

        "How would you explain an important data insight to a non-technical person?",

        "Why do you want to work as a Data Analyst?"
    ],

    "Mobile App Developer": [

        "Tell me about yourself and your experience with mobile application development.",

        "Which mobile development framework or technology are you most comfortable with?",

        "Tell me about a mobile application you have developed.",

        "How do you manage application state in a mobile application?",

        "How do you make a mobile application responsive across different devices?",

        "Why do you want to work as a Mobile App Developer?"
    ],

    "Other": [

        "Tell me about yourself.",

        "What are your strongest technical skills?",

        "Tell me about an important project you have worked on.",

        "Describe a difficult problem you faced and how you solved it.",

        "What are your biggest strengths and areas for improvement?",

        "Why are you interested in this career?"
    ],

    "default": [

        "Tell me about yourself and your background.",

        "What are your strongest technical skills?",

        "Tell me about an important project you have worked on.",

        "Describe a challenging problem you faced and how you solved it.",

        "What are your strengths and areas for improvement?",

        "Why should we consider you for this role?"
    ]
}


# =========================================================
# PREDEFINED GD BOT PERSONALITIES
# =========================================================

GD_BOT_PROFILES = [

    {
        "id": 1,
        "name": "Arjun",
        "role": "Analytical Thinker",
        "description": (
            "Focuses on facts, logic, statistics, "
            "evidence and practical reasoning."
        )
    },

    {
        "id": 2,
        "name": "Priya",
        "role": "Social Perspective",
        "description": (
            "Focuses on society, students, people, "
            "communities and social consequences."
        )
    },

    {
        "id": 3,
        "name": "Rahul",
        "role": "Technology Expert",
        "description": (
            "Focuses on technology, innovation, "
            "digital transformation and technical possibilities."
        )
    },

    {
        "id": 4,
        "name": "Meera",
        "role": "Critical Thinker",
        "description": (
            "Challenges assumptions, identifies risks, "
            "limitations and opposing arguments."
        )
    },

    {
        "id": 5,
        "name": "Vikram",
        "role": "Business & Economic Perspective",
        "description": (
            "Focuses on business, employment, economy, "
            "industry and financial impact."
        )
    },

    {
        "id": 6,
        "name": "Ananya",
        "role": "Ethical Perspective",
        "description": (
            "Focuses on ethics, fairness, responsibility, "
            "privacy and long-term consequences."
        )
    },

    {
        "id": 7,
        "name": "Karthik",
        "role": "Balanced Speaker",
        "description": (
            "Provides balanced arguments and connects "
            "different perspectives into a practical conclusion."
        )
    }
]


# =========================================================
# PREDEFINED GD CONTENT
# =========================================================

PREDEFINED_GD_CONTENT = {

    "artificial intelligence": [

        "AI can improve productivity by automating repetitive tasks and helping people make faster decisions.",

        "AI can affect society by changing how students learn, how people work and how organizations deliver services.",

        "AI is transforming industries through machine learning, automation, computer vision and generative AI.",

        "AI also has limitations such as incorrect outputs, bias, privacy concerns and dependence on technology.",

        "From an economic perspective, AI can create new jobs while also changing or replacing some existing job roles.",

        "AI should be developed responsibly with transparency, fairness, privacy protection and human oversight.",

        "AI should be viewed as a tool that augments human capabilities rather than simply replacing human intelligence."
    ],

    "impact of social media on students": [

        "Social media allows students to access educational content, communities and opportunities very quickly.",

        "Excessive social media usage can affect concentration, time management and academic productivity.",

        "Technology platforms can help students discover courses, internships, professional networks and career opportunities.",

        "Students may compare themselves with others online, which can create unrealistic expectations and pressure.",

        "Social media has also become an important channel for businesses, creators and the digital economy.",

        "Students need digital responsibility, privacy awareness and healthy usage habits.",

        "The overall impact depends on how intentionally students use social media rather than simply whether they use it."
    ],

    "work from home": [

        "Remote work can save commuting time and provide employees with greater flexibility.",

        "For some employees, working from home can improve work-life balance and productivity.",

        "Remote work can reduce face-to-face interaction and make team communication more difficult.",

        "Technology has made distributed teams possible through video conferencing, collaboration tools and cloud platforms.",

        "Companies can reduce some office-related expenses through remote or hybrid work models.",

        "Remote work requires discipline, communication skills and reliable technology.",

        "A hybrid model can provide a balance between flexibility and direct team collaboration."
    ],

    "default": [

        "The topic has both advantages and disadvantages, so it should be examined from multiple perspectives.",

        "The impact of this issue depends strongly on how individuals, organizations and governments manage it.",

        "From a social perspective, the topic can create both opportunities and challenges for different groups.",

        "There are practical limitations that should be considered before adopting a particular approach.",

        "From an economic perspective, the topic can influence employment, businesses, productivity and growth.",

        "Ethical considerations such as fairness, responsibility, privacy and long-term consequences are also important.",

        "A balanced approach that combines innovation with responsibility may provide the most sustainable solution."
    ]
}


# =========================================================
# HELPER - PREDEFINED INTERVIEW QUESTIONS
# =========================================================

def get_predefined_questions(target_role):

    target_role = (
        target_role or ""
    ).strip()

    if target_role in PREDEFINED_QUESTIONS:

        return PREDEFINED_QUESTIONS[
            target_role
        ]

    return PREDEFINED_QUESTIONS[
        "default"
    ]


def get_next_predefined_question(
    interview_data
):

    questions = interview_data.get(
        "predefined_questions",
        []
    )

    if not questions:

        questions = get_predefined_questions(
            interview_data.get(
                "target_role",
                ""
            )
        )

        interview_data[
            "predefined_questions"
        ] = questions

    question_number = interview_data.get(
        "question_number",
        1
    )

    question_index = question_number

    if question_index < len(questions):

        return questions[
            question_index
        ]

    return questions[-1]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_path):

    text_parts = []

    document = pymupdf.open(
        pdf_path
    )

    try:

        for page in document:

            text_parts.append(
                page.get_text()
            )

    finally:

        document.close()

    return "\n".join(
        text_parts
    )


# =========================================================
# JSON RESPONSE PARSER
# =========================================================

def parse_json_response(output):

    if not output:

        raise ValueError(
            "Empty response received from AI."
        )

    output = output.strip()

    output = re.sub(
        r"^```json\s*",
        "",
        output,
        flags=re.IGNORECASE
    )

    output = re.sub(
        r"^```\s*",
        "",
        output
    )

    output = re.sub(
        r"\s*```$",
        "",
        output
    )

    output = output.strip()

    try:

        return json.loads(
            output
        )

    except json.JSONDecodeError:

        pass

    start = output.find(
        "{"
    )

    end = output.rfind(
        "}"
    )

    if (
        start != -1
        and end != -1
        and end > start
    ):

        json_text = output[
            start:end + 1
        ]

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError:

            pass

    raise ValueError(
        "Unable to parse AI response as JSON."
    )


# =========================================================
# INDEX
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# INTERVIEW ROUTES
# =========================================================

@app.route("/interview")
def interview():

    return redirect(
        url_for(
            "interview_upload"
        )
    )


@app.route("/interview-setup")
def interview_setup():

    return redirect(
        url_for(
            "interview_upload"
        )
    )


# =========================================================
# INTERVIEW UPLOAD / MANUAL DETAILS
# =========================================================

@app.route(
    "/interview-upload",
    methods=["GET", "POST"]
)
def interview_upload():

    if request.method == "GET":

        return render_template(
            "interview_upload.html"
        )

    input_method = request.form.get(
        "input_method",
        "pdf"
    ).strip()

    target_role = ""

    resume_text = ""

    # =====================================================
    # PDF MODE
    # =====================================================

    if input_method == "pdf":

        resume_file = request.files.get(
            "resume"
        )

        target_role = request.form.get(
            "target_role_pdf",
            ""
        ).strip()

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

        if not resume_file.filename.lower().endswith(
            ".pdf"
        ):

            return render_template(
                "interview_upload.html",
                error="Only PDF resumes are supported."
            )

        safe_filename = re.sub(
            r"[^a-zA-Z0-9._-]",
            "_",
            resume_file.filename
        )

        unique_filename = (
            f"{uuid.uuid4().hex}_"
            f"{safe_filename}"
        )

        pdf_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )

        try:

            resume_file.save(
                pdf_path
            )

        except Exception as error:

            print(
                "File save error:",
                error
            )

            return render_template(
                "interview_upload.html",
                error="Unable to save the uploaded resume."
            )

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

    # =====================================================
    # MANUAL MODE
    # =====================================================

    else:

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        target_role = request.form.get(
            "target_role",
            ""
        ).strip()

        profile = request.form.get(
            "profile",
            ""
        ).strip()

        skills = request.form.get(
            "skills",
            ""
        ).strip()

        projects = request.form.get(
            "projects",
            ""
        ).strip()

        experience = request.form.get(
            "experience",
            ""
        ).strip()

        if not name:

            return render_template(
                "interview_upload.html",
                error="Please enter your name."
            )

        if not target_role:

            return render_template(
                "interview_upload.html",
                error="Please select your target job role."
            )

        if not profile:

            return render_template(
                "interview_upload.html",
                error="Please tell us something about yourself."
            )

        resume_text = f"""
Candidate Name:
{name}

Email:
{email}

Phone:
{phone}

Target Role:
{target_role}

About Candidate:
{profile}

Skills and Technologies:
{skills}

Important Projects:
{projects}

Experience:
{experience}
""".strip()

    # =====================================================
    # TRY GEMINI
    # =====================================================

    try:

        if not gemini_client:

            raise RuntimeError(
                "Gemini API is not configured."
            )

        interview_data = create_interview_with_ai(
            resume_text,
            target_role
        )

        interview_mode = "ai"

        print(
            "Interview started using Gemini AI."
        )

    except Exception as error:

        print(
            "Gemini interview error:",
            error
        )

        print(
            "Switching to predefined interview questions."
        )

        predefined_questions = get_predefined_questions(
            target_role
        )

        interview_data = {

            "candidate": {
                "name": "",
                "email": "",
                "phone": "",
                "profile": "",
                "skills": [],
                "projects": [],
                "experience": ""
            },

            "first_question":
                predefined_questions[0]
        }

        interview_mode = "predefined"

    # =====================================================
    # MANUAL CANDIDATE INFORMATION
    # =====================================================

    if input_method == "manual":

        manual_skills = [

            skill.strip()

            for skill in skills.split(",")

            if skill.strip()
        ]

        manual_projects = []

        if projects:

            manual_projects.append(
                projects
            )

        manual_candidate = {

            "name":
                name,

            "email":
                email,

            "phone":
                phone,

            "profile":
                profile,

            "skills":
                manual_skills,

            "projects":
                manual_projects,

            "experience":
                experience
        }

        interview_data[
            "candidate"
        ] = manual_candidate

    # =====================================================
    # CREATE INTERVIEW SESSION
    # =====================================================

    interview_id = uuid.uuid4().hex

    predefined_questions = get_predefined_questions(
        target_role
    )

    INTERVIEWS[interview_id] = {

        "resume_text":
            resume_text[:20000],

        "candidate":
            interview_data.get(
                "candidate",
                {}
            ),

        "target_role":
            target_role,

        "question_number":
            1,

        "max_questions":
            MAX_INTERVIEW_QUESTIONS,

        "current_question":
            interview_data.get(
                "first_question",
                predefined_questions[0]
            ),

        "history":
            [],

        "interaction_id":
            interview_data.get(
                "interaction_id"
            ),

        "completed":
            False,

        "feedback":
            None,

        "mode":
            interview_mode,

        "predefined_questions":
            predefined_questions
    }

    return redirect(
        url_for(
            "interview_session",
            interview_id=interview_id
        )
    )


# =========================================================
# CREATE PERSONALIZED INTERVIEW USING GEMINI
# =========================================================

def create_interview_with_ai(
    resume_text,
    target_role
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini client is not available."
        )

    prompt = f"""
You are CareerForge AI, an expert technical and HR interviewer.

You are starting a realistic job interview for a student
or job candidate.

Analyze the candidate information below.

Target role:
{target_role if target_role else "Infer the most suitable role from the candidate information."}

Candidate information:
----------------------
{resume_text[:20000]}
----------------------

Your task:

1. Identify the candidate's name if available.
2. Identify their email if available.
3. Identify their phone if available.
4. Identify their likely professional or technical profile.
5. Identify their strongest skills.
6. Identify their important projects.
7. Identify their experience level.
8. Start the interview with ONE natural question.

The interview must NOT be a generic fixed interview.

The first question should be relevant to the candidate's
information.

If the candidate has a project, skill, internship,
experience or technical technology, use it when appropriate.

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

    data = parse_json_response(
        output
    )

    data["interaction_id"] = response.id

    return data


# =========================================================
# INTERVIEW SESSION PAGE
# =========================================================

@app.route(
    "/interview/session/<interview_id>"
)
def interview_session(interview_id):

    interview_data = INTERVIEWS.get(
        interview_id
    )

    if not interview_data:

        return redirect(
            url_for(
                "interview_upload"
            )
        )

    return render_template(

        "interview.html",

        interview_id=
            interview_id,

        candidate=
            interview_data[
                "candidate"
            ],

        target_role=
            interview_data[
                "target_role"
            ],

        question=
            interview_data[
                "current_question"
            ],

        question_number=
            interview_data[
                "question_number"
            ],

        max_questions=
            interview_data[
                "max_questions"
            ]
    )


# =========================================================
# SUBMIT INTERVIEW ANSWER
# =========================================================

@app.route(
    "/api/interview/<interview_id>/answer",
    methods=["POST"]
)
def interview_answer(interview_id):

    interview_data = INTERVIEWS.get(
        interview_id
    )

    if not interview_data:

        return jsonify({

            "success":
                False,

            "error":
                "Interview session not found."
        }), 404

    if interview_data["completed"]:

        return jsonify({

            "success":
                False,

            "error":
                "Interview has already been completed."
        }), 400

    data = request.get_json(
        silent=True
    ) or {}

    answer = data.get(
        "answer",
        ""
    ).strip()

    if not answer:

        return jsonify({

            "success":
                False,

            "error":
                "No answer was received."
        }), 400

    current_question = interview_data[
        "current_question"
    ]

    interview_data[
        "history"
    ].append({

        "question":
            current_question,

        "answer":
            answer
    })

    question_number = interview_data[
        "question_number"
    ]

    # =====================================================
    # INTERVIEW FINISHED
    # =====================================================

    if question_number >= interview_data[
        "max_questions"
    ]:

        try:

            if gemini_client:

                feedback = generate_final_feedback(
                    interview_data
                )

            else:

                raise RuntimeError(
                    "Gemini is not available."
                )

        except Exception as error:

            print(
                "Final feedback error:",
                error
            )

            feedback = generate_fallback_feedback(
                interview_data
            )

        interview_data[
            "completed"
        ] = True

        interview_data[
            "feedback"
        ] = feedback

        return jsonify({

            "success":
                True,

            "done":
                True,

            "feedback":
                feedback
        })

    # =====================================================
    # GET NEXT QUESTION
    # =====================================================

    try:

        if interview_data[
            "mode"
        ] == "ai":

            next_question = generate_next_question(
                interview_data,
                answer
            )

        else:

            next_question = get_next_predefined_question(
                interview_data
            )

    except Exception as error:

        print(
            "Next question error:",
            error
        )

        print(
            "Switching to predefined questions."
        )

        interview_data[
            "mode"
        ] = "predefined"

        next_question = get_next_predefined_question(
            interview_data
        )

    # =====================================================
    # UPDATE SESSION
    # =====================================================

    interview_data[
        "question_number"
    ] += 1

    interview_data[
        "current_question"
    ] = next_question

    return jsonify({

        "success":
            True,

        "done":
            False,

        "next_question":
            next_question,

        "question_number":
            interview_data[
                "question_number"
            ],

        "max_questions":
            interview_data[
                "max_questions"
            ]
    })


# =========================================================
# GENERATE NEXT AI QUESTION
# =========================================================

def generate_next_question(
    interview_data,
    answer
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini client is not available."
        )

    candidate = interview_data[
        "candidate"
    ]

    current_question = interview_data[
        "current_question"
    ]

    history_text = ""

    for item in interview_data[
        "history"
    ]:

        history_text += (

            f"\nInterviewer: "
            f"{item['question']}"

            f"\nCandidate: "
            f"{item['answer']}"

            f"\n"
        )

    prompt = f"""
You are CareerForge AI conducting a realistic job interview.

Candidate:
{json.dumps(candidate, indent=2)}

Target role:
{interview_data["target_role"] or "Infer from candidate information."}

Current question:
{current_question}

Candidate's latest answer:
{answer}

Previous interview:
{history_text}

Generate the NEXT interview question.

Rules:

- Make the question adaptive.
- Use information from the candidate.
- Use information from the previous answer.
- Do not repeat previous questions.
- Mix technical, project, behavioral and situational questions.
- If the candidate mentioned a project or technology,
  you may ask a deeper follow-up.
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

        previous_interaction_id=
            interview_data[
                "interaction_id"
            ]
    )

    interview_data[
        "interaction_id"
    ] = response.id

    question = response.output_text.strip()

    if not question:

        raise RuntimeError(
            "Gemini returned an empty question."
        )

    return question


# =========================================================
# FINAL AI FEEDBACK
# =========================================================

def generate_final_feedback(
    interview_data
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini client is not available."
        )

    candidate = interview_data[
        "candidate"
    ]

    history_text = ""

    for index, item in enumerate(
        interview_data["history"],
        start=1
    ):

        history_text += (

            f"\nQuestion {index}: "
            f"{item['question']}"

            f"\nAnswer {index}: "
            f"{item['answer']}"

            f"\n"
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

Do not judge the candidate's accent, gender,
appearance, race, religion, disability or other
personal characteristics.

Scores must be between 0 and 100.

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
"""

    response = gemini_client.interactions.create(

        model=GEMINI_MODEL,

        input=prompt,

        previous_interaction_id=
            interview_data[
                "interaction_id"
            ]
    )

    feedback = parse_json_response(
        response.output_text
    )

    return normalize_feedback(
        feedback
    )


# =========================================================
# NORMALIZE FEEDBACK
# =========================================================

def normalize_feedback(
    feedback
):

    score_fields = [

        "overall_score",

        "communication_score",

        "technical_score",

        "relevance_score",

        "clarity_score"
    ]

    for field in score_fields:

        try:

            value = int(
                feedback.get(
                    field,
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0

        value = max(
            0,
            min(
                100,
                value
            )
        )

        feedback[
            field
        ] = value

    if not isinstance(
        feedback.get(
            "strengths"
        ),
        list
    ):

        feedback[
            "strengths"
        ] = []

    if not isinstance(
        feedback.get(
            "weaknesses"
        ),
        list
    ):

        feedback[
            "weaknesses"
        ] = []

    if not isinstance(
        feedback.get(
            "recommendations"
        ),
        list
    ):

        feedback[
            "recommendations"
        ] = []

    if not feedback.get(
        "summary"
    ):

        feedback[
            "summary"
        ] = "Interview completed."

    return feedback


# =========================================================
# FALLBACK FEEDBACK
# =========================================================

def generate_fallback_feedback(
    interview_data
):

    history = interview_data.get(
        "history",
        []
    )

    total_questions = len(
        history
    )

    if total_questions == 0:

        return {

            "overall_score":
                0,

            "communication_score":
                0,

            "technical_score":
                0,

            "relevance_score":
                0,

            "clarity_score":
                0,

            "strengths":
                [],

            "weaknesses":
                [],

            "recommendations": [

                "Try answering the interview questions."
            ],

            "summary":
                "Interview completed."
        }

    answered = 0

    for item in history:

        answer = item.get(
            "answer",
            ""
        ).strip()

        if answer:

            answered += 1

    completion_score = round(

        (
            answered /
            total_questions
        ) * 100
    )

    return {

        "overall_score":
            completion_score,

        "communication_score":
            completion_score,

        "technical_score":
            completion_score,

        "relevance_score":
            completion_score,

        "clarity_score":
            completion_score,

        "strengths": [

            "Completed the interview questions.",

            "Demonstrated willingness to communicate.",

            "Provided responses to the interview."
        ],

        "weaknesses": [

            "Detailed AI evaluation was unavailable.",

            "Technical depth could not be fully evaluated."
        ],

        "recommendations": [

            "Practice answering interview questions clearly.",

            "Use specific examples from your projects.",

            "Explain your technical decisions with confidence.",

            "Practice speaking with concise and structured answers."
        ],

        "summary": (
            "The interview was completed successfully. "
            "Detailed AI evaluation was unavailable, "
            "so basic completion-based feedback was provided."
        )
    }


# =========================================================
# INTERVIEW RESULT
# =========================================================

@app.route(
    "/interview/<interview_id>/result"
)
def interview_result(
    interview_id
):

    interview_data = INTERVIEWS.get(
        interview_id
    )

    if not interview_data:

        return redirect(
            url_for(
                "interview_upload"
            )
        )

    if not interview_data[
        "completed"
    ]:

        return redirect(
            url_for(
                "interview_session",
                interview_id=interview_id
            )
        )

    return render_template(

        "interview_result.html",

        interview_id=
            interview_id,

        candidate=
            interview_data[
                "candidate"
            ],

        feedback=
            interview_data[
                "feedback"
            ]
    )


# =========================================================
# INTERVIEW COMPLETE
# =========================================================

@app.route(
    "/interview-complete"
)
def interview_complete():

    session.pop(
        "candidate",
        None
    )

    return redirect(
        url_for(
            "index"
        )
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

    target_role = request.form.get(
        "target_role",
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
            error="Please select a resume PDF."
        )

    if not resume_file.filename.lower().endswith(
        ".pdf"
    ):

        return render_template(
            "resume.html",
            error="Only PDF files are supported."
        )

    safe_filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        resume_file.filename
    )

    unique_filename = (
        f"{uuid.uuid4().hex}_"
        f"{safe_filename}"
    )

    pdf_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    try:

        resume_file.save(
            pdf_path
        )

        resume_text = extract_pdf_text(
            pdf_path
        )

    except Exception as error:

        print(
            "Resume processing error:",
            error
        )

        return render_template(
            "resume.html",
            error="Unable to process the resume."
        )

    if not resume_text.strip():

        return render_template(
            "resume.html",
            error="No readable text found in the PDF."
        )

    skills = extract_skills(
        resume_text
    )

    sections = analyze_resume_sections(
        resume_text
    )

    candidate = extract_candidate_information(
        resume_text
    )

    match_score = calculate_resume_match_score(
        resume_text,
        target_role,
        job_description
    )

    recommendations = generate_recommendations(
        resume_text,
        target_role,
        job_description,
        sections,
        skills
    )

    analysis = {

        "candidate":
            candidate,

        "target_role":
            target_role,

        "job_description":
            job_description,

        "match_score":
            match_score,

        "skills":
            skills,

        "sections":
            sections,

        "recommendations":
            recommendations,

        "resume_text":
            resume_text[:20000]
    }

    session[
        "resume_analysis"
    ] = analysis

    return redirect(
        url_for(
            "resume_analytics_dashboard"
        )
    )


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
            url_for(
                "resume"
            )
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
        url_for(
            "resume"
        )
    )


# =========================================================
# RESUME - SKILL EXTRACTION
# =========================================================

COMMON_SKILLS = [

    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C",
    "C++",
    "C#",

    "HTML",
    "CSS",
    "Bootstrap",
    "Tailwind CSS",

    "React",
    "Angular",
    "Vue",

    "Node.js",
    "Express.js",
    "Flask",
    "Django",

    "PHP",
    "Laravel",

    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "SQLite",

    "Git",
    "GitHub",

    "Docker",

    "Flutter",
    "Dart",
    "Android",

    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "AI",
    "NLP",
    "Natural Language Processing",

    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",

    "Data Science",
    "Data Analysis",

    "Power BI",
    "Tableau",

    "AWS",
    "Azure",
    "Google Cloud",

    "REST API",
    "REST APIs",

    "SQL",

    "OOP",
    "Object Oriented Programming",

    "Problem Solving",
    "Communication",
    "Leadership"
]


def extract_skills(text):

    text_lower = text.lower()

    found = []

    for skill in COMMON_SKILLS:

        if skill.lower() in text_lower:

            found.append(
                skill
            )

    return found


# =========================================================
# RESUME - SECTION ANALYSIS
# =========================================================

def analyze_resume_sections(
    text
):

    text_lower = text.lower()

    sections = {

        "contact":
            False,

        "summary":
            False,

        "education":
            False,

        "experience":
            False,

        "projects":
            False,

        "skills":
            False,

        "certifications":
            False,

        "achievements":
            False
    }

    section_keywords = {

        "contact": [
            "email",
            "phone",
            "mobile",
            "@"
        ],

        "summary": [
            "summary",
            "profile",
            "objective",
            "about me"
        ],

        "education": [
            "education",
            "bachelor",
            "master",
            "degree",
            "university",
            "college"
        ],

        "experience": [
            "experience",
            "employment",
            "work history",
            "internship"
        ],

        "projects": [
            "projects",
            "project"
        ],

        "skills": [
            "skills",
            "technical skills",
            "technologies"
        ],

        "certifications": [
            "certification",
            "certifications",
            "certificate"
        ],

        "achievements": [
            "achievement",
            "achievements",
            "awards",
            "award"
        ]
    }

    for section, keywords in section_keywords.items():

        for keyword in keywords:

            if keyword in text_lower:

                sections[
                    section
                ] = True

                break

    return sections


# =========================================================
# RESUME - CANDIDATE INFORMATION
# =========================================================

def extract_candidate_information(
    text
):

    lines = [

        line.strip()

        for line in text.splitlines()

        if line.strip()
    ]

    name = ""

    if lines:

        first_line = lines[0]

        if (

            len(first_line) <= 80

            and "@" not in first_line

            and not re.search(
                r"\d{5,}",
                first_line
            )
        ):

            name = first_line

    email = ""

    email_match = re.search(

        r"[\w\.-]+@[\w\.-]+\.\w+",

        text
    )

    if email_match:

        email = email_match.group(
            0
        )

    phone = ""

    phone_match = re.search(

        r"(?:\+91[\s-]?)?[6-9]\d{9}",

        text
    )

    if phone_match:

        phone = phone_match.group(
            0
        )

    return {

        "name":
            name,

        "email":
            email,

        "phone":
            phone
    }


# =========================================================
# RESUME - MATCH SCORE
# =========================================================

def calculate_resume_match_score(
    resume_text,
    target_role,
    job_description
):

    resume_text = resume_text.strip()

    if not resume_text:

        return 0

    comparison_text = ""

    if target_role:

        comparison_text += (
            target_role + " "
        )

    if job_description:

        comparison_text += (
            job_description
        )

    if not comparison_text.strip():

        return 0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform([

            resume_text,

            comparison_text
        ])

        score = cosine_similarity(

            vectors[0:1],

            vectors[1:2]
        )[0][0]

        return round(
            score * 100,
            2
        )

    except Exception as error:

        print(
            "Resume similarity error:",
            error
        )

        return 0


# =========================================================
# RESUME - RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    resume_text,
    target_role,
    job_description,
    sections,
    skills
):

    recommendations = []

    if not sections.get(
        "summary"
    ):

        recommendations.append(
            "Add a concise professional summary or career objective."
        )

    if not sections.get(
        "skills"
    ):

        recommendations.append(
            "Add a clearly organized technical skills section."
        )

    if not sections.get(
        "projects"
    ):

        recommendations.append(
            "Add relevant academic or personal projects."
        )

    if not sections.get(
        "experience"
    ):

        recommendations.append(
            "Add internships, work experience, freelance work or practical experience."
        )

    if not sections.get(
        "education"
    ):

        recommendations.append(
            "Make your education details clearly visible."
        )

    if not skills:

        recommendations.append(
            "Add specific technologies and technical skills relevant to your target role."
        )

    if target_role:

        target_lower = target_role.lower()

        if target_lower not in resume_text.lower():

            recommendations.append(

                f"Consider highlighting experience or projects related to {target_role}."
            )

    if job_description:

        recommendations.append(
            "Customize your resume keywords according to the target job description."
        )

    if not recommendations:

        recommendations.extend([

            "Your resume contains the major sections expected in a professional resume.",

            "Continue adding measurable achievements to strengthen your experience.",

            "Keep your project descriptions focused on your contribution and technical impact."
        ])

    return recommendations


# =========================================================
# COMMUNICATION
# =========================================================

@app.route(
    "/communication"
)
def communication():

    return render_template(
        "communication.html"
    )


# =========================================================
# GROUP DISCUSSION SETUP
# =========================================================

@app.route(
    "/group-discussion-setup",
    methods=["GET", "POST"]
)
def group_discussion_setup():

    if request.method == "GET":

        return render_template(
            "group_discussion_setup.html"
        )

    topic = request.form.get(
        "topic",
        ""
    ).strip()

    if not topic:

        return render_template(
            "group_discussion_setup.html",
            error="Please enter or select a discussion topic."
        )

    try:

        duration = int(
            request.form.get(
                "duration",
                10
            )
        )

    except (
        TypeError,
        ValueError
    ):

        duration = 10

    duration = max(
        3,
        min(
            duration,
            30
        )
    )

    # =====================================================
    # CREATE GD CONTENT
    # =====================================================

    try:

        if not gemini_client:

            raise RuntimeError(
                "Gemini API is not configured."
            )

        participants = generate_gd_content(
            topic
        )

        mode = "ai"

        print(
            "Group discussion generated using Gemini AI."
        )

    except Exception as error:

        print(
            "GD AI generation error:",
            error
        )

        print(
            "Switching to predefined GD content."
        )

        participants = generate_fallback_gd_content(
            topic
        )

        mode = "predefined"

    # =====================================================
    # CREATE GD SESSION
    # =====================================================

    discussion_id = uuid.uuid4().hex

    GROUP_DISCUSSIONS[
        discussion_id
    ] = {

        "topic":
            topic,

        "duration":
            duration,

        "participants":
            participants,

        "mode":
            mode,

        "user_contributions":
            [],

        "started":
            False
    }

    return redirect(
        url_for(
            "group_discussion",
            discussion_id=discussion_id
        )
    )


# =========================================================
# GENERATE 7 AI GD PARTICIPANTS
# =========================================================

def generate_gd_content(
    topic
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini client is not available."
        )

    profiles_text = ""

    for bot in GD_BOT_PROFILES:

        profiles_text += f"""

Participant {bot["id"]}:
Name: {bot["name"]}
Role: {bot["role"]}
Perspective: {bot["description"]}
"""

    prompt = f"""
You are CareerForge AI conducting a realistic
Group Discussion practice session.

Discussion Topic:
{topic}

There are exactly 7 AI participants.

Their profiles are:

{profiles_text}

Generate ONE strong opening contribution for
each participant.

Important rules:

- Generate exactly 7 participants.
- Every participant must have a different argument.
- Do not repeat the same point.
- Each participant must speak from their assigned perspective.
- The content should sound natural when spoken aloud.
- Each contribution should be around 50 to 90 words.
- Use simple but professional English.
- Include practical examples where appropriate.
- Some participants may partially agree with another
  participant but should add a new point.
- Do not make all participants agree.
- Include both advantages and disadvantages where relevant.
- Do not use offensive, discriminatory or political persuasion.
- Do not mention that the participants are AI.
- These are opening statements, not questions.

Return ONLY valid JSON.

Required format:

{{
    "participants": [
        {{
            "id": 1,
            "name": "Arjun",
            "role": "Analytical Thinker",
            "content": ""
        }},
        {{
            "id": 2,
            "name": "Priya",
            "role": "Social Perspective",
            "content": ""
        }},
        {{
            "id": 3,
            "name": "Rahul",
            "role": "Technology Expert",
            "content": ""
        }},
        {{
            "id": 4,
            "name": "Meera",
            "role": "Critical Thinker",
            "content": ""
        }},
        {{
            "id": 5,
            "name": "Vikram",
            "role": "Business & Economic Perspective",
            "content": ""
        }},
        {{
            "id": 6,
            "name": "Ananya",
            "role": "Ethical Perspective",
            "content": ""
        }},
        {{
            "id": 7,
            "name": "Karthik",
            "role": "Balanced Speaker",
            "content": ""
        }}
    ]
}}
"""

    response = gemini_client.interactions.create(

        model=GEMINI_MODEL,

        input=prompt
    )

    data = parse_json_response(
        response.output_text
    )

    participants = data.get(
        "participants",
        []
    )

    if not isinstance(
        participants,
        list
    ):

        raise ValueError(
            "Invalid participant data returned by Gemini."
        )

    if len(participants) != 7:

        raise ValueError(
            "Gemini did not return exactly 7 participants."
        )

    # =====================================================
    # NORMALIZE PARTICIPANTS
    # =====================================================

    normalized = []

    for index, profile in enumerate(
        GD_BOT_PROFILES
    ):

        generated = participants[index]

        content = str(
            generated.get(
                "content",
                ""
            )
        ).strip()

        if not content:

            raise ValueError(
                f"Participant {index + 1} has empty content."
            )

        normalized.append({

            "id":
                profile["id"],

            "name":
                profile["name"],

            "role":
                profile["role"],

            "content":
                content
        })

    return normalized


# =========================================================
# FALLBACK GD CONTENT
# =========================================================

def generate_fallback_gd_content(
    topic
):

    topic_key = topic.lower().strip()

    content_list = None

    for key, values in PREDEFINED_GD_CONTENT.items():

        if key in topic_key:

            content_list = values

            break

    if not content_list:

        content_list = (
            PREDEFINED_GD_CONTENT[
                "default"
            ]
        )

    participants = []

    for index, profile in enumerate(
        GD_BOT_PROFILES
    ):

        content = content_list[
            index % len(content_list)
        ]

        participants.append({

            "id":
                profile["id"],

            "name":
                profile["name"],

            "role":
                profile["role"],

            "content":
                content
        })

    return participants


# =========================================================
# GROUP DISCUSSION PAGE
# =========================================================

@app.route(
    "/group-discussion"
)
def group_discussion():

    discussion_id = request.args.get(
        "discussion_id",
        ""
    ).strip()

    # -----------------------------------------------------
    # BACKWARD COMPATIBILITY
    # -----------------------------------------------------

    if not discussion_id:

        return render_template(
            "group_discussion.html",
            topic="",
            participants=[],
            discussion_id="",
            duration=10,
            mode="none"
        )

    discussion = GROUP_DISCUSSIONS.get(
        discussion_id
    )

    if not discussion:

        return redirect(
            url_for(
                "group_discussion_setup"
            )
        )

    return render_template(

        "group_discussion.html",

        discussion_id=
            discussion_id,

        topic=
            discussion[
                "topic"
            ],

        duration=
            discussion[
                "duration"
            ],

        participants=
            discussion[
                "participants"
            ],

        mode=
            discussion[
                "mode"
            ]
    )


# =========================================================
# GET GROUP DISCUSSION DATA
# =========================================================

@app.route(
    "/api/group-discussion/<discussion_id>"
)
def get_group_discussion(
    discussion_id
):

    discussion = GROUP_DISCUSSIONS.get(
        discussion_id
    )

    if not discussion:

        return jsonify({

            "success":
                False,

            "error":
                "Group discussion session not found."
        }), 404

    return jsonify({

        "success":
            True,

        "discussion_id":
            discussion_id,

        "topic":
            discussion[
                "topic"
            ],

        "duration":
            discussion[
                "duration"
            ],

        "participants":
            discussion[
                "participants"
            ],

        "mode":
            discussion[
                "mode"
            ]
    })


# =========================================================
# USER CONTRIBUTION TO GD
# =========================================================

@app.route(
    "/api/group-discussion/<discussion_id>/contribute",
    methods=["POST"]
)
def group_discussion_contribute(
    discussion_id
):

    discussion = GROUP_DISCUSSIONS.get(
        discussion_id
    )

    if not discussion:

        return jsonify({

            "success":
                False,

            "error":
                "Group discussion session not found."
        }), 404

    data = request.get_json(
        silent=True
    ) or {}

    content = data.get(
        "content",
        ""
    ).strip()

    if not content:

        return jsonify({

            "success":
                False,

            "error":
                "No contribution was received."
        }), 400

    discussion[
        "user_contributions"
    ].append({

        "content":
            content
    })

    return jsonify({

        "success":
            True,

        "message":
            "Contribution recorded.",

        "contribution_number":
            len(
                discussion[
                    "user_contributions"
                ]
            )
    })


# =========================================================
# GENERATE NEXT GD RESPONSE
# =========================================================

@app.route(
    "/api/group-discussion/<discussion_id>/next",
    methods=["POST"]
)
def group_discussion_next(
    discussion_id
):

    discussion = GROUP_DISCUSSIONS.get(
        discussion_id
    )

    if not discussion:

        return jsonify({

            "success":
                False,

            "error":
                "Group discussion session not found."
        }), 404

    data = request.get_json(
        silent=True
    ) or {}

    user_message = data.get(
        "user_message",
        ""
    ).strip()

    if not user_message:

        return jsonify({

            "success":
                False,

            "error":
                "Please provide your contribution."
        }), 400

    discussion[
        "user_contributions"
    ].append({

        "content":
            user_message
    })

    # =====================================================
    # AI RESPONSE
    # =====================================================

    if not gemini_client:

        return jsonify({

            "success":
                True,

            "participant":
                GD_BOT_PROFILES[0],

            "content":
                (
                    "That is an interesting point. "
                    "I would also consider the practical "
                    "impact of this issue on students, "
                    "organizations and society."
                )
        })

    try:

        history_text = ""

        for item in discussion[
            "user_contributions"
        ][-5:]:

            history_text += (
                "\nUser contribution: "
                f"{item['content']}"
            )

        prompt = f"""
You are one participant in a professional
Group Discussion practice session.

Topic:
{discussion["topic"]}

Your personality:
Balanced Speaker

Other participant perspectives:
Analytical, Social, Technology,
Critical, Business, Ethical

The human candidate just said:

{user_message}

Recent discussion:
{history_text}

Generate ONE short response that could naturally
be spoken by another GD participant.

Rules:

- 40 to 70 words.
- Do not repeat the candidate's exact words.
- Respond directly to the candidate's point.
- Either agree with an additional insight,
  politely disagree, or introduce a new perspective.
- Stay relevant to the topic.
- Sound natural and conversational.
- Do not mention AI.
- Do not ask multiple questions.
- Return only the spoken contribution.
"""

        response = gemini_client.interactions.create(

            model=GEMINI_MODEL,

            input=prompt
        )

        content = response.output_text.strip()

        if not content:

            raise RuntimeError(
                "Gemini returned empty GD content."
            )

        participant = GD_BOT_PROFILES[
            len(
                discussion[
                    "user_contributions"
                ]
            ) % MAX_GD_BOTS
        ]

        return jsonify({

            "success":
                True,

            "participant":
                participant,

            "content":
                content
        })

    except Exception as error:

        print(
            "GD response generation error:",
            error
        )

        return jsonify({

            "success":
                True,

            "participant":
                GD_BOT_PROFILES[0],

            "content":
                (
                    "I agree that this issue should be "
                    "looked at from multiple perspectives. "
                    "Along with the benefits, we should "
                    "also consider its practical limitations "
                    "and long-term impact."
                ),

            "fallback":
                True
        })


# =========================================================
# GROUP DISCUSSION RESULT / FEEDBACK
# =========================================================

@app.route(
    "/api/group-discussion/<discussion_id>/finish",
    methods=["POST"]
)
def finish_group_discussion(
    discussion_id
):

    discussion = GROUP_DISCUSSIONS.get(
        discussion_id
    )

    if not discussion:

        return jsonify({

            "success":
                False,

            "error":
                "Group discussion session not found."
        }), 404

    contributions = discussion.get(
        "user_contributions",
        []
    )

    if not contributions:

        return jsonify({

            "success":
                False,

            "error":
                "No user contributions were recorded."
        }), 400

    # =====================================================
    # AI GD EVALUATION
    # =====================================================

    if not gemini_client:

        feedback = generate_fallback_gd_feedback(
            contributions
        )

        return jsonify({

            "success":
                True,

            "feedback":
                feedback
        })

    try:

        contribution_text = ""

        for index, item in enumerate(
            contributions,
            start=1
        ):

            contribution_text += (
                f"\nContribution {index}: "
                f"{item['content']}"
            )

        prompt = f"""
You are CareerForge AI evaluating a student's
performance in a Group Discussion.

Topic:
{discussion["topic"]}

Student contributions:
{contribution_text}

Evaluate the student's GD performance.

Evaluate:

1. Overall performance
2. Communication
3. Relevance
4. Clarity
5. Confidence based on response structure
6. Quality of arguments
7. Participation
8. Strengths
9. Weaknesses
10. Recommendations

Do not evaluate accent, appearance, gender,
race, religion, disability or other personal
characteristics.

Scores must be between 0 and 100.

Return ONLY valid JSON.

Format:

{{
    "overall_score": 0,
    "communication_score": 0,
    "relevance_score": 0,
    "clarity_score": 0,
    "argument_score": 0,
    "participation_score": 0,
    "strengths": [],
    "weaknesses": [],
    "recommendations": [],
    "summary": ""
}}
"""

        response = gemini_client.interactions.create(

            model=GEMINI_MODEL,

            input=prompt
        )

        feedback = parse_json_response(
            response.output_text
        )

        feedback = normalize_gd_feedback(
            feedback
        )

        return jsonify({

            "success":
                True,

            "feedback":
                feedback
        })

    except Exception as error:

        print(
            "GD feedback error:",
            error
        )

        feedback = generate_fallback_gd_feedback(
            contributions
        )

        return jsonify({

            "success":
                True,

            "feedback":
                feedback,

            "fallback":
                True
        })


# =========================================================
# NORMALIZE GD FEEDBACK
# =========================================================

def normalize_gd_feedback(
    feedback
):

    score_fields = [

        "overall_score",

        "communication_score",

        "relevance_score",

        "clarity_score",

        "argument_score",

        "participation_score"
    ]

    for field in score_fields:

        try:

            value = int(
                feedback.get(
                    field,
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0

        feedback[
            field
        ] = max(
            0,
            min(
                100,
                value
            )
        )

    list_fields = [

        "strengths",

        "weaknesses",

        "recommendations"
    ]

    for field in list_fields:

        if not isinstance(
            feedback.get(
                field
            ),
            list
        ):

            feedback[
                field
            ] = []

    if not feedback.get(
        "summary"
    ):

        feedback[
            "summary"
        ] = "Group discussion completed."

    return feedback


# =========================================================
# FALLBACK GD FEEDBACK
# =========================================================

def generate_fallback_gd_feedback(
    contributions
):

    count = len(
        contributions
    )

    participation_score = min(
        100,
        count * 20
    )

    return {

        "overall_score":
            participation_score,

        "communication_score":
            participation_score,

        "relevance_score":
            participation_score,

        "clarity_score":
            participation_score,

        "argument_score":
            participation_score,

        "participation_score":
            participation_score,

        "strengths": [

            "Participated in the group discussion.",

            "Presented personal viewpoints.",

            "Attempted to engage with the topic."
        ],

        "weaknesses": [

            "Detailed AI evaluation was unavailable.",

            "Argument quality could not be fully evaluated."
        ],

        "recommendations": [

            "Support your opinions with practical examples.",

            "Listen to opposing viewpoints before responding.",

            "Keep your contributions concise and structured.",

            "Try to introduce new points instead of repeating others."
        ],

        "summary": (

            f"You made {count} contribution(s) "
            "during the group discussion. "
            "Continue practicing structured speaking "
            "and evidence-based arguments."
        )
    }


# =========================================================
# ERROR HANDLER
# =========================================================

@app.errorhandler(
    413
)
def request_entity_too_large(
    error
):

    return render_template(

        "interview_upload.html",

        error=(
            "The uploaded file is larger "
            "than the 10 MB limit."
        )

    ), 413


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "CareerForge AI"
    )

    print(
        "=" * 60
    )

    if gemini_client:

        print(
            "Interview AI : ENABLED"
        )

        print(
            "Group Discussion AI : ENABLED"
        )

        print(
            f"Gemini Model : {GEMINI_MODEL}"
        )

    else:

        print(
            "Interview AI : FALLBACK MODE"
        )

        print(
            "Group Discussion AI : FALLBACK MODE"
        )

        print(
            "Predefined content will be used."
        )

    print(
        "=" * 60
    )

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )
