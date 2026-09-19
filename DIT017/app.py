import os
import uuid
from datetime import datetime, timedelta, timezone

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
import jwt
from dotenv import load_dotenv

from ai.interview_engine import (
    create_interview,
    generate_question,
    create_gemini_client
)


load_dotenv()

app = Flask(__name__)

CORS(app)

app.config["SECRET_KEY"] = os.getenv(
    "FLASK_SECRET_KEY",
    "careerforge-secret-key"
)

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is missing in .env")

mongo_client = MongoClient(MONGO_URI)

db = mongo_client["careerforge"]

users_collection = db["users"]

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

gemini_client = create_gemini_client()

INTERVIEWS = {}


print("MongoDB connected")
print("Gemini API key loaded:", bool(os.getenv("GEMINI_API_KEY")))
print("Gemini model:", GEMINI_MODEL)
print("Gemini client:", bool(gemini_client))


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def create_token(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }

    return jwt.encode(
        payload,
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )


def get_token():
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    return auth_header.split(" ", 1)[1].strip()


def get_current_user():
    token = get_token()

    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            app.config["SECRET_KEY"],
            algorithms=["HS256"]
        )

        user_id = payload.get("user_id")

        if not user_id:
            return None

        return users_collection.find_one({
            "_id": ObjectId(user_id)
        })

    except Exception:
        return None


def user_response(user):
    return {
        "id": str(user["_id"]),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "phoneno": user.get("phoneno", ""),
        "country": user.get("country", "")
    }


def get_interview(interview_id):
    return INTERVIEWS.get(interview_id)


def require_interview_access(interview_id):
    session = get_interview(interview_id)

    if not session:
        return None, (
            jsonify({
                "success": False,
                "message": "Interview not found."
            }),
            404
        )

    user = get_current_user()

    if not user:
        return None, (
            jsonify({
                "success": False,
                "message": "Unauthorized."
            }),
            401
        )

    if session.get("user_id") != str(user["_id"]):
        return None, (
            jsonify({
                "success": False,
                "message": "You do not have access to this interview."
            }),
            403
        )

    return session, None


# ---------------------------------------------------------
# Basic
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "CareerForge AI API is running."
    })


# ---------------------------------------------------------
# Register
# ---------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    phoneno = data.get("phoneno", "").strip()
    country = data.get("country", "").strip()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "message": "Name, email and password are required."
        }), 400

    existing_user = users_collection.find_one({
        "email": email
    })

    if existing_user:
        return jsonify({
            "success": False,
            "message": "Email already registered."
        }), 409

    user = {
        "name": name,
        "email": email,
        "phoneno": phoneno,
        "country": country,
        "password": generate_password_hash(password),
        "created_at": datetime.now(timezone.utc)
    }

    result = users_collection.insert_one(user)

    user["_id"] = result.inserted_id

    token = create_token(user["_id"])

    return jsonify({
        "success": True,
        "message": "Registration successful.",
        "token": token,
        "user": user_response(user)
    }), 201


# ---------------------------------------------------------
# Login
# ---------------------------------------------------------

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required."
        }), 400

    user = users_collection.find_one({
        "email": email
    })

    if not user:
        return jsonify({
            "success": False,
            "message": "Invalid email or password."
        }), 401

    if not check_password_hash(
        user.get("password", ""),
        password
    ):
        return jsonify({
            "success": False,
            "message": "Invalid email or password."
        }), 401

    token = create_token(user["_id"])

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "token": token,
        "user": user_response(user)
    })


# ---------------------------------------------------------
# Current User
# ---------------------------------------------------------

@app.route("/api/auth/me", methods=["GET"])
def current_user():
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "message": "Unauthorized."
        }), 401

    return jsonify({
        "success": True,
        "user": user_response(user)
    })


# ---------------------------------------------------------
# Start Interview
# ---------------------------------------------------------

@app.route("/api/interview/start", methods=["POST"])
def start_interview():
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    data = request.get_json() or {}

    interview_type = data.get(
        "interview_type",
        ""
    ).strip().lower()

    details = data.get("details", {})

    if interview_type not in ["technical", "hr"]:
        return jsonify({
            "success": False,
            "message": "Interview type must be technical or hr."
        }), 400

    if not isinstance(details, dict):
        return jsonify({
            "success": False,
            "message": "Invalid interview details."
        }), 400

    interview_id = str(uuid.uuid4())

    session = create_interview(
        interview_id,
        interview_type,
        details
    )

    session["user_id"] = str(user["_id"])

    # -----------------------------------------------------
    # First question
    # -----------------------------------------------------

    try:
        generated = generate_question(
            session,
            gemini_client,
            GEMINI_MODEL
        )

        first_question = generated["question"]
        source = generated["source"]

    except Exception as error:
        print("First question error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to generate interview question."
        }), 500

    session["questions"].append({
        "number": 1,
        "question": first_question,
        "answer": "",
        "source": source
    })

    INTERVIEWS[interview_id] = session

    return jsonify({
        "success": True,
        "interview_id": interview_id,
        "interview_type": interview_type,
        "question_number": 1,
        "total_questions": 15,
        "question": first_question,
        "question_source": source
    })


# ---------------------------------------------------------
# Get Interview
# ---------------------------------------------------------

@app.route("/api/interview/<interview_id>", methods=["GET"])
def get_interview_api(interview_id):
    session, error = require_interview_access(
        interview_id
    )

    if error:
        return error

    questions = session.get("questions", [])

    current_question = None

    if questions:
        current_question = questions[-1]

    return jsonify({
        "success": True,
        "interview_id": interview_id,
        "interview_type": session.get("type"),
        "candidate": session.get("details", {}),
        "current_question": current_question,
        "question_number": len(questions),
        "total_questions": session.get("max_questions", 15),
        "completed": session.get("completed", False)
    })


# ---------------------------------------------------------
# Submit Answer
# ---------------------------------------------------------

@app.route(
    "/api/interview/<interview_id>/answer",
    methods=["POST"]
)
def submit_answer(interview_id):
    session, error = require_interview_access(
        interview_id
    )

    if error:
        return error

    if session.get("completed"):
        return jsonify({
            "success": False,
            "message": "Interview already completed."
        }), 400

    data = request.get_json() or {}

    answer = data.get("answer", "").strip()

    questions = session.get("questions", [])

    if not questions:
        return jsonify({
            "success": False,
            "message": "No interview question found."
        }), 400

    # Save answer to current question
    questions[-1]["answer"] = answer

    # -----------------------------------------------------
    # Check Q15
    # -----------------------------------------------------

    if len(questions) >= session["max_questions"]:
        session["completed"] = True

        return jsonify({
            "success": True,
            "completed": True,
            "message": "Interview completed.",
            "question_number": len(questions),
            "total_questions": session["max_questions"]
        })

    # -----------------------------------------------------
    # Generate next question
    # Gemini FIRST -> fallback if Gemini fails
    # -----------------------------------------------------

    try:
        generated = generate_question(
            session,
            gemini_client,
            GEMINI_MODEL
        )

        next_question = generated["question"]
        source = generated["source"]

    except Exception as error:
        print("Question generation error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to generate the next question."
        }), 500

    question_number = len(questions) + 1

    questions.append({
        "number": question_number,
        "question": next_question,
        "answer": "",
        "source": source
    })

    return jsonify({
        "success": True,
        "completed": False,
        "question_number": question_number,
        "total_questions": session["max_questions"],
        "question": next_question,
        "question_source": source
    })


# ---------------------------------------------------------
# Upload Interview Audio
# ---------------------------------------------------------

@app.route(
    "/api/interview/<interview_id>/audio",
    methods=["POST"]
)
def upload_audio(interview_id):
    session, error = require_interview_access(
        interview_id
    )

    if error:
        return error

    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "message": "Audio file is required."
        }), 400

    audio = request.files["audio"]

    if not audio.filename:
        return jsonify({
            "success": False,
            "message": "Invalid audio file."
        }), 400

    filename = (
        f"{interview_id}_"
        f"{uuid.uuid4().hex}.mp3"
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    audio.save(filepath)

    session["audio_file"] = filename

    return jsonify({
        "success": True,
        "message": "Audio uploaded successfully."
    })


# ---------------------------------------------------------
# Interview Result
# ---------------------------------------------------------

@app.route(
    "/api/interview/<interview_id>/result",
    methods=["GET"]
)
def interview_result(interview_id):
    session, error = require_interview_access(
        interview_id
    )

    if error:
        return error

    return jsonify({
        "success": True,
        "interview_id": interview_id,
        "interview_type": session.get("type"),
        "candidate": session.get("details", {}),
        "total_questions": len(
            session.get("questions", [])
        ),
        "questions": session.get("questions", []),
        "audio_available": bool(
            session.get("audio_file")
        )
    })


# ---------------------------------------------------------
# Download Interview Audio
# ---------------------------------------------------------

@app.route(
    "/api/interview/<interview_id>/audio",
    methods=["GET"]
)
def download_audio(interview_id):
    session, error = require_interview_access(
        interview_id
    )

    if error:
        return error

    filename = session.get("audio_file")

    if not filename:
        return jsonify({
            "success": False,
            "message": "Audio not available."
        }), 404

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not os.path.exists(filepath):
        return jsonify({
            "success": False,
            "message": "Audio file not found."
        }), 404

    return send_file(
        filepath,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="careerforge-interview.mp3"
    )

@app.route("/api/users/getdetails", methods=["GET"])
def get_user_details():
    count = users_collection.count_documents({})

    return jsonify({
        "success": True,
        "count": count,
        "message": f"App is used by {count} users"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )