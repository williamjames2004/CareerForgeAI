from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

# Secret key for storing interview data during the session
app.secret_key = "careerforge-secret-key"


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# AI INTERVIEW - SETUP
# =========================================================

@app.route("/interview")
def interview():
    return render_template("interview_setup.html")


# =========================================================
# AI INTERVIEW - START
# =========================================================

@app.route("/interview/start", methods=["POST"])
def start_interview():

    # -----------------------------------------------------
    # Collect candidate information
    # -----------------------------------------------------

    interview_data = {

        "full_name": request.form.get("full_name", "").strip(),

        "email": request.form.get("email", "").strip(),

        "phone": request.form.get("phone", "").strip(),

        "location": request.form.get("location", "").strip(),

        "degree": request.form.get("degree", "").strip(),

        "college": request.form.get("college", "").strip(),

        "graduation_year": request.form.get(
            "graduation_year", ""
        ).strip(),

        "specialization": request.form.get(
            "specialization", ""
        ).strip(),

        "work_experience": request.form.get(
            "work_experience", ""
        ).strip(),

        "company": request.form.get(
            "company", ""
        ).strip(),

        "job_role": request.form.get(
            "job_role", ""
        ).strip(),

        "projects": request.form.get(
            "projects", ""
        ).strip(),

        "skills": request.form.get(
            "skills", ""
        ).strip(),

        "career_goal": request.form.get(
            "career_goal", ""
        ).strip()
    }


    # -----------------------------------------------------
    # Store candidate data in session
    # -----------------------------------------------------

    session["interview_data"] = interview_data


    # -----------------------------------------------------
    # Start interview
    # -----------------------------------------------------

    return redirect(url_for("interview_session"))


# =========================================================
# AI INTERVIEW - QUESTIONS
# =========================================================

@app.route("/interview/session")
def interview_session():

    # Make sure setup was completed
    if "interview_data" not in session:

        return redirect(url_for("interview"))


    candidate = session["interview_data"]


    return render_template(
        "interview.html",
        candidate=candidate
    )


# =========================================================
# AI INTERVIEW - COMPLETE
# =========================================================

@app.route("/interview/complete")
def interview_complete():

    # Clear interview information
    session.pop("interview_data", None)


    return render_template("interview_success.html")


# =========================================================
# GROUP DISCUSSION SETUP
# =========================================================

@app.route("/group-discussion-setup")
def group_discussion_setup():

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

    if request.method == "POST":

        pass

    return render_template(
        "group_discussion.html"
    )


# =========================================================
# AI COMMUNICATION TEST
# =========================================================

@app.route("/communication")
def communication():

    return render_template(
        "communication.html"
    )


# =========================================================
# AI RESUME TESTER
# =========================================================

@app.route(
    "/resume",
    methods=["GET", "POST"]
)
def resume():

    if request.method == "POST":

        resume_file = request.files.get(
            "resume"
        )

        job_role = request.form.get(
            "job_role"
        )

        job_description = request.form.get(
            "job_description"
        )


        print(
            "Resume:",
            resume_file.filename
            if resume_file
            else None
        )

        print(
            "Job Role:",
            job_role
        )

        print(
            "Job Description:",
            job_description
        )


    return render_template(
        "resume.html"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )