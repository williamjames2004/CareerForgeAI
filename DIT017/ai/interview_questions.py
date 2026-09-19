import random

TECHNICAL_GENERAL = [
    "Tell me about yourself and your technical background.",
    "What is the most challenging technical problem you have solved?",
    "How do you approach debugging when your application is not working as expected?",
    "How do you learn a new technology or programming language?",
    "Explain a technical concept that you are confident about.",
    "How do you make sure your code is maintainable and readable?",
    "Tell me about a time when you had to solve a problem under time pressure.",
    "How do you handle errors and exceptions in your applications?",
    "What is your approach to testing an application before deployment?",
    "How do you decide which technology to use for a project?",
    "Tell me about a technical mistake you made and what you learned from it.",
    "How do you improve the performance of a slow application?",
    "How do you manage your time when working on multiple technical tasks?",
    "What area of technology are you currently trying to improve?",
    "Where do you see yourself technically in the next few years?"
]

HR_GENERAL = [
    "Tell me about yourself.",
    "Why are you interested in this role?",
    "Why should we hire you?",
    "What are your strengths?",
    "What is one area you are currently trying to improve?",
    "Tell me about a challenge you faced and how you handled it.",
    "How do you handle pressure?",
    "How do you work with people who have different opinions from you?",
    "Tell me about a time you worked successfully as part of a team.",
    "Describe a situation where you took responsibility for something.",
    "What motivates you to perform well?",
    "What are your career goals?",
    "How do you handle failure?",
    "What kind of work environment helps you perform your best?",
    "Where do you see yourself in the next five years?"
]

TECHNICAL_PROJECT = [
    "Tell me about the project {value} that you worked on.",
    "What problem does your project {value} solve?",
    "What was your specific contribution to {value}?",
    "What was the most difficult part of developing {value}?",
    "Why did you choose the technologies used in {value}?",
    "How would you improve {value} if you had more development time?"
]

TECHNICAL_LANGUAGE = [
    "How have you used {value} in your projects?",
    "What are the important concepts of {value} that you understand well?",
    "Describe a problem you solved using {value}.",
    "What are some advantages of using {value}?"
]

TECHNICAL_SKILL = [
    "How have you applied {value} in a real project?",
    "What is your level of experience with {value}, and how did you develop it?",
    "Describe a challenging problem related to {value} that you solved."
]

TECHNICAL_DATABASE = [
    "How have you used {value} in your projects?",
    "What database design considerations do you follow when working with {value}?",
    "Tell me about a database problem you encountered and how you solved it."
]

TECHNICAL_TECHNOLOGY = [
    "Why did you choose {value} for one of your projects?",
    "What are the main advantages of using {value}?",
    "Describe how you have practically worked with {value}."
]

TECHNICAL_EXPERIENCE = [
    "Tell me about your experience as {value}.",
    "What important technical skills did you gain from your experience as {value}?",
    "Describe a challenging situation you handled during your experience as {value}."
]

HR_STRENGTH = [
    "You mentioned {value} as a strength. Can you give me a real example where you demonstrated it?",
    "How has your strength in {value} helped you in your studies or work?",
    "How do you use {value} when facing a difficult situation?"
]

HR_IMPROVEMENT = [
    "You mentioned that you are improving in {value}. What steps are you taking to improve?",
    "Why do you consider {value} an area for improvement?",
    "Can you describe a situation where you noticed the need to improve in {value}?"
]

HR_TEAMWORK = [
    "Tell me about your teamwork experience involving {value}.",
    "How did you contribute to a team while working on {value}?",
    "What did you learn from your teamwork experience with {value}?"
]

HR_LEADERSHIP = [
    "Tell me about a leadership situation involving {value}.",
    "What did you learn from your leadership experience with {value}?",
    "How did you handle responsibility while working on {value}?"
]

HR_ACHIEVEMENT = [
    "Tell me more about your achievement: {value}.",
    "What was the biggest challenge you faced while achieving {value}?",
    "What did you learn from achieving {value}?"
]

HR_INTEREST = [
    "You mentioned {value} as an interest. What do you enjoy about it?",
    "How does your interest in {value} help you personally or professionally?",
    "Tell me something interesting you have learned through {value}."
]

HR_CAREER = [
    "You mentioned {value} as your career goal. Why is this important to you?",
    "What steps are you currently taking toward your goal of {value}?",
    "What skills do you need to develop to achieve {value}?"
]


def clean_values(value):
    if not value:
        return []

    if isinstance(value, list):
        values = value
    else:
        values = str(value).replace("\n", ",").split(",")

    return [
        str(item).strip()
        for item in values
        if str(item).strip()
    ]


def add_question(question_list, question):
    question = str(question).strip()

    if question and question not in question_list:
        question_list.append(question)


def add_templates(question_list, templates, values):
    for value in values:
        for template in templates:
            add_question(
                question_list,
                template.format(value=value)
            )


def build_technical_pool(details):
    questions = []

    projects = clean_values(details.get("projects"))
    languages = clean_values(details.get("languages"))
    skills = clean_values(details.get("technical_skills"))
    databases = clean_values(details.get("databases"))
    technologies = clean_values(details.get("technologies"))
    experience = clean_values(details.get("experience"))

    add_templates(questions, TECHNICAL_PROJECT, projects)
    add_templates(questions, TECHNICAL_LANGUAGE, languages)
    add_templates(questions, TECHNICAL_SKILL, skills)
    add_templates(questions, TECHNICAL_DATABASE, databases)
    add_templates(questions, TECHNICAL_TECHNOLOGY, technologies)
    add_templates(questions, TECHNICAL_EXPERIENCE, experience)

    questions.extend(TECHNICAL_GENERAL)

    random.shuffle(questions)

    return list(dict.fromkeys(questions))


def build_hr_pool(details):
    questions = []

    strengths = clean_values(details.get("strengths"))
    improvements = clean_values(details.get("improvements"))
    teamwork = clean_values(details.get("teamwork"))
    leadership = clean_values(details.get("leadership"))
    achievements = clean_values(details.get("achievements"))
    interests = clean_values(details.get("interests"))
    career_goal = clean_values(details.get("career_goal"))

    add_templates(questions, HR_STRENGTH, strengths)
    add_templates(questions, HR_IMPROVEMENT, improvements)
    add_templates(questions, HR_TEAMWORK, teamwork)
    add_templates(questions, HR_LEADERSHIP, leadership)
    add_templates(questions, HR_ACHIEVEMENT, achievements)
    add_templates(questions, HR_INTEREST, interests)
    add_templates(questions, HR_CAREER, career_goal)

    questions.extend(HR_GENERAL)

    random.shuffle(questions)

    return list(dict.fromkeys(questions))


def prepare_question_pool(session):
    details = session.get("details", {})

    if session.get("type") == "technical":
        return build_technical_pool(details)

    return build_hr_pool(details)


def get_fallback_question(session):
    if not session.get("question_pool"):
        session["question_pool"] = prepare_question_pool(session)

    used_questions = {
        item.get("question")
        for item in session.get("questions", [])
        if item.get("question")
    }

    for question in session.get("question_pool", []):
        if question not in used_questions:
            return question

    return None