"""
group_discussion.py
--------------------
Everything in ONE file: the AI group-discussion engine + the Flask app +
the frontend (HTML/CSS/JS inlined as a string). Drop this single file into
your project and run it.

8 members total: 7 AI personas + 1 human (the site visitor).
Give it a topic -> Claude generates exactly 3 key discussion points ->
the personas discuss them, reacting to each other and to you.

Setup:
    pip install flask anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python group_discussion.py
Then open http://localhost:5000
"""

import os
import json
import random
import uuid

from flask import Flask, request, jsonify, render_template_string
from anthropic import Anthropic

# =============================================================================
# 1. ENGINE — personas, prompts, Claude API calls
# =============================================================================

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5"  # swap for a different model if you want

PERSONAS = [
    {
        "id": "moderator", "name": "Aarav", "role": "Moderator",
        "style": ("Keeps the discussion focused and on-topic. Opens the discussion, "
                  "occasionally summarizes what's been said, and gently steers things "
                  "back if the conversation drifts. Neutral, calm, concise."),
    },
    {
        "id": "expert", "name": "Priya", "role": "Subject-Matter Expert",
        "style": ("Brings facts, definitions, and domain knowledge. Speaks with quiet "
                  "authority, cites the kind of evidence an expert would know, and "
                  "corrects misconceptions politely."),
    },
    {
        "id": "skeptic", "name": "Karthik", "role": "Critical Thinker",
        "style": ("Questions assumptions and asks 'but what about...' style questions. "
                  "Not negative for its own sake, but genuinely probing for weak points "
                  "in an argument."),
    },
    {
        "id": "creative", "name": "Meera", "role": "Creative Innovator",
        "style": ("Thinks in analogies, offers unconventional angles, and connects the "
                  "topic to unexpected ideas. Energetic and imaginative."),
    },
    {
        "id": "pragmatist", "name": "Rohan", "role": "Pragmatist",
        "style": ("Cares about real-world feasibility: cost, time, practicality. "
                  "Grounds the discussion by asking 'how would this actually work?'"),
    },
    {
        "id": "analyst", "name": "Divya", "role": "Data Analyst",
        "style": ("Leans on numbers, trends, and evidence-based reasoning. Speaks in "
                  "measured, precise terms and likes to quantify claims where possible."),
    },
    {
        "id": "devils_advocate", "name": "Sanjay", "role": "Devil's Advocate",
        "style": ("Deliberately takes the counter-position to whatever consensus is "
                  "forming, to stress-test the group's thinking. Respectful but firm."),
    },
]

HUMAN_MEMBER = {"id": "human", "name": "You", "role": "Participant"}
ALL_MEMBERS = PERSONAS + [HUMAN_MEMBER]


def generate_key_points(topic: str) -> list[str]:
    """Ask Claude for exactly 3 concise discussion points on the topic."""
    prompt = (
        f'Topic: "{topic}"\n\n'
        "Give exactly 3 key points worth discussing about this topic. "
        "Each point should be a single, distinct angle (e.g. a benefit, a risk, "
        "a practical consideration) — not overlapping with the others.\n\n"
        "Respond with ONLY valid JSON, no preamble, no markdown fences, in this "
        'exact shape: {"points": ["...", "...", "..."]}'
    )
    response = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    points = data["points"][:3]
    while len(points) < 3:
        points.append("(no point generated)")
    return points


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(discussion has not started yet)"
    return "\n".join(f"{t['name']}: {t['text']}" for t in history)


def generate_member_message(persona: dict, topic: str, key_points: list[str],
                             history: list[dict]) -> str:
    """Generate the next chat message for a single AI persona."""
    points_block = "\n".join(f"- {p}" for p in key_points)

    system_prompt = (
        f"You are {persona['name']}, the {persona['role']} in a group discussion "
        f"of 8 people (7 AI participants and 1 human). Your personality and "
        f"approach: {persona['style']}\n\n"
        "Rules:\n"
        "- Speak in the first person, as yourself, in 1-3 sentences.\n"
        "- Stay strictly in character and stick to your role's perspective.\n"
        "- Build naturally on what others have already said — react, agree, "
        "push back, or add something new. Don't just repeat prior points.\n"
        "- Don't narrate stage directions or prefix your name; just say the "
        "message itself.\n"
        "- Keep it conversational, like real spoken discussion, not an essay."
    )
    user_prompt = (
        f'Topic: "{topic}"\n\n'
        f"Key discussion points:\n{points_block}\n\n"
        f"Discussion so far:\n{_format_history(history)}\n\n"
        f"Now give {persona['name']}'s next contribution to the discussion."
    )
    response = client.messages.create(
        model=MODEL, max_tokens=200,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def pick_next_persona(spoken_ids: list[str]) -> dict:
    """Round-robin: pick a persona who hasn't spoken most recently."""
    counts = {p["id"]: spoken_ids.count(p["id"]) for p in PERSONAS}
    min_count = min(counts.values())
    candidates = [p for p in PERSONAS if counts[p["id"]] == min_count]
    return random.choice(candidates)


def run_opening_round(topic: str) -> dict:
    key_points = generate_key_points(topic)
    history: list[dict] = []

    moderator = PERSONAS[0]
    opening = generate_member_message(moderator, topic, key_points, history)
    history.append({"id": moderator["id"], "name": moderator["name"], "text": opening})

    for persona in [PERSONAS[1], PERSONAS[3]]:
        msg = generate_member_message(persona, topic, key_points, history)
        history.append({"id": persona["id"], "name": persona["name"], "text": msg})

    return {"topic": topic, "key_points": key_points, "history": history}


def advance_discussion(state: dict) -> dict:
    spoken_ids = [t["id"] for t in state["history"] if t["id"] != "human"]
    persona = pick_next_persona(spoken_ids)
    msg = generate_member_message(persona, state["topic"], state["key_points"], state["history"])
    state["history"].append({"id": persona["id"], "name": persona["name"], "text": msg})
    return state


def add_human_message(state: dict, text: str) -> dict:
    state["history"].append({"id": "human", "name": "You", "text": text})
    return advance_discussion(state)


# =============================================================================
# 2. FRONTEND — HTML/CSS/JS inlined as one template string
# =============================================================================

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Group Discussion</title>
<style>
  :root { --bg:#f6f5f2; --panel:#fff; --ink:#23241f; --muted:#6b6b63; --accent:#3b5f4f; --border:#e2dfd6; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  .layout { display:grid; grid-template-columns:220px 1fr; min-height:100vh; }
  .roster { background:var(--panel); border-right:1px solid var(--border); padding:24px 16px; }
  .roster h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom:12px; }
  #member-list { list-style:none; padding:0; margin:0; }
  #member-list li { padding:8px 0; border-bottom:1px solid var(--border); font-size:14px; }
  #member-list li .role { display:block; font-size:12px; color:var(--muted); }
  .board { padding:32px 40px; max-width:760px; }
  header h1 { margin:0 0 16px; font-size:26px; }
  #topic-form { display:flex; gap:8px; }
  #topic-input, #human-input { flex:1; padding:10px 12px; border:1px solid var(--border); border-radius:6px; font-size:14px; }
  button { padding:10px 16px; border:none; border-radius:6px; background:var(--accent); color:#fff; font-size:14px; cursor:pointer; }
  button:hover { opacity:.9; }
  .key-points { margin-top:24px; padding:16px; background:var(--panel); border:1px solid var(--border); border-radius:8px; }
  .key-points h3 { margin:0 0 8px; font-size:14px; color:var(--muted); }
  .key-points ol { margin:0; padding-left:20px; }
  .transcript { margin-top:24px; display:flex; flex-direction:column; gap:12px; }
  .message { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 14px; }
  .message.human { background:#eaf1ec; border-color:var(--accent); }
  .message .speaker { font-weight:600; font-size:13px; margin-bottom:4px; }
  .human-form { margin-top:20px; display:flex; gap:8px; flex-wrap:wrap; }
  .hidden { display:none; }
</style>
</head>
<body>

<div class="layout">
  <aside class="roster">
    <h2>Members</h2>
    <ul id="member-list"></ul>
  </aside>

  <main class="board">
    <header>
      <h1>Group Discussion</h1>
      <form id="topic-form">
        <input id="topic-input" type="text" placeholder="Enter a topic, e.g. Should schools ban smartphones?" required />
        <button type="submit">Start</button>
      </form>
    </header>

    <section id="key-points" class="key-points hidden">
      <h3>Key points</h3>
      <ol id="key-points-list"></ol>
    </section>

    <section id="transcript" class="transcript"></section>

    <form id="human-form" class="human-form hidden">
      <input id="human-input" type="text" placeholder="Add your point to the discussion..." autocomplete="off" />
      <button type="submit">Send</button>
      <button type="button" id="next-btn">Let the group continue</button>
    </form>
  </main>
</div>

<script>
let sessionId = null;
const topicForm = document.getElementById("topic-form");
const topicInput = document.getElementById("topic-input");
const humanForm = document.getElementById("human-form");
const humanInput = document.getElementById("human-input");
const nextBtn = document.getElementById("next-btn");
const transcript = document.getElementById("transcript");
const keyPointsSection = document.getElementById("key-points");
const keyPointsList = document.getElementById("key-points-list");
const memberList = document.getElementById("member-list");

function renderMembers(members) {
  memberList.innerHTML = "";
  members.forEach((m) => {
    const li = document.createElement("li");
    li.innerHTML = `${m.name}<span class="role">${m.role}</span>`;
    memberList.appendChild(li);
  });
}

function renderHistory(history) {
  transcript.innerHTML = "";
  history.forEach((turn) => {
    const div = document.createElement("div");
    div.className = "message" + (turn.id === "human" ? " human" : "");
    div.innerHTML = `<div class="speaker">${turn.name}</div><div>${turn.text}</div>`;
    transcript.appendChild(div);
  });
  transcript.scrollTop = transcript.scrollHeight;
}

topicForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = topicInput.value.trim();
  if (!topic) return;
  transcript.innerHTML = "<p>Starting discussion...</p>";

  const res = await fetch("/api/discussion/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  const data = await res.json();
  if (data.error) { transcript.innerHTML = `<p>${data.error}</p>`; return; }

  sessionId = data.session_id;
  renderMembers(data.members);

  keyPointsList.innerHTML = "";
  data.key_points.forEach((p) => {
    const li = document.createElement("li");
    li.textContent = p;
    keyPointsList.appendChild(li);
  });
  keyPointsSection.classList.remove("hidden");

  renderHistory(data.history);
  humanForm.classList.remove("hidden");
});

humanForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!sessionId) return;
  const text = humanInput.value.trim();
  if (!text) return;

  const res = await fetch(`/api/discussion/${sessionId}/human`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  humanInput.value = "";
  renderHistory(data.history);
});

nextBtn.addEventListener("click", async () => {
  if (!sessionId) return;
  const res = await fetch(`/api/discussion/${sessionId}/next`, { method: "POST" });
  const data = await res.json();
  renderHistory(data.history);
});
</script>
</body>
</html>
"""

# =============================================================================
# 3. FLASK APP — API routes + serves the page above
# =============================================================================

app = Flask(__name__)

# In-memory demo storage. Swap for Redis/DB in production.
SESSIONS: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/discussion/start", methods=["POST"])
def start_discussion():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    state = run_opening_round(topic)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = state

    return jsonify({
        "session_id": session_id,
        "topic": state["topic"],
        "key_points": state["key_points"],
        "history": state["history"],
        "members": [{"name": p["name"], "role": p["role"]} for p in PERSONAS]
                    + [{"name": "You", "role": "Participant"}],
    })


@app.route("/api/discussion/<session_id>/next", methods=["POST"])
def next_turn(session_id):
    state = SESSIONS.get(session_id)
    if not state:
        return jsonify({"error": "unknown session_id"}), 404
    state = advance_discussion(state)
    SESSIONS[session_id] = state
    return jsonify({"history": state["history"]})


@app.route("/api/discussion/<session_id>/human", methods=["POST"])
def human_turn(session_id):
    state = SESSIONS.get(session_id)
    if not state:
        return jsonify({"error": "unknown session_id"}), 404

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    state = add_human_message(state, text)
    SESSIONS[session_id] = state
    return jsonify({"history": state["history"]})


@app.route("/api/discussion/<session_id>", methods=["GET"])
def get_discussion(session_id):
    state = SESSIONS.get(session_id)
    if not state:
        return jsonify({"error": "unknown session_id"}), 404
    return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
