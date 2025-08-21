import json
from typing import List, Dict, Any
import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(backend.py), '..')))

BACKEND_IMPORT_ERROR = None
try:
    from backend import get_subtasks, map_subtasks
except Exception as e:
    BACKEND_IMPORT_ERROR = e

st.set_page_config(page_title="ChatBot Demo", page_icon="🤖", layout="wide")
st.title("🤖 ChatBot — Demo UI")

if BACKEND_IMPORT_ERROR:
    st.error(f"Backend imports failed:\n\n{BACKEND_IMPORT_ERROR}")
    st.stop()

ACTION_EMOJI = {
    "send_mail": "✉️",
    "book_classroom": "🏫",
    "arrange_equipment": "🧰",
    "generate_poster": "🖼️",
    "schedule_meeting": "🗓️",
    "execute_task": "⚙️",
    "generate_form": "📝",
    "generate_certificate": "🏅",
}

MOCK_LOGS = {
    "send_mail": ["Generating mail content...", "Connecting to server...", "Sending mail...", "✅ Mail sent!"],
    "book_classroom": ["Checking classroom availability...", "Submitting booking...", "✅ Classroom booked!"],
    "arrange_equipment": ["Listing equipment...", "Contacting manager...", "✅ Equipment arranged!"],
    "generate_poster": ["Drafting poster...", "Adding event details...", "✅ Poster ready!"],
    "schedule_meeting": ["Checking schedules...", "Creating invite...", "✅ Meeting scheduled!"],
    "execute_task": ["Breaking into subtasks...", "Assigning roles...", "✅ Task executed!"],
    "generate_form": ["Creating form...", "Adding fields...", "✅ Form published!"],
    "generate_certificate": ["Fetching details...", "Generating certificates...", "✅ Certificates ready!"],
}

query = st.text_input("Enter your query:", "Organize a robotics club workshop")

if st.button("Run Pipeline"):
    with st.spinner("Generating subtasks..."):
        subtasks: List[str] = get_subtasks(query)

    st.subheader("📌 Subtasks")
    for i, s in enumerate(subtasks, 1):
        st.write(f"{i}. {s}")

    with st.spinner("Mapping subtasks to actions..."):
        mapped = map_subtasks(subtasks)
        mapped = json.loads(mapped) if isinstance(mapped, str) else mapped

    st.subheader("⚡ Mapped Actions")
    for i, m in enumerate(mapped, 1):
        emoji = ACTION_EMOJI.get(m.get("action_type", ""), "🔧")
        st.write(f"{i}. {emoji} **{m['action_type']}** — {m['subtask']}")

    st.subheader("📜 Agent Logs")
    for i, m in enumerate(mapped, 1):
        action = m["action_type"]
        emoji = ACTION_EMOJI.get(action, "🔧")
        with st.expander(f"{i}. {emoji} {m['subtask']}", expanded=(i == 0)):
            for step in MOCK_LOGS.get(action, ["Processing...", "✅ Done!"]):
                st.write(step)
