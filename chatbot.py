import streamlit as st
import time

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")

st.title("🤖 Agentic Chatbot")
st.markdown("### 👋 Welcome to the Agentic Chatbot!")
st.markdown("This chatbot breaks a request into subtasks and shows how agents complete them step by step.")


if "messages" not in st.session_state:
    st.session_state.messages = []

def break_into_subtasks(query):
    if "workshop" in query.lower():
        return ["Book venue", "Email coordinator", "Prepare poster", "Announce event"]
    elif "project" in query.lower():
        return ["Define project", "Assign tasks", "Track progress", "Submit work"]
    else:
        return ["Understand request", "Plan steps", "Execute task"]

def agent_progress(task):
    return [
        f"Agent for '{task}': Started working on the task...",
        f"Agent for '{task}': Gathering required resources and planning details...",
        f"Agent for '{task}': Currently making progress and handling key steps...",
        f"Agent for '{task}': Task completed successfully ✅"
    ]

if st.sidebar.button("Organize a robotics workshop"):
    query = "Organize a robotics workshop"

    st.session_state.messages.append(("user", query))
    st.chat_message("user").write(query)

    subtasks = break_into_subtasks(query)

    for task in subtasks:
        logs = agent_progress(task)
        for log in logs:
            time.sleep(0.8)  
            st.session_state.messages.append(("assistant", log))
            st.chat_message("assistant").write(log)


for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
