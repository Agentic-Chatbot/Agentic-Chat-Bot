
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field
import threading

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if HF_TOKEN is None:
    raise ValueError("❌ No Hugging Face API token found. Please set HUGGINGFACEHUB_API_TOKEN in your .env")

# Debug print for token (comment out in production)
print("DEBUG HF Token:", HF_TOKEN[:6] + "..." if HF_TOKEN else "❌ No token found")

# -------------------------------
# Hugging Face model setup
# -------------------------------
# ❌ WRONG (before): You directly passed endpoint_url into ChatHuggingFace
# ✅ FIXED (now): Wrap with HuggingFaceEndpoint first
repo_id = "distilgpt2"  # <-- change this to your desired model

from langchain_huggingface import HuggingFaceEndpoint

hf_endpoint = HuggingFaceEndpoint(
    repo_id="distilgpt2",             # ✅ public API-enabled model
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
    max_new_tokens=512
)

llm = ChatHuggingFace(
    llm=hf_endpoint  # ✅ Correct usage
)

# -------------------------------
# Output schema for subtasks
# -------------------------------
class Subtask(BaseModel):
    title: str = Field(description="The title of the subtask")
    description: str = Field(description="Details about the subtask")
    assigned_to: str = Field(description="The role responsible")

class WorkshopPlan(BaseModel):
    subtasks: List[Subtask]

# -------------------------------
# Prompts
# -------------------------------
workshop_prompt = PromptTemplate.from_template(
    """
    You are planning a workshop.
    Topic: {topic}
    Audience: {audience}
    Date: {date}
    Break it into subtasks with titles, descriptions, and responsible roles.
    """
)

summary_prompt = PromptTemplate.from_template(
    """
    Summarize the workshop plan in 3–4 sentences.
    Steps:
    {steps}
    """
)

# -------------------------------
# Parser
# -------------------------------
parser = PydanticOutputParser(pydantic_object=WorkshopPlan)

# -------------------------------
# Worker function (background thread)
# -------------------------------
def run_workflow_in_thread(topic, audience, date):
    st.session_state.is_running = True
    st.session_state.error = None
    st.session_state.plan_result = None
    st.session_state.summary_result = None

    try:
        # (Optional) Test Hugging Face API reachability
        print("🔎 Testing Hugging Face inference...")
        test_resp = requests.post(
            f"https://api-inference.huggingface.co/models/{repo_id}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": "Hello from agt.py, just a test!"}
        )
        if test_resp.status_code != 200:
            print("HF API Call ❌ Error:", test_resp.text)
            st.session_state.error = f"Hugging Face API error {test_resp.status_code}: {test_resp.text}"
            return
        else:
            print("HF API Call ✅ Success")

        # Step 1: Generate subtasks
        chain = workshop_prompt | llm | parser
        plan = chain.invoke({"topic": topic, "audience": audience, "date": date})
        st.session_state.plan_result = plan

        # Step 2: Summarize plan
        executed = [f"{s.assigned_to}: {s.title}" for s in plan.subtasks]
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"steps": "\n".join(executed)})
        st.session_state.summary_result = summary

    except Exception as e:
        st.session_state.error = f"❌ Workflow error: {e}"
        print("Exception in workflow:", e)
    finally:
        st.session_state.is_running = False

# -------------------------------
# Streamlit UI
# -------------------------------
# st.title("🚀 Workshop Planner")
#
# topic = st.text_input("Workshop Topic")
# audience = st.text_input("Audience")
# date = st.text_input("Date (e.g., 21 Aug 2025)")
#
# if "is_running" not in st.session_state:
#     st.session_state.is_running = False
#
# if st.button("Generate Plan") and not st.session_state.is_running:
#     threading.Thread(target=run_workflow_in_thread, args=(topic, audience, date)).start()
#
# # UI updates handled in main thread only
# if st.session_state.is_running:
#     st.info("⏳ Running workflow... Please wait...")
#
# if st.session_state.error:
#     st.error(st.session_state.error)
#
# if st.session_state.plan_result:
#     st.subheader("📌 Workshop Plan")
#     for sub in st.session_state.plan_result.subtasks:
#         st.markdown(f"- **{sub.title}** ({sub.assigned_to}): {sub.description}")
#
# if st.session_state.summary_result:
#     st.subheader("📝 Summary")
#     st.write(st.session_state.summary_result)
st.title("🚀 Workshop Planner")

topic = st.text_input("Workshop Topic")
audience = st.text_input("Audience")
date = st.text_input("Date (e.g., 21 Aug 2025)")

# -------------------------------
# Initialize session_state keys if they don't exist
# -------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "error" not in st.session_state:
    st.session_state.error = None
if "plan_result" not in st.session_state:
    st.session_state.plan_result = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

# -------------------------------
# Trigger workflow
# -------------------------------
if st.button("Generate Plan") and not st.session_state.is_running:
    threading.Thread(target=run_workflow_in_thread, args=(topic, audience, date)).start()

# -------------------------------
# UI updates handled in main thread only
# -------------------------------
if st.session_state.is_running:
    st.info("⏳ Running workflow... Please wait...")

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.plan_result:
    st.subheader("📌 Workshop Plan")
    for sub in st.session_state.plan_result.subtasks:
        st.markdown(f"- **{sub.title}** ({sub.assigned_to}): {sub.description}")

if st.session_state.summary_result:
    st.subheader("📝 Summary")
    st.write(st.session_state.summary_result)

