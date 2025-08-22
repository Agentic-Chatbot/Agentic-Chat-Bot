import os
import time
import streamlit as st
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# -------------------------------
# Setup
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ No Hugging Face API token found. Please set HUGGINGFACEHUB_API_TOKEN in your .env")

#import os
print("HF Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


print("✅ Loaded Hugging Face Token:", HF_TOKEN[:10], "...")

# Single clean LLM definition
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",  # you can also use llama-3
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
    )
)

# -------------------------------
# Define Models
# -------------------------------
class WorkshopSubtask(BaseModel):
    id: int
    title: str = Field(description="Short name for the subtask")
    assigned_to: str = Field(description="Which agent is responsible")
    progress: List[str] = Field(description="Progress updates by this agent")

class WorkshopPlan(BaseModel):
    subtasks: List[WorkshopSubtask]

parser = PydanticOutputParser(pydantic_object=WorkshopPlan)

# -------------------------------
# Prompt Template for Subtasks
# -------------------------------
workshop_prompt = PromptTemplate(
    input_variables=["topic", "audience", "date"],
    template="""
You are a planner for a robotics workshop.  
Break down the preparation into 3–5 subtasks.  

Each subtask must include:
- title (a few words only)
- assigned_to (choose from: Logistics, Content, Marketing, Communication, Tech Support)
- progress (list of 3–4 updates that sound human-like, e.g., "Looking into venues", "Almost done", "Completed ✅")

Topic: {topic}  
Audience: {audience}  
Date: {date}  

{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# -------------------------------
# Prompt for Final Summary
# -------------------------------
summary_prompt = PromptTemplate(
    input_variables=["steps"],
    template="""
You are the Master Organizer.  
The following subtasks were executed:\n{steps}\n
Now merge them into a single, polished workshop plan.  
Keep it concise and structured with sections for Logistics, Content, Marketing, and Other Notes.
"""
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🤖 Robotics Workshop Planner", layout="centered")
st.title("🤖 Robotics Workshop Agentic Planner")

topic = st.text_input("Workshop Topic", "Introduction to Robotics")
audience = st.number_input("Audience Size", 20, 500, 100)
date = st.date_input("Workshop Date")

if st.button("Run Workflow"):
    # Step 1: Generate subtasks
    with st.spinner("Generating subtasks..."):
        chain = workshop_prompt | llm | parser
        plan = chain.invoke({"topic": topic, "audience": audience, "date": date})

    # Step 2: Show each subtask with animated logs
    st.subheader("📌 Subtasks in Progress")
    executed = []
    for subtask in plan.subtasks:
        st.markdown(f"### {subtask.title}")
        st.caption(f"👤 Assigned to: {subtask.assigned_to}")

        progress_text = ""
        for update in subtask.progress:
            progress_text += f"- {update}\n"
            st.markdown(progress_text)
            time.sleep(0.6)

        executed.append(f"{subtask.assigned_to}: {subtask.title}")

    # Step 3: Summarize into final plan
    with st.spinner("Compiling final plan..."):
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"steps": "\n".join(executed)})

    st.subheader("📋 Final Workshop Plan")
    st.write(summary)  # no .content needed, already string
    st.success("✅ Workflow Complete")