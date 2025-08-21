import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
subtask_model = ChatGroq(model="llama3-70b-8192", temperature=0.3)
mapping_model = ChatGroq(model="llama3-70b-8192", temperature=0)

parser = JsonOutputParser()

subtask_prompt = ChatPromptTemplate.from_template("""
You are an assistant that breaks down a user task into smaller subtasks.  

Rules:  
- Avoid big vague tasks, always split them into manageable subtasks.  
- Keep 3–6 subtasks maximum.  
- No duplicates.  
- Return only a numbered list of subtasks.  

Task: "{query}"
""")

map_prompt = ChatPromptTemplate.from_template("""
You are an AI that converts subtasks into structured actions for execution.  

For each subtask, return a JSON object with:  
- "subtask": the original subtask  
- "action_type": choose from [send_mail, book_classroom, generate_poster, arrange_equipment, schedule_meeting, execute_task, generate_form, generate_certificate]  
- "description": short explanation of how the action would be performed  

Return the result as an array of JSON objects only. No extra text.  

Subtasks:  
{subtasks}
""")

def get_subtasks(query: str):
    """Convert query into clean subtasks."""
    chain = subtask_prompt | subtask_model
    response = chain.invoke({"query": query})
    lines = response.content.strip().split("\n")
    subtasks = [line.lstrip("123456.-•* ").strip() for line in lines if line.strip()]
    return subtasks

def map_subtasks(subtasks: list):
    """Map subtasks into structured actions."""
    chain = map_prompt | mapping_model | parser
    response = chain.invoke({"subtasks": "\n".join(subtasks)})
    return response

def process_query(query: str):
    """Main pipeline: query → subtasks → actions."""
    subtasks = get_subtasks(query)
    mapped = map_subtasks(subtasks)
    return subtasks, mapped
