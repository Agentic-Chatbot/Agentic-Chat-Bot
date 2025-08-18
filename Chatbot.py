import streamlit as st
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch
from pydantic import BaseModel,Field
from typing import List,Literal
from dotenv import load_dotenv

def get_subtasks(user_query:str):
    template = PromptTemplate(
    template="""
You are a task planner agent.  
Your job is to take a single user query and break it into clear, actionable subtasks and give them to specific agents.

Rules:  
- Keep subtasks short and specific.  
- Return 3–6 subtasks only (depending on the complexity of the query).  
- For each subtask, create a **list of progress_log entries** as if an agent is performing it.  
- Progress logs should be **short, conversational, and realistic** (e.g., "Starting...", "Gathering resources...", "Halfway done...", "Almost complete...", "Done ✅").  
- Do NOT include dates, numbers, or timestamps.  
- Make the logs **natural and varied** to reflect a human-like agent.

Query: {user_query}  
{format_instruction}
""",
    input_variables=['user_query'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

    chain = template | model | parser

    result = chain.invoke({"user_query": user_query})
    return result
    # for subtask in result.subtasks:
    #     print("Subtask",subtask.id,":",subtask.task)
    #     print("description:",subtask.description)
        

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Subtask(BaseModel):
    id : int
    task : str = Field(description="Title for each subtask")
    agent: Literal['Math Solver Agent','Code Builder Agent','Research Scout Agent','Summarizer Agent','Creative Writer Agent','Organizer Agent','Data Analyst Agent']= Field(default='Organizer Agent',description="""
    Math Solver Agent → solves equations, calculations, analytics.

    Code Builder Agent → writes, explains, or debugs code.

    Research Scout Agent → gathers knowledge, reads sources, finds info.

    Summarizer Agent → condenses text/documents into clear notes.

    Creative Writer Agent → generates posters, taglines, or content.

    Organizer Agent → handles scheduling, task ordering, to-do lists.

    Data Analyst Agent → processes datasets, makes insights.

""")
    # description: str = Field(description="Explanation for each subtask")
    progress_log : List[str] = Field(description="""Each progress_log must be a short, conversational update 
    like an agent would say in chat""")

class Task(BaseModel):
    Agent : str 
    subtasks: List[Subtask]

parser = PydanticOutputParser(pydantic_object=Task)

st.set_page_config(page_title="Agentic Task Planner",
                   page_icon="🤖",
                   initial_sidebar_state="expanded",
                   layout="wide"
                   )

with st.sidebar:
    st.title("🤖 Agentic Task Planner")
    st.subheader("Break down any query into subtasks and watch agents complete them.")
    if st.button("Example: Organize Robotics Workshop"):
        st.session_state["example_query"] = "Organize a robotics workshop"

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

user_query = st.chat_input("Enter your query...") 

if "example_query" in st.session_state and not user_query:
    user_query =  st.session_state.pop("example_query")

if user_query:
    st.session_state["messages"].append({"role":"user","content":user_query})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        log_box = st.empty()

        result = get_subtasks(user_query)
        logs = ""
        for subtask in result.subtasks:
            logs += f"**Subtask : {subtask.task}**\n \n _Agent Assigned : {subtask.agent}_ \n"
            log_box.markdown(logs)
            time.sleep(1)
            for log in subtask.progress_log:
                logs += f"- {log} \n"
                log_box.markdown(logs)
                time.sleep(0.5)
       

        st.session_state["messages"].append({"role": "assistant", "content": logs})


        

    









