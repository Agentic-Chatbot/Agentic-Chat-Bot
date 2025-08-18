from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Subtask(BaseModel):
    id : int
    task : str = Field(description="Title for each subtask")
    description: str = Field(description="Explanation for each subtask")

class Task(BaseModel):
    subtasks: List[Subtask]

parser = PydanticOutputParser(pydantic_object=Task)

template = PromptTemplate(
    template="""You are a task planner agent.  
    Your job is to take a single user query and break it into clear, actionable subtasks.  

    Rules:  
    - Keep subtasks short and specific.  
    - Return 3–6 subtasks only (depending on the complexity of query)

    Query: {user_query} \n {format_instruction}
    """,
input_variables=['user_query'],
partial_variables={"format_instruction":parser.get_format_instructions()}
)

user_query = input("Enter your query : ")

chain = template | model | parser

result = chain.invoke({"user_query": user_query})

for subtask in result.subtasks:
    print("Subtask",subtask.id,":",subtask.task)
    print("description:",subtask.description)

