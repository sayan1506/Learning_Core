from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
import os
import subprocess
import sys

load_dotenv()

model = ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("MISTRAL_API_KEY"))


code_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""You are a helpful assistant that can write code to solve problems. You write code in Python. You can also use built-in Python libraries. You should write code that is efficient and easy to understand. You should also include comments in your code to explain what you are doing.""",
)


planner_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""You are a helpful assistant that can plan how to solve problems. You should break down problems into smaller steps and create a plan to solve them. You should also consider different approaches to solving the problem and choose the best one.""",
)




@tool
def execute_code(code: str) -> str:
    """Execute the given code and return the output."""
    print(f"Executing code:\n{code}")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return str({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    })
    
    

tester_agent = create_agent(
    model=model,
    tools=[execute_code],
    system_prompt="""You are a helpful assistant that can test code to ensure it works correctly. You should write test cases to cover different scenarios and edge cases. You should also run the code and check the output to make sure it is correct.""",
)


@tool
def planner_tool(task: str) -> str:
    """Use the planner agent to create a plan for solving the given task."""
    print(f"Planning for task: {task}")
    response = planner_agent.invoke(
        messages=[
            HumanMessage(content=task)
            ]
    )
    # Extract the last message content returned by the planner.
    messages = response.get("messages", [])
    return messages[-1].content if messages else ""

team_lead_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""You are a helpful assistant that can lead a team of agents to solve problems. You should assign tasks to the code agent, planner agent, and tester agent based on their strengths and capabilities. You should also coordinate the efforts of the agents to ensure that they are working together effectively.""",
)


    
response = team_lead_agent.invoke({
    "messages": [
        HumanMessage(content="You are a team lead agent. Your task is to solve the problem of finding the largest prime number less than 100. You have access to a code agent, a planner agent, and a tester agent. The code agent can write code to solve problems, the planner agent can create plans to solve problems, and the tester agent can test code to ensure it works correctly. Please create a plan to solve this problem and assign tasks to the agents accordingly.")
        ]
})
# Agent responses come back as a list of LangChain messages, so print each one.
for message in response.get("messages", []):
    if hasattr(message, "content"):
        print(message.content)

