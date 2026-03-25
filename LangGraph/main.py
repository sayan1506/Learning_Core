import os
from typing import Dict

from dotenv import load_dotenv
from langchain.messages import AnyMessage
from langchain.tools import BaseTool, tool
from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from typing_extensions import Annotated, TypedDict

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("MISTRAL_API_KEY"))


Tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def searchInternet(query: str) -> str:
    """Run a web search and return the raw results string."""
    result = Tavily_client.search(query=query)
    return str(result["results"])


tools = [searchInternet]
tools_dict: Dict[str, BaseTool] = {t.name: t for t in tools}
model_with_tools = model.with_tools(tools_dict)


def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the running message history."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def call_tool(state: AgentState) -> AgentState:
    """Execute requested tools and return their outputs as ToolMessages."""
    last_ai_message = state["messages"][-1]
    tool_calls = getattr(last_ai_message, "tool_calls", []) or []
    tool_results = []

    for tool_call in tool_calls:
        selected_tool = tools_dict[tool_call["name"]]
        result = selected_tool.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(
                content=str(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_results}


def should_continue(state: AgentState) -> str:
    """Route to tools when the model requested them, otherwise end."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    return "tools" if tool_calls else END


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tool)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


if __name__ == "__main__":
    user_question = input("Ask me anything: ")
    initial_state: AgentState = {"messages": [HumanMessage(content=user_question)]}

    for update in app.stream(initial_state, stream_mode="values"):
        latest = update["messages"][-1]
        print(latest.content)

