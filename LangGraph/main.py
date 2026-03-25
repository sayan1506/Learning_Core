from langgraph.graph import StateGraph, START, END
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict
from langchain.messages import AnyMessage

load_dotenv()
import os

model = ChatMistralAI(model="mistral-small-latest",api_key=os.getenv("MISTRAL_API_KEY"))



Tavily_client = TavilyClient(api_key=os.getenv("TAVILYT_API_KEY"))

@tool
def searchInternet(query: str):
    """
    Docstring for searchInternet
    """
    result = Tavily_client.search(query=query)

    return str(result["results"])

tools = [searchInternet]
tools_dict = {tool.name: tool for tool in tools}

model_with_tools = model.with_tools(tools_dict)

