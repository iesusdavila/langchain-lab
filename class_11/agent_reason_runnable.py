from langchain_groq import ChatGroq
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['TAVILY_API_KEY'] = os.environ.get('TAVILY_API_KEY', '')
groq_api_key=os.environ['GROP_API_KEY']
llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Get the current system time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic")
react_prompt = hub.pull("hwchase17/react")

tools = [get_system_time, search_tool]

react_agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)