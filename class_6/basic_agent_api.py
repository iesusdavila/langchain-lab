import os
from langchain_core.tools import tool
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import load_tools

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "")

def get_x_y(input_str: str):
    """Extract x and y from input string."""
    cleaned_str = input_str.strip("'\"")
    parts = cleaned_str.split(",")
    x = float(parts[0].strip())
    y = float(parts[1].strip())
    return x, y

@tool
def add(input_str: str) -> float:
    """Add two numbers. Input format: 'x, y' or '{"x": value, "y": value}'"""
    x, y = get_x_y(input_str)
    return x + y

@tool
def multiply(input_str: str) -> float:
    """Multiply two numbers. Input format: 'x, y' or '{"x": value, "y": value}'"""
    x, y = get_x_y(input_str)
    return x * y

@tool
def exponentiate(input_str: str) -> float:
    """Raise x to the power of y. Input format: 'x, y' or '{"x": value, "y": value}'"""
    x, y = get_x_y(input_str)
    return x ** y

@tool
def subtract(input_str: str) -> float:
    """Subtract x from y. Input format: 'x, y' or '{"x": value, "y": value}'"""
    x, y = get_x_y(input_str)
    return y - x

@tool
def get_location_from_ip():
    """Get the geographical location based on the IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            data = (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {data.get('city', 'N/A')},\n"
                f"Country: {data.get('country', 'N/A')}"
            )
            return data
        else:
            return "Location could not be determined."
    except Exception as e:
        return f"Error occurred: {e}"

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

llm = LlamaCpp(
    model_path="models/Mistral-7B-Instruct-v0.3.IQ4_XS.gguf",
    n_ctx=4096,
    temperature=0.0,
    verbose=True,
    n_gpu_layers=10,  
    n_threads=8
)

toolbox = load_tools(tool_names=['serpapi'], llm=llm)

tools = [add, subtract, multiply, exponentiate, get_location_from_ip, get_current_datetime] + toolbox
prompt = hub.pull("hwchase17/react")

# Crear agente ReAct
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "Tengo algunas preguntas, ¿qué día y hora es ahora? ¿Qué temperatura hace donde estoy?."
})

print("="*20)
print(response["output"])
