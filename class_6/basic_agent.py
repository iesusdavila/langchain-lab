import os
from langchain_core.tools import tool
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

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

memory = ConversationBufferMemory(
    memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
    return_messages=True  # to return Message objects
)

llm = LlamaCpp(
    model_path="models/Mistral-7B-Instruct-v0.3.IQ4_XS.gguf",
    n_ctx=2048,
    temperature=0.0,
    verbose=True,
    n_gpu_layers=10,  
    n_threads=8
)

tools = [add, subtract, multiply, exponentiate]
prompt = hub.pull("hwchase17/react")

# Crear agente ReAct
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

response = agent_executor.invoke({
    "input": "What is 10.7 added 7.68?",
    "chat_history": memory.chat_memory.messages,
})

print("="*20)
print(response["output"])

# Continuar con respuestas para aumentar la memoria
response = agent_executor.invoke({
    "input": "What is 10.7 subtracted from 7.68?",
    "chat_history": memory.chat_memory.messages,
})

print("="*20)
print(response["output"])

response = agent_executor.invoke({
    "input": "My name is James",
    "chat_history": memory
})

print("="*20)
print(response["output"])

response = agent_executor.invoke({
    "input": "What is my name?",
    "chat_history": memory.chat_memory.messages,
})

print("="*20)
print(response["output"])

