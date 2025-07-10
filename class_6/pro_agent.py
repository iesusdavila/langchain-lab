from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROP_API_KEY']

class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")  # we're forcing tool use again
        )

    def invoke(self, input: str) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_call = self.agent.invoke({
                "input": input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            # add initial tool call to scratchpad
            agent_scratchpad.append(tool_call)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]
            tool_out = name2tool[tool_name].func(**tool_args)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            # add a print so we can see intermediate steps
            print(f"{count}: {tool_name}({tool_args})")
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history
        final_answer = tool_out["answer"] if isinstance(tool_out, dict) else tool_out
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return json.dumps(tool_out)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def exponentiate(a: float, b: float) -> float:
    """Raise the first number to the power of the second."""
    return a ** b

@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user."""
    return answer

tools = [add, subtract, multiply, exponentiate, final_answer]

name2tool = {tool.name: tool for tool in tools}

llm=ChatGroq(groq_api_key=groq_api_key,temperature=0,model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant. Use tools to perform calculations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_executor = CustomAgentExecutor()

response = agent_executor.invoke(input="What is 10.35 added to 11.7?")

print(response)

# agent: RunnableSerializable = (
#     {
#         "input": lambda x: x["input"],
#         "chat_history": lambda x: x["chat_history"],
#         "agent_scratchpad": lambda x: x.get("agent_scratchpad",[])
#     }
#     | prompt
#     | llm.bind_tools(tools, tool_choice="auto")
# )

# tool_call = agent.invoke({"input": "What is 10 added 10", "chat_history": []})

# print(tool_call.tool_calls[0]["name"])
# print(tool_call.tool_calls[0]["args"])

# name_func_tool = name2tool[tool_call.tool_calls[0]["name"]]

# tool_exec_content =name_func_tool.func(**tool_call.tool_calls[0]["args"])

# print(tool_exec_content)