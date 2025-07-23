from dotenv import load_dotenv

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()

# AgentState es creado en otro archivo, sus parametros son:
# - input: str, la entrada del usuario
# - agent_outcome: Union[AgentAction, AgentFinish, None], la salida del agente
# - intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
#   una lista de tuplas que representan los pasos intermedios del agente

# react_agent_runnable es un runnable que utiliza el modelo LLM y las herramientas definidas
# para crear un agente que puede razonar y ejecutar acciones basadas en la entrada del usuario
def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

# act_node es un runnable que toma el resultado del agente y ejecuta la herramienta correspondiente
# basándose en la acción del agente. Devuelve los pasos intermedios del agente.
def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    
    # Extrae el nombre de la herramienta y la entrada de AgentAction
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input
    
    # Encuentra la herramienta correspondiente en la lista de herramientas
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    # Ejecutar la herramienta con su input
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"
    
    return {"intermediate_steps": [(agent_action, str(output))]}