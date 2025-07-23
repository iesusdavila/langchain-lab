import operator
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish

"""
- La salida es AgentAction o AgentFinish, pero puede ser None al inicio
- La salida es una lista de tuplas (AgentAction, str) que representan los
pasos intermedios del agente, donde cada tupla contiene la acción del agente
y la salida de la herramienta ejecutada.
- Esto permite rastrear el progreso del agente y las herramientas utilizadas.
- Se usa Annotated para agregar metadatos a la lista, como el tipo de
operación que se aplica a la lista (en este caso, operator.add).
"""
class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None] 
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]