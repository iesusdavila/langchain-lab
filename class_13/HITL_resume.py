from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

class State(TypedDict):
    text: str
    number: int
    all_numbers: list[int]

def node_a(state: State): 
    print("Node A")
    new_number = state["number"] + 1
    return Command(
        goto="node_b", 
        update={
            "text": state["text"] + "a",
            "number": new_number,
            "all_numbers": state["all_numbers"] + [new_number]
        }
    )

def node_b(state: State): 
    print("Node B")
    new_number = state["number"] + 2
    return Command(
        goto="node_c", 
        update={
            "text": state["text"] + "b",
            "number": new_number,
            "all_numbers": state["all_numbers"] + [new_number]
        }
    )

def node_c(state: State): 
    print("Node C")

    human_response = interrupt("Quieres continuar con la ejecucion? (D/E): ")

    print("Human Review Values: ", human_response)
    new_number = state["number"] + 3
    if(human_response == "D"): 
        return Command(
            goto="node_d", 
            update={
                "text": state["text"] + "c",
                "number": new_number,
                "all_numbers": state["all_numbers"] + [new_number]
            }
        ) 
    elif(human_response == "E"): 
        return Command(
            goto="node_e", 
            update={
                "text": state["text"] + "c",
                "number": new_number,
                "all_numbers": state["all_numbers"] + [new_number]
            }
        )

def node_d(state: State): 
    print("Node D")
    new_number = state["number"] + 4
    return Command(
        goto=END, 
        update={
            "text": state["text"] + "d",
            "number": new_number,
            "all_numbers": state["all_numbers"] + [new_number]
        }
    )

def node_e(state: State): 
    print("Node E")
    new_number = state["number"] + 5
    return Command(
        goto=END, 
        update={
            "text": state["text"] + "e",
            "number": new_number,
            "all_numbers": state["all_numbers"] + [new_number]
        }
)

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)
graph.add_node("node_e", node_e)

graph.set_entry_point("node_a") 

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

initialState = {
    "text": "",
    "number": 0,
    "all_numbers": []
}

first_result = app.invoke(initialState, config, stream_mode="updates")
print(first_result)

second_result = app.invoke(Command(resume="E"), config=config, stream_mode="updates")
print(second_result)

