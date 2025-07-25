from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict

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
    new_number = state["number"] + 3
    return Command(
        goto=END, 
        update={
            "text": state["text"] + "c",
            "number": new_number,
            "all_numbers": state["all_numbers"] + [new_number]
        }
    )

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.set_entry_point("node_a")

app = graph.compile()

response = app.invoke({
    "text": "",
    "number": 0,
    "all_numbers": []
})

print(response)