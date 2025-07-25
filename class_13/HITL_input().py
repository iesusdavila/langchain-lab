from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq

class State(TypedDict): 
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State): 
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

def get_review_decision(state: State):  
    post_content = state["messages"][-1].content 
    
    print("\nCurrent Twitter Post:\n")
    print(post_content)
    print("\n")

    decision = input("Puedo publicar el post? (yes/no): ")

    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK

def post(state: State):  
    final_post = state["messages"][-1].content  
    print("\nPost Final para publicacion:\n")
    print(final_post)
    print("\nEl post ha sido publicado exitosamente!")

def collect_feedback(state: State):  
    feedback = input("Como puedo mejorar el post? ")
    return {
        "messages": [HumanMessage(content=feedback)]
    }

graph = StateGraph(State)

# 1) LLM genera el post
graph.add_node(GENERATE_POST, generate_post)
# 2) LLM pide confirmación para publicar el post
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
# 3.1) Si el usuario rechaza, se pide feedback
graph.add_node(COLLECT_FEEDBACK, collect_feedback)
# 3.2) Si el usuario acepta, se publica el post
graph.add_node(POST, post)

graph.set_entry_point(GENERATE_POST)

graph.add_conditional_edges(GENERATE_POST, get_review_decision)
# Solamente se cierra el graph si el usuario acepta el post
graph.add_edge(POST, END)
# Si el usuario rechaza, genera el post de nuevo
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

response = app.invoke({
    "messages": [HumanMessage(content="Dame un post de Twitter sobre la importancia de la IA en el futbol.")],
})

print(response)