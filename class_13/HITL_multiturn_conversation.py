from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict): 
    twitter_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

def model(state: State): 
    """ Here, we're using the LLM to generate a Twitter post with human feedback incorporated """

    twitter_topic = state["twitter_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]

    input_user = f"""
        Tema de Twitter: {twitter_topic}
        Feedback de la persona: {feedback[-1] if feedback else "Sin feedback todavia"}

        Considerando el feedback previo redefine la respuesta.
    """
    prompt = "Eres un experto en generar contenido de Twitter"

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=input_user)
    ])

    twitter_post_generado = response.content

    print(f"Post generado por IA: {twitter_post_generado}\n")

    return {
        "twitter_topic": twitter_topic,
        "generated_post": [AIMessage(content=twitter_post_generado)], 
        "human_feedback": feedback
    }

def human_node(state: State): 
    """Human Intervention node - loops back to model unless input is done"""

    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post": generated_post, 
        }
    )

    if user_feedback.lower() == "done": 
        return Command(
            update={"human_feedback": state["human_feedback"] + ["Finalised"]}, 
            goto="end_node")

    return Command(
        update={"human_feedback": state["human_feedback"] + [user_feedback]}, 
        goto="model")

def end_node(state: State): 
    """ Final node """
    print("Post final generado:", state["generated_post"][-1])
    return {
        "generated_post": state["generated_post"], 
        "human_feedback": state["human_feedback"]
    }

graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.set_entry_point("model")

graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": 1}}

twitter_topic = input("Introduce el tema del post de Twitter: ")
initial_state = {
    "twitter_topic": twitter_topic, 
    "generated_post": [], 
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if(node_id == "__interrupt__"):
            while True: 
                user_feedback = input("Escribe tu feedback sobre el post generado (o escribe 'done' para finalizar): ")

                app.invoke(Command(resume=user_feedback), config=thread_config)

                if user_feedback.lower() == "done":
                    break