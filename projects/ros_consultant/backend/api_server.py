import uvicorn
from fastapi import FastAPI
from config import GROQ_API_KEY
from langserve import add_routes
from langchain_groq import ChatGroq
from vector_store_manager import DocumentManager
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def simple_rag_wrapper(question: str):
    return retrieval_chain.invoke({"input": question})

def simple_rag_quiz_wrapper(question: str):
    return retrieval_chain_quiz.invoke({"input": question})

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

doc_manager = DocumentManager()
vectors = doc_manager.load_vectors_store()
retriever = vectors.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 8, 'fetch_k': 50}
            )

llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Eres un experto en ROS2 Humble. Responde las preguntas basándote únicamente en el contexto proporcionado de la documentación oficial de ROS2 Humble.
Proporciona respuestas precisas, detalladas y útiles. Si la información no está en el contexto, indica que no tienes esa información específica en la documentación cargada.

Contexto de la documentación de ROS2 Humble:
{context}

Pregunta: {input}

Respuesta:
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

promp_quiz=ChatPromptTemplate.from_template(
"""Eres un experto en ROS2 Humble. Responde las preguntas basándote únicamente en el contexto proporcionado de la documentación oficial de ROS2 Humble.
Proporciona varias preguntas de opción múltiple con una respuesta correcta y varias incorrectas. Asegúrate de que las preguntas sean claras y concisas.
Realiza un minimo de 5 preguntas y un máximo de 10. Cada pregunta debe tener una respuesta correcta y 2 respuestas incorrectas.

Contexto de la documentación de ROS2 Humble:
{context}

Pregunta: {input}

Respuesta:
"""
)
document_chain_quiz = create_stuff_documents_chain(llm, promp_quiz)
retrieval_chain_quiz = create_retrieval_chain(retriever, document_chain_quiz)

simple_rag_chain = RunnableLambda(simple_rag_wrapper)
simple_rag_quiz_chain = RunnableLambda(simple_rag_quiz_wrapper)

add_routes(
    app,
    llm,
    path="/llm"
)

add_routes(
    app,
    simple_rag_chain,
    path="/ros2_consultant"
)

add_routes(
    app,
    simple_rag_quiz_chain,
    path="/ros2_quiz"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
