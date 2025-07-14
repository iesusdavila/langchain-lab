import uvicorn
from fastapi import FastAPI
from config import GROQ_API_KEY
from langserve import add_routes
from langchain_groq import ChatGroq
from vector_store_manager import RetrieverManager
from prompts import PROMPT_CONSULTANT, PROMPT_QUIZ
from langchain_core.runnables import RunnableLambda
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

retriever_manager = RetrieverManager()
retriever = retriever_manager.get_retriever()

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

document_chain = create_stuff_documents_chain(llm, PROMPT_CONSULTANT)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

document_chain_quiz = create_stuff_documents_chain(llm, PROMPT_QUIZ)
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
