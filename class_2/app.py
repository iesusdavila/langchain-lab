from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms.llamacpp import LlamaCpp
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)

llm = LlamaCpp(
    model_path="models/models--MaziyarPanahi--Llama-3.2-1B-Instruct-GGUF/snapshots/b64ae94264258a3d7516a34a8c54928d32b19869/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=20,  
    n_threads=8,
    max_tokens=-1
)

prompt = ChatPromptTemplate.from_template("Escribe un resumen del siguiente tema: {topic}")

add_routes(
    app,
    llm,
    path="/llmcpp"
)

add_routes(
    app,
    prompt|llm,
    path="/summary"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
