import streamlit as st
from langchain_community.llms.llamacpp import LlamaCpp

st.title("LangChain con Streamlit y llama.cpp")
input_text = st.text_input("Ingresa tu prompt:")

llm = LlamaCpp(
    model_path="models/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf",        
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=10,  
    n_threads=8
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response)  