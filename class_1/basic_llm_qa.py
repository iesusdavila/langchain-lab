import streamlit as st
from langchain_community.llms.llamacpp import LlamaCpp

st.title("LangChain con Streamlit y llama.cpp")
input_text = st.text_input("Ingresa tu prompt:")

llm = LlamaCpp(
    model_path="models/models--MaziyarPanahi--Llama-3.2-1B-Instruct-GGUF/snapshots/b64ae94264258a3d7516a34a8c54928d32b19869/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=10,  
    n_threads=8
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response)  
    print(dir(llm.client))
    st.write(f"Porcentaje de contexto usado: {(llm.client.n_tokens / llm.client.n_ctx()) * 100:.2f}%")