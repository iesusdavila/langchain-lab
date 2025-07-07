import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.llms.llamacpp import LlamaCpp

if "vectors" not in st.session_state:
    model_path = "models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf"
    if not os.path.exists(model_path):
        print(f"Model path does not exist. Please check the path.")

    st.session_state.embeddings = LlamaCppEmbeddings(
        model_path=model_path,
    )
    st.session_state.loader=WebBaseLoader("https://iesusdavila.vercel.app/cv")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Llamacpp Demo")
model_llm = LlamaCpp(
    model_path="models/Mistral-7B-Instruct-v0.3.IQ4_XS.gguf",
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=20,  
    n_threads=8,
)

prompt = PromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(model_llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    