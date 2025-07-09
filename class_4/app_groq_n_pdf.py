import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROP_API_KEY']

def vector_embedding():
    print("=="*20)
    print("Entrando al vector embedding")
    print("=="*20)
    if "vectors" not in st.session_state:
        model_path = "models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf"
        if not os.path.exists(model_path):
            print(f"Model path does not exist. Please check the path.")

        st.session_state.embeddings = LlamaCppEmbeddings(
            model_path=model_path,
        )

        st.session_state.loader=PyPDFDirectoryLoader("class_4_project_1/pdf") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

st.title("ChatGroq Demo")
st.session_state.model_llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    print("=="*20)
    print("Entrando al promp1")
    print("=="*20)
    document_chain = create_stuff_documents_chain(st.session_state.model_llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()

    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    