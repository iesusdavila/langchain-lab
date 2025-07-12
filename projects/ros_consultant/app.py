import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROP_API_KEY']

if "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True}
    )

    faiss_path = "faiss_ros2_index"
    
    if os.path.exists(f"{faiss_path}"):
        st.info("Cargando FAISS desde disco...")
        st.session_state.vectors = FAISS.load_local(folder_path=faiss_path, embeddings=st.session_state.embeddings, allow_dangerous_deserialization=True)
        st.success("FAISS cargado!")
    else:
        ros2_urls = [
            "https://docs.ros.org/en/humble",
            "https://docs.ros.org/en/humble/Installation.html",
            "https://docs.ros.org/en/humble/Releases.html",
            "https://docs.ros.org/en/humble/Tutorials.html",
            "https://docs.ros.org/en/humble/How-To-Guides.html",
            "https://docs.ros.org/en/humble/Concepts.html",
            "https://docs.ros.org/en/humble/Contact.html",
            "https://docs.ros.org/en/humble/The-ROS2-Project.html",
            "https://docs.ros.org/en/humble/Package-Docs.html",
            "https://docs.ros.org/en/humble/Related-Projects.html",
            "https://docs.ros.org/en/humble/Glossary.html",
            "https://docs.ros.org/en/humble/Citations.html"
        ]
        
        all_docs = []
        with st.spinner("Cargando documentación de ROS2 Humble..."):
            for url in ros2_urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    all_docs.extend(docs)
                    st.write(f"✅ Cargado: {url}")
                except Exception as e:
                    st.write(f"Error cargando {url}: {str(e)}")
        
        st.session_state.docs = all_docs
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        st.session_state.vectors.save_local(faiss_path)
        st.success("FAISS creado y guardado!")

st.title("Asistente Virtual ROS2 Humble")
st.write("Pregunta sobre cualquier aspecto de ROS2 Humble: instalación, tutoriales, conceptos, etc.")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

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
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Escribe tu pregunta sobre ROS2 Humble aquí:")

if prompt:
    with st.spinner("Buscando en la documentación..."):
        start=time.process_time()
        response=retrieval_chain.invoke({"input":prompt})
        response_time = time.process_time()-start
        
    st.write("### Respuesta:")
    st.write(response['answer'])
    
    st.info(f"Tiempo de respuesta: {response_time:.2f} segundos")

    with st.expander("Documentos de referencia utilizados"):
        for i, doc in enumerate(response["context"]):
            st.write(f"**Documento {i+1}:**")
            st.write(doc.page_content)
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                st.write(f"*Fuente: {doc.metadata['source']}*")
            st.write("---")
    