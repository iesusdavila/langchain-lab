from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import os

# Un retriever (recuperador - traducido al español) es una interfaz que devuelve documentos 
# dada una consulta no estructurada. Es más general que un vector store.

# Un retriever no necesita poder almacenar documentos, solo devolverlos (o recuperarlos). 
# Los vector store pueden utilizarse como la columna vertebral de un retriever, pero 
# también existen otros tipos de retrievers.

# Retrieval chain (Cadena de recuperación): Esta cadena recibe la consulta de un usuario,
# que se envía al retriever (recuperador) para obtener los documentos relevantes. 
# Estos documentos (y las entradas originales) se envían a un LLM para generar una respuesta.

QUERY_TO_RAG = "Cual es el numero de telefono de Iesus?"
QUERY_TO_RAG_2 = "Cual es el correo electronico de Iesus?"
QUERY_TO_RAG_3 = "Donde ha trabajado Iesus?"
QUERY_TO_RAG_4 = "Cual es la formacion academica de Iesus?"
QUERY_TO_RAG_5 = "Cuantos y cuales idiomas habla Iesus?"

loader=PyPDFLoader("class_3/CV_ES.pdf")
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

model_path = "models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf"
if not os.path.exists(model_path):
    print(f"Model path does not exist. Please check the path.")

llama = LlamaCppEmbeddings(
    model_path=model_path,
)

db_faiss = FAISS.from_documents(documents, llama)

retireved_results=db_faiss.similarity_search(QUERY_TO_RAG)
print("=="*20)
print(retireved_results)

print("=="*20)
model_llm = LlamaCpp(
    model_path="models/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf",        
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=20,  
    n_threads=8,
)

prompt = PromptTemplate.from_template("""
    Responda la siguiente pregunta basándose únicamente en el contexto 
    proporcionado. Piense paso a paso antes de dar una respuesta detallada.
    Unicamente devuelve la respuesta a la pregunta, sin explicaciones adicionales 
    ni texto adicional. Si no hay suficiente información en el contexto,
    responde con 'No tengo suficiente información para responder a esta pregunta'.
    <context>
    {context}
    </context>
    Pregunta: {input}""")

document_chain=create_stuff_documents_chain(model_llm, prompt)

retriever=db_faiss.as_retriever()

retrieval_chain=create_retrieval_chain(retriever, document_chain)

response=retrieval_chain.invoke({"input":QUERY_TO_RAG})
response2=retrieval_chain.invoke({"input":QUERY_TO_RAG_2})
response3=retrieval_chain.invoke({"input":QUERY_TO_RAG_3})
response4=retrieval_chain.invoke({"input":QUERY_TO_RAG_4})
response5=retrieval_chain.invoke({"input":QUERY_TO_RAG_5})  

print("=="*20)
print(response['answer'])
print("=="*20)
print(response2['answer'])
print("=="*20)
print(response3['answer'])
print("=="*20)
print(response4['answer'])
print("=="*20)
print(response5['answer'])
