from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
import time
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
groq_api_key=os.environ['GROP_API_KEY']

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True}
)

faiss_path = "faiss_ros2_index"
    
if os.path.exists(f"{faiss_path}"):
    print("Cargando FAISS desde disco...")
    vectors = FAISS.load_local(folder_path=faiss_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    print("FAISS cargado!")
else:
    url_humble = "https://docs.ros.org/en/humble/"
    info_urls = [
        "index.html",
        "Installation.html",
        "Releases.html",
        "Tutorials.html",
        "Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html",
        "Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html",
        "Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html",
        "Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html",
        "Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html",
        "Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Parameters/Understanding-ROS2-Parameters.html",
        "Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Actions/Understanding-ROS2-Actions.html",
        "Tutorials/Beginner-CLI-Tools/Using-Rqt-Console/Using-Rqt-Console.html",
        "Tutorials/Beginner-CLI-Tools/Launching-Multiple-Nodes/Launching-Multiple-Nodes.html",
        "Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html",
        "Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html",
        "Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html",
        "Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html",
        "Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html",
        "Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html",
        "Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Service-And-Client.html",
        "Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html",
        "Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html",
        "Tutorials/Beginner-Client-Libraries/Single-Package-Define-And-Use-Interface.html",
        "Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-CPP.html",
        "Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html",
        "Tutorials/Beginner-Client-Libraries/Getting-Started-With-Ros2doctor.html",
        "Tutorials/Beginner-Client-Libraries/Pluginlib.html",
        "Tutorials/Intermediate/Rosdep.html",
        "Tutorials/Intermediate/Creating-an-Action.html",
        "Tutorials/Intermediate/Writing-an-Action-Server-Client/Cpp.html",
        "Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html",
        "Tutorials/Intermediate/Writing-a-Composable-Node.html",
        "Tutorials/Intermediate/Composition.html",
        "Tutorials/Intermediate/Using-Node-Interfaces-Template-Class.html",
        "Tutorials/Intermediate/Monitoring-For-Parameter-Changes-CPP.html",
        "Tutorials/Intermediate/Launch/Launch-Main.html",
        "Tutorials/Intermediate/Tf2/Tf2-Main.html",
        "Tutorials/Intermediate/Testing/Testing-Main.html",
        "Tutorials/Intermediate/URDF/URDF-Main.html",
        "Tutorials/Intermediate/RViz/RViz-Main.html",
        "Tutorials/Advanced/Topic-Statistics-Tutorial/Topic-Statistics-Tutorial.html",
        "Tutorials/Advanced/Discovery-Server/Discovery-Server.html",
        "Tutorials/Advanced/Allocator-Template-Tutorial.html",
        "Tutorials/Advanced/Ament-Lint-For-Clean-Code.html",
        "Tutorials/Advanced/FastDDS-Configuration.html",
        "Tutorials/Advanced/Recording-A-Bag-From-Your-Own-Node-CPP.html",
        "Tutorials/Advanced/Recording-A-Bag-From-Your-Own-Node-Py.html",
        "Tutorials/Advanced/ROS2-Tracing-Trace-and-Analyze.html",
        "Tutorials/Advanced/Reading-From-A-Bag-File-CPP.html",
        "Tutorials/Advanced/Simulators/Simulation-Main.html",
        "Tutorials/Advanced/Security/Security-Main.html",
        "Tutorials/Demos/Quality-of-Service.html",
        "Tutorials/Demos/Managed-Nodes.html",
        "Tutorials/Demos/Intra-Process-Communication.html",
        "Tutorials/Demos/Rosbag-with-ROS1-Bridge.html",
        "Tutorials/Demos/Real-Time-Programming.html",
        "Tutorials/Demos/dummy-robot-demo.html",
        "Tutorials/Demos/Logging-and-logger-configuration.html",
        "Tutorials/Demos/Content-Filtering-Subscription.html",
        "Tutorials/Demos/Wait-for-Acknowledgment.html",
        "Tutorials/Miscellaneous/Deploying-ROS-2-on-IBM-Cloud.html",
        "Tutorials/Miscellaneous/Eclipse-Oxygen-with-ROS-2-and-rviz2.html",
        "Tutorials/Miscellaneous/Building-Realtime-rt_preempt-kernel-for-ROS-2.html",
        "Tutorials/Miscellaneous/Building-ROS2-Package-with-eclipse-2021-06.html",
        "How-To-Guides.html",
        "Concepts.html",
        "Contact.html",
        "The-ROS2-Project.html",
        "Package-Docs.html",
        "Related-Projects.html",
        "Glossary.html",
        "Citations.html"
    ]
    
    all_docs = []

    for info_url in info_urls:
        try:
            url = f"{url_humble}{info_url}"
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ Cargado: {url}")
        except Exception as e:
            print(f"Error cargando {url}: {str(e)}")
    
    docs = all_docs
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_documents=text_splitter.split_documents(docs)
    vectors=FAISS.from_documents(final_documents,embeddings)
    vectors.save_local(faiss_path)
    print("FAISS creado y guardado!")

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
retriever = vectors.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 8, 'fetch_k': 50}
            )
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def simple_rag_wrapper(question: str):
    return retrieval_chain.invoke({"input": question})

simple_rag_chain = RunnableLambda(simple_rag_wrapper)

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

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
