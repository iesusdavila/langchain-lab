from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.llms.llamacpp import LlamaCpp
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# Agent: WikipediaQueryRun
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# Agent: WebBaseLoader
loader = WebBaseLoader("https://iesusdavila.vercel.app/cv")
docs = loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

model_path = "models/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/db9d1fe0f31addb4978201b2bf3e577f3f8900d2/gguf/mxbai-embed-large-v1-f16.gguf"
if not os.path.exists(model_path):
    print(f"Model path does not exist. Please check the path.")

llama = LlamaCppEmbeddings(
    model_path=model_path,
)

vectordb = FAISS.from_documents(documents, llama)
retriever = vectordb.as_retriever()

retriever_tool=create_retriever_tool(retriever, "web_search", "Busca información sobre el señor Davila")

# Agent: Arvix
arxiv_wrapper = ArxivAPIWrapper(top_k_results= 1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper= arxiv_wrapper)

# Create a list of tools
tools = [wiki, retriever_tool, arxiv]

# LLM for the agent
model_llm = LlamaCpp(
    model_path="models/Mistral-7B-Instruct-v0.3.IQ4_XS.gguf",
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=20,  
    n_threads=8,
)

# Get the prompt template from the hub
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(model_llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

agent_executor.invoke({"input":"El señor Davila estuvo alguna vez en Corea del Sur en la ciudad de Incheon?"})
agent_executor.invoke({"input":"What's the paper 1605.08386 about?"})

