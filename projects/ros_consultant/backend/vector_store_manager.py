import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import INFO_URLS, FAISS_PATH, URL_HUMBLE_DOC


class RetrieverManager:
    def __init__(self, faiss_path: str = FAISS_PATH):
        self.faiss_path = faiss_path
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_documents(self):
        """Load documents from configured URLs"""
        all_docs = []
        for info_url in INFO_URLS:
            try:
                url = f"{URL_HUMBLE_DOC}{info_url}"
                loader = WebBaseLoader(url)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"Cargado: {url}")
            except Exception as e:
                print(f"Error cargando {url}: {str(e)}")
        return all_docs
    
    def load_vectors_store(self):
        """Load or create the FAISS vector store"""
        if os.path.exists(self.faiss_path):
            print("Loading FAISS from disk...")
            vectors = FAISS.load_local(
                folder_path=self.faiss_path, 
                embeddings=self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("FAISS loaded!")
        else:
            docs = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, self.embeddings)
            vectors.save_local(self.faiss_path)
            print("FAISS created and saved!")
        
        return vectors
    
    def get_retriever(self, search_type="mmr", k=8, fetch_k=50):
        """Returns a configured document retriever"""
        vectors = self.load_vectors_store()
        retriever = vectors.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k, 'fetch_k': fetch_k}
        )

        return retriever