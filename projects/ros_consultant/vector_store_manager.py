import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import INFO_URLS, FAISS_PATH, URL_HUMBLE_DOC


class DocumentManager:
    def __init__(self, faiss_path: str = FAISS_PATH):
        self.faiss_path = faiss_path
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_documents(self):
        """Carga documentos desde las URLs configuradas"""
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
        """Carga o crea el vector store FAISS"""
        if os.path.exists(self.faiss_path):
            print("Cargando FAISS desde disco...")
            vectors = FAISS.load_local(
                folder_path=self.faiss_path, 
                embeddings=self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("FAISS cargado!")
        else:
            docs = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, self.embeddings)
            vectors.save_local(self.faiss_path)
            print("FAISS creado y guardado!")
        
        return vectors