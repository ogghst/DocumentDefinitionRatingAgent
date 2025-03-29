import os
import traceback
from typing import List, Optional

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStoreRetriever

from llm_setup import initialize_embeddings

def load_document(document_path: str) -> List[Document]:
    """Load and chunk a document file."""
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document file not found at: {document_path}")

    loader = Docx2txtLoader(document_path)
    raw_docs = loader.load()
    
    if not raw_docs or not raw_docs[0].page_content.strip():
        raise ValueError("Document is empty or could not be loaded/parsed.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    
    documents = text_splitter.split_documents(raw_docs)
    
    if not documents:
        raise ValueError("Document processed into zero chunks. Check content and splitter settings.")
    
    return documents

def create_retriever(documents: List[Document]) -> VectorStoreRetriever:
    """Create a vector store retriever from document chunks."""
    try:
        embeddings = initialize_embeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        return retriever
    except Exception as e:
        print(f"ERROR creating retriever: {e}")
        traceback.print_exc()
        raise 