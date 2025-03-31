import os
import traceback
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json

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
        add_start_index=True, # Whether to add the length in tokens to the metadata  
        keep_separator=False, # Whether to keep separators in the resulting chunks
        separators=[
            "\n\n",          # Paragraph breaks
            "\n",            # Line breaks
            '""',            # Quotation endings
            '" ',            # End of quoted speech
            ".",             # Sentence endings
            "?", "!",        # Question and exclamation marks
            ";", ":",        # Other punctuation
            ",",             # Commas
            " ",             # Word boundaries
            ""               # Character level (last resort)
        ]
    )
    
    documents = text_splitter.split_documents(raw_docs)
    
    if not documents:
        raise ValueError("Document processed into zero chunks. Check content and splitter settings.")

    # Log the split documents to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Convert documents to a list of dictionaries with relevant information
    split_docs_data = []
    for i, doc in enumerate(documents, 1):
        doc_dict = {
            "chunk_id": i,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "start_index": doc.metadata.get("start_index", 0),
            "end_index": doc.metadata.get("end_index", 0),
            "length": len(doc.page_content)
        }
        split_docs_data.append(doc_dict)
    
    # Save to JSON file
    json_file = output_dir / f"split_documents_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "document_path": document_path,
            "total_chunks": len(documents),
            "chunk_size": text_splitter._chunk_size,
            "chunk_overlap": text_splitter._chunk_overlap,
            "split_documents": split_docs_data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Split documents saved to: {json_file}")
    
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