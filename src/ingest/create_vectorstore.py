import os
import torch
import chromadb
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DATA_PATH = "data/10ks-raw/"
VECTOR_STORE_DIR = "data/vectorstore"

# Switched to BGE-Base (Better than MiniLM)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "finder_rag_collection" 

CHUNK_SIZE = 512  
CHUNK_OVERLAP = 64

def load_and_split_documents():
    """Loads all documents from the raw 10-k directory and splits them."""
    
    print(f"Loading documents from {RAW_DATA_PATH}...")
    loader = DirectoryLoader(RAW_DATA_PATH, glob="**/*.txt", silent_errors=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} source files.")

    # Split the documents for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Prepend the company ticker to the text of every chunk.
    for chunk in chunks:
        file_path = chunk.metadata.get('source', '')
        company = os.path.basename(file_path).replace('.txt', '')
        chunk.page_content = f"[COMPANY: {company}] {chunk.page_content}"
        chunk.metadata['company_ticker'] = company
    
    print(f"Created {len(chunks)} total enriched chunks.")
    return chunks

def create_and_save_vectorstore(chunks):
    """Embeds document chunks and saves the vector store locally"""
    
    # Check if vectorstore exists and clear it to prevent duplicating data on re-runs
    if os.path.exists(VECTOR_STORE_DIR):
        print(f"Removing existing vectorstore at {VECTOR_STORE_DIR} to rebuild...")
        shutil.rmtree(VECTOR_STORE_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialising embedding model '{EMBEDDING_MODEL_NAME}' on {device}...")
    
    # BGE requires specific parameters for best performance
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True} # BGE recommendation

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print(f"Creating and saving ChromaDB to {VECTOR_STORE_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print("Vector Store creation complete.")
    return vectorstore

def check_vectorstore_contents():
    """Loads the vector store and prints the contents of the collection for verification."""
    try:
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection = client.get_collection(COLLECTION_NAME)

        count = collection.count()
        print(f"Total documents (chunks) in collection '{COLLECTION_NAME}': {count}")
        
        # Test output of one chunk
        if count > 0:
            contents = collection.get(limit=1, include=['documents', 'metadatas'])
            print(f"Sample Chunk: {contents['documents'][0][:150]}...")
            print(f"Metadata: {contents['metadatas'][0]}")
        else:
            print("Collection is empty.")
            
    except Exception as e:
        print(f"❌ Error checking vector store contents: {e}")

if __name__ == "__main__":
    doc_chunks = load_and_split_documents()
    
    create_and_save_vectorstore(doc_chunks)
    
    #check_vectorstore_contents()