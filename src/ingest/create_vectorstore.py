import os
import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_DATA_PATH = "data/10ks-raw/"
PROCESSED_DATA_PATH = "data/processed/" 
CHUNKS_FILE_NAME = "chunks.jsonl"

CHUNK_SIZE = 1000  # TODO: Tune these values 
CHUNK_OVERLAP = 200

VECTOR_STORE_DIR = "data/vectorstore"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"




def load_and_split_documents():
    """Loads all documents from the raw 10-k directory and splits them."""
    

    print(f"Loading documents from {RAW_DATA_PATH}...")
    loader = DirectoryLoader(RAW_DATA_PATH, glob="**/*.txt", silent_errors=True)
    documents = loader.load()

    # Split the documents for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    
    chunks_data = [
        {
            "id": i, # Assign a chunk ID
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]
    
    output_path = os.path.join(PROCESSED_DATA_PATH, CHUNKS_FILE_NAME)
    
    print(f"Saving {len(chunks)} chunk metadata to {output_path}...")
    
    
    # Write each chunk as a separate JSON object
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in chunks_data:
            f.write(json.dumps(item) + '\n') 
            
    print(f"✅ Loaded {len(documents)} documents and created {len(chunks)} chunks.")


    return chunks






# Embeds document chunks and saves the vector store locally
def create_and_save_vectorstore(chunks):
    
    # Initialise local embedding model
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Or 'cuda' if you have an NVIDIA GPU
    )

    # Create the vector store
    print(f"Creating and saving ChromaDB to {VECTOR_STORE_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vectorstore.persist()
    print("✅ Vector Store creation complete.")
    
    # Return the vectorstore object if needed
    return vectorstore





if __name__ == "__main__":
    chunks = load_and_split_documents()
    create_and_save_vectorstore(chunks)