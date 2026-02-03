import os
import shutil
import pickle

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter 
from llama_index.core.schema import TextNode  

RAW_DATA_PATH = "data/10k-markdown/"
VECTOR_STORE_DIR = "data/vectorstore-v2"
NODES_CACHE_PATH = "data/bm25_nodes/bm25_nodes.pkl"  

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "finder_rag_collection" 

CHUNK_SIZE = 2000  
CHUNK_OVERLAP = 200

def load_and_split_documents():
    print(f"Loading Markdown documents from {RAW_DATA_PATH}...")
    loader = DirectoryLoader(
        RAW_DATA_PATH, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'},
        silent_errors=False
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} files.")

    text_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        file_path = chunk.metadata.get('source', '')
        filename = os.path.basename(file_path)
        company_ticker = filename.rsplit('.', 1)[0]
        chunk.metadata['company_ticker'] = company_ticker
        chunk.page_content = f"[COMPANY: {company_ticker}]\n{chunk.page_content}"
    
    print(f"Created {len(chunks)} Markdown-aware chunks.")
    return chunks

def save_nodes_for_bm25(langchain_chunks):
    print(f"Converting {len(langchain_chunks)} chunks to nodes for BM25...")
    llama_nodes = []
    for chunk in langchain_chunks:
        node = TextNode(
            text=chunk.page_content,
            metadata=chunk.metadata
        )
        llama_nodes.append(node)

    if not os.path.exists(os.path.dirname(NODES_CACHE_PATH)):
        os.makedirs(os.path.dirname(NODES_CACHE_PATH))
        
    with open(NODES_CACHE_PATH, "wb") as f:
        pickle.dump(llama_nodes, f)
    print(f"Saved BM25 nodes to {NODES_CACHE_PATH}")

def create_and_save_vectorstore(chunks):
    if os.path.exists(VECTOR_STORE_DIR):
        print("Removing old vectorstore...")
        shutil.rmtree(VECTOR_STORE_DIR)

    device = "cpu"
    print(f"Embedding with {EMBEDDING_MODEL_NAME} on {device}...")
    
    # 1. Initialize Embeddings with a safe batch size
    # This prevents the model from trying to process too much at once internally
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 32}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print(f"Creating ChromaDB at {VECTOR_STORE_DIR}...")
    
    BATCH_SIZE = 100
    total_chunks = len(chunks)
    
    # Create the DB with the first batch to initialize it
    first_batch = chunks[:BATCH_SIZE]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"Processed batch 1/{(total_chunks // BATCH_SIZE) + 1}")

    # Process the rest
    for i in range(BATCH_SIZE, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Processed batch {(i // BATCH_SIZE) + 1}/{(total_chunks // BATCH_SIZE) + 1}")

    print("Vector Store created successfully.")

if __name__ == "__main__":
    doc_chunks = load_and_split_documents()
    create_and_save_vectorstore(doc_chunks)
    save_nodes_for_bm25(doc_chunks)