import os
import chromadb
import torch


from llama_index.core import VectorStoreIndex

from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.langchain import LangchainEmbedding 
from llama_index.vector_stores.chroma import ChromaVectorStore 

from langchain_community.embeddings import HuggingFaceEmbeddings


# Settings
VECTOR_STORE_DIR = "data/vectorstore"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "finder_rag_collection" 
OLLAMA_MODEL = "llama3" 
SIMILARITY_TOP_K = 5 


def initialise_rag_system():
    """
    Initializes self-hosted RAG query engine by connecting to the
    vector store and the local LLM 
    """
    print("\n--- Initialising Self-Hosted System ---")
    
    # Initialise Embedding Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        base_embed = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device} 
        )
        embed_model = LangchainEmbedding(base_embed)
        print(f"Embedding Model '{EMBEDDING_MODEL_NAME}' initialised")
    except Exception as e:
        print(f"Error initialising embedding model: {e}")
        return None

    # Connect to ChromaDB
    try:
        db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            embed_model=embed_model,
        )
        print(f"Vector Store '{COLLECTION_NAME}' loaded from disk")
    except Exception as e:
        print(f"Error loading ChromaDB: {e}. Ensure create_vectorstore.py was run")
        return None

    # Initialise Local LLM 
    try:
        llm = LlamaIndexOllama(model=OLLAMA_MODEL, temperature=0.0, request_timeout=300)
        print(f"Local LLM '{OLLAMA_MODEL}' connected directly via LlamaIndexOllama")
    except Exception as e:
        print(f"Error initialising Ollama LLM: {e}. Is Ollama application running?")
        return None

    # Create the Query Engine
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    print("--- Initialisation Complete ---")
    return query_engine

def run_rag_query(query_engine, question):
    """Executes a RAG query and returns the answer and source chunks."""
    if query_engine is None:
        return "RAG system failed to initialize.", []

    print(f"\n> Querying Self-Hosted System: {question}")
    
    # Execute the query
    response = query_engine.query(question)
    
    answer = str(response)
    source_nodes = response.source_nodes
    
    return answer, source_nodes

if __name__ == "__main__":
    rag_query_engine = initialise_rag_system()

    if rag_query_engine:
        test_question = "Retrive data about NVDA"
        
        answer, sources = run_rag_query(rag_query_engine, test_question)

        print("\n" + "="*70)
        print(f"| Final Answer:")
        cleaned_answer = answer.strip().replace('\n', ' ') 
        print(f"| {cleaned_answer}")
        print("="*70)
        
        if sources:
            print(f"| Retrieved {len(sources)} Source Chunks:")
            for i, node in enumerate(sources):
                metadata = node.metadata
                source_file = metadata.get('source', 'N/A').split('\\')[-1].split('/')[-1]
                score = round(node.score, 4)
                
                cleaned_chunk_text = node.text[:180].replace('\n', ' ')
                print(f"| --- Source {i+1} ({source_file}, Score: {score}) ---")
                print(f"| ... {cleaned_chunk_text} ...")
        else:
            print("No source nodes retrieved")
        print("="*70)