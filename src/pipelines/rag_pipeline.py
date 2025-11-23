import os
import chromadb
import torch

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.langchain import LangchainEmbedding 
from llama_index.vector_stores.chroma import ChromaVectorStore 

# Re-ranking imports
from llama_index.core.postprocessor import SentenceTransformerRerank

from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
VECTOR_STORE_DIR = "data/vectorstore"
COLLECTION_NAME = "finder_rag_collection" 

# Advanced RAG: Matching the ingestion model
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Reranker Model (Cross-Encoder)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "llama3" 

# Retrieval Settings
RETRIEVAL_TOP_K = 20  # Fetch a broad net of candidates
RERANK_TOP_N = 5      # Filter down to the absolute best 5 for the LLM

SYSTEM_PROMPT = """
You are an expert financial analyst assistant.  
Your goal is to provide accurate, concise answers based ONLY on provided context from SEC 10-K filings.
Focus on finding the relevant answer for the specified company in the question. 

Guidelines:
1. Identify the specific company and metric requested.
2. If the context contains the answer, extract the exact number or text.
3. If a calculation is required (e.g. Gross Profit), perform it step-by-step using data from the context.
4. If the context does NOT contain the answer, say "Data not available in context."
5. Do not hallucinate or use outside knowledge.
"""

def initialise_rag_system():
    """
    Initialises self-hosted RAG query engine with BGE Embeddings and Re-ranking.
    """
    print("\n--- Initialising RAG System ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Initialise Embedding Model
    try:
        base_embed = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
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
        llm = LlamaIndexOllama(
            model=OLLAMA_MODEL, 
            temperature=0.0, 
            request_timeout=300, 
            system_prompt=SYSTEM_PROMPT
        )
        print(f"Local LLM '{OLLAMA_MODEL}' connected")
    except Exception as e:
        print(f"Error initialising Ollama LLM: {e}")
        return None

    # Initialise Re-ranker
    try:
        print(f"Loading Re-ranker: {RERANKER_MODEL}...")
        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL, 
            top_n=RERANK_TOP_N,
            device=device
        )
        print("Re-ranker loaded.")
    except Exception as e:
        print(f"Error loading Re-ranker: {e}")
        return None

    # 5. Create the Query Engine with Post-Processing
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=RETRIEVAL_TOP_K, # Retrieve 20
        node_postprocessors=[reranker]    # Filter to 5
    )

    print("--- Initialisation Complete ---")
    return query_engine

def run_rag_query(query_engine, question):
    """Executes a RAG query and returns the answer and source chunks"""
    if query_engine is None:
        return "RAG system failed to initialise.", []

    print(f"\n> Querying: {question}")
    
    # Execute the query
    response = query_engine.query(question)
    
    answer = str(response)
    source_nodes = response.source_nodes
    
    return answer, source_nodes

if __name__ == "__main__":
    rag_query_engine = initialise_rag_system()

    if rag_query_engine:
        # Example Test
        test_question = "What was the Gross profit for Analog Devices (ADI) in 2024?"
        
        answer, sources = run_rag_query(rag_query_engine, test_question)

        print("\n" + "="*70)
        print(f"| Final Answer:")
        print(f"| {answer.strip()}")
        print("="*70)
        
        if sources:
            print(f"| Retrieved & Re-ranked {len(sources)} Source Chunks:")
            for i, node in enumerate(sources):
                score = round(node.score, 4)
                # Text usually comes with the [COMPANY] tag now
                cleaned_text = node.text[:200].replace('\n', ' ')
                print(f"| --- Rank {i+1} (Score: {score}) ---")
                print(f"| {cleaned_text}...")
        else:
            print("No source nodes retrieved")
        print("="*70)