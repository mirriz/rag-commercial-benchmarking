import os
import pickle
import chromadb
import torch
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to prevent event loop errors in notebooks
nest_asyncio.apply()

from llama_index.core import (
    VectorStoreIndex, 
    get_response_synthesizer,
    Settings
)

# --- CHANGED: Native LlamaIndex Imports (Fixes Compatibility) ---
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

load_dotenv()

# --- Configuration ---
VECTOR_STORE_DIR = "data/vectorstore-v2"
COLLECTION_NAME = "finder_rag_collection" 
NODES_CACHE_PATH = "data/bm25_nodes/bm25_nodes.pkl" 

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5" 
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

GEMINI_MODEL = "gemini-3-flash-preview" # Or "models/gemini-1.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

# Retrieval Settings
RETRIEVAL_TOP_K = 40 
RERANK_TOP_N = 6

SYSTEM_PROMPT = """
<role>
You are a Senior Equity Research Analyst at a top-tier investment bank.
Your goal is to deliver high-conviction, data-backed insights from SEC 10-K filings. 
Prioritise information density over length. Provide the answer immediately, then support it with concise evidence.
</role>

<critical_constraints>
1. **Target Match:** Identify the company in the question, and then identify the ticker using your knowledge. ONLY use context chunks starting with [COMPANY: Ticker] that match the target. Ignore all others.
2. **Zero Fluff:** Start the answer immediately. NEVER use filler phrases like "Based on the context," "The text states," or "In conclusion."
</critical_constraints>

<guidelines>
1. **Structure:** Begin with a direct answer. Follow with bullet points for context/drivers.
2. **Calculations:** If math is required, be compact. Show the logic in a single line (e.g., "($10M - $8M) / $8M = +25%").
3. **Tone:** Use professional, clipped, institutional language. Avoid adjectives and narrative flowery.
4. **Precision:** Extract exact numbers, dates, and names. Do not round unless necessary.
</guidelines>

<context>
{context_str}
</context>

Question: {query_str}
"""

def load_nodes_for_bm25(cache_path):
    """
    Loads BM25 nodes from pickle cache. 
    Ensure this cache is in sync with your ChromaDB!
    """
    if not os.path.exists(cache_path):
        print(f"BM25 cache not found at {cache_path}.")
        return None
        
    print(f"Loading nodes for BM25 from {cache_path}...")
    with open(cache_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"Loaded {len(nodes)} nodes.")
    return nodes

def initialise_rag_system():
    print("\n--- Initialising RAG System ---")
    
    if not GOOGLE_API_KEY:
        print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not set.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialise Embedding Model (Native LlamaIndex)
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device=device,
            embed_batch_size=10, 
            normalize=True
        )
        Settings.embed_model = embed_model # Set global setting
        print(f"Embedding Model '{EMBEDDING_MODEL}' initialised")
    except Exception as e:
        print(f"Error initialising embedding model: {e}")
        return None

    # 2. Initialise Gemini (Native LlamaIndex)
    try:
        llm = Gemini(
            model=GEMINI_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=0.1,
            system_prompt=SYSTEM_PROMPT, # Correct argument for system prompt
            timeout=300
        )
        
        Settings.llm = llm # Set global setting
        print(f"LLM '{GEMINI_MODEL}' connected")

    except Exception as e:
        print(f"Error initialising Gemini LLM: {e}")
        return None

    # 3. Initialise Re-ranker
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

    # 4. Connect to ChromaDB
    try:
        db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            embed_model=embed_model,
        )
        vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
        print(f"Vector Store '{COLLECTION_NAME}' loaded from disk")
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

    # 5. Create Hybrid Retriever
    nodes = load_nodes_for_bm25(NODES_CACHE_PATH)
    
    if nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=RETRIEVAL_TOP_K
        )
        print("BM25 Retriever initialised.")

        print("Combining Vector and BM25 retrievers...")
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=RETRIEVAL_TOP_K,
            num_queries=1, 
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            llm=llm 
        )
        final_retriever = hybrid_retriever
    else:
        print("Warning: BM25 nodes not found. Falling back to Vector-only retrieval.")
        final_retriever = vector_retriever

    # 6. Create Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=final_retriever,
        node_postprocessors=[reranker],
        llm=llm
    )

    print("--- Initialisation Complete ---")
    return query_engine

def run_rag_query(query_engine, question):
    if query_engine is None:
        return "RAG system failed to initialise.", []

    print(f"\n> Querying: {question}")
    response = query_engine.query(question)
    return str(response), response.source_nodes

if __name__ == "__main__":
    rag_query_engine = initialise_rag_system()

    if rag_query_engine:
        test_question = "Cboe's operational stability, governance in cybersecurity, and financial health."
        answer, sources = run_rag_query(rag_query_engine, test_question)

        print("\n" + "="*70)
        print(f"Final Answer:")
        print(f"{answer.strip()}")
        print("="*70)
        
        if sources:
            print(f"| Retrieved & Re-ranked {len(sources)} Source Chunks:")
            for i, node in enumerate(sources):
                score = round(node.score, 4) if node.score else 0.0
                cleaned_text = node.text[:200].replace('\n', ' ')
                print(f"--- Rank {i+1} (Score: {score}) ---")
                print(f"{cleaned_text}...")
        else:
            print("No source nodes retrieved")
        print("="*70)