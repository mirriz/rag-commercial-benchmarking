import os
import pickle
import chromadb
import torch
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()

from llama_index.core import (
    VectorStoreIndex, 
    get_response_synthesizer,
    Settings
)
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
GEMINI_MODEL = "gemini-3-flash-preview"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 


RETRIEVAL_TOP_K = 15 
RERANK_TOP_N = 8 

SYSTEM_PROMPT = """
<role>
You are a Senior Equity Research Analyst. Deliver high-conviction, data-backed insights from SEC 10-K filings.
</role>
<critical_constraints>
1. **Target Match:** Identify the company in the question. ONLY use context chunks starting with [COMPANY: Ticker] that match.
2. **Context First:** The retrieved context may contain tables formatted as text. Look for headers above the data rows.
3. **Zero Fluff:** Direct answers only.
</critical_constraints>
<context>
{context_str}
</context>
Question: {query_str}
"""

def load_nodes_for_bm25(cache_path):
    if not os.path.exists(cache_path):
        print(f"BM25 cache not found at {cache_path}.")
        return None
    with open(cache_path, "rb") as f:
        nodes = pickle.load(f)
    return nodes

def initialise_rag_system():
    print("\n--- Initialising RAG System ---")
    if not GOOGLE_API_KEY:
        print("CRITICAL ERROR: GOOGLE_API_KEY not set.")
        return None

    device = "cpu"

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=device,
        embed_batch_size=10, 
        normalize=True
    )
    Settings.embed_model = embed_model

    llm = Gemini(
        model=GEMINI_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
        system_prompt=SYSTEM_PROMPT,
        timeout=1000
    )
    Settings.llm = llm 

    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL, 
        top_n=RERANK_TOP_N,
        device=device
    )

    db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        embed_model=embed_model,
    )
    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)

    nodes = load_nodes_for_bm25(NODES_CACHE_PATH)
    
    if nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=RETRIEVAL_TOP_K
        )
        final_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=RETRIEVAL_TOP_K,
            num_queries=1, 
            mode="reciprocal_rerank",
            use_async=True,
            llm=llm 
        )
    else:
        final_retriever = vector_retriever

    query_engine = RetrieverQueryEngine.from_args(
        retriever=final_retriever,
        node_postprocessors=[reranker],
        llm=llm
    )

    print("--- Initialisation Complete ---")
    return query_engine

def run_rag_query(query_engine, question):
    if query_engine is None: return "System failed.", []
    print(f"\n> Querying: {question}")
    response = query_engine.query(question)
    return str(response), response.source_nodes


if __name__ == "__main__":
    rag_engine = initialise_rag_system()
    sample_question = "Amgen's liquidity risk amid capex allocation."
    answer, sources = run_rag_query(rag_engine, sample_question)
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    for i, node in enumerate(sources):
        print(f"Source {i+1} Metadata: {node}")