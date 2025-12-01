import os
import pickle
import chromadb
import torch

from llama_index.core import (
    VectorStoreIndex, 
    get_response_synthesizer,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding 
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_DIR = "data/vectorstore-v2"
COLLECTION_NAME = "finder_rag_collection" 
NODES_CACHE_PATH = "data/bm25_nodes/bm25_nodes.pkl" 

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_MODEL = "llama3" 

# Retrieval Settings
RETRIEVAL_TOP_K = 75  # Fetch a broad net of candidates
RERANK_TOP_N = 10      # Filter down to the best 10 for the LLM

SYSTEM_PROMPT = """
You are the world's best financial analyst with extensive knowledge of SEC 10-K filings.
Your goal is to provide accurate, concise, and direct answers based primarily on the provided context.

CRITICAL RULES:
1. Check Company Labels: The context chunks start with [COMPANY: Ticker]. You MUST ONLY use chunks that match the company asked about in the user's question.
2. Direct Answers: Answer the question immediately. Do not start with "The query is asking about..." or "Based on the provided context...".


Guidelines:
1. Identify the specific company and answer requested.
2. If the context contains the answer, extract the exact number or text.
3. You are permitted to do calculation. If a calculation is required (e.g. Gross Profit), perform it step-by-step using data from the context. Whenever this is the case, show your working.
4. Do not hallucinate or use outside knowledge.
5. State the response / explanation ALONE.
6. Give reasoning and explanation for your answer. Single word or one sentence answers are not acceptable.
7. Do not hallucinate specific numbers or facts about the company. Use general knowledge only for definitions or standard accounting principles when context is missing.
"""


def load_nodes_for_bm25(cache_path):
    """
    Loads BM25 nodes from pickle cache
    """
    if not os.path.exists(cache_path):
        print(f"BM25 cache not found.")
        return None
        
    print(f"Loading nodes for BM25 from {cache_path}...")
    with open(cache_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"Loaded {len(nodes)} nodes.")
    return nodes



def initialise_rag_system():
    """
    Initialises self-hosted RAG query engine with BGE Embeddings and Re-ranking.
    """
    print("\n--- Initialising RAG System ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"




    # Initialise Embedding Model
    try:
        base_embed = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        embed_model = LangchainEmbedding(base_embed)
        print(f"Embedding Model '{EMBEDDING_MODEL}' initialised")
    except Exception as e:
        print(f"Error initialising embedding model: {e}")
        return None



        # Initialise Local LLM
    try:
        llm = Ollama(
            model=OLLAMA_MODEL, 
            temperature=0.0, 
            request_timeout=300, 
            system_prompt=SYSTEM_PROMPT
        )
        Settings.llm = llm
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




    # Connect to ChromaDB
    try:
        db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            embed_model=embed_model,
        )

        # Vector Retriever
        vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)

        print(f"Vector Store '{COLLECTION_NAME}' loaded from disk")

    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None
    



    # Create Hybrid Retriever
    nodes = load_nodes_for_bm25(NODES_CACHE_PATH)
    
    if nodes:
        # BM25 Retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=RETRIEVAL_TOP_K
        )
        print("BM25 Retriever initialised.")

        # Fuse Retrievers
        print("Combining Vector and BM25 retrievers...")
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=RETRIEVAL_TOP_K,
            num_queries=1,  # Use original query only
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True
        )
        final_retriever = hybrid_retriever
    else:
        print("Falling back to Vector-only retrieval.")
        final_retriever = vector_retriever




    # Create the Query Engine with Post-Processing
    query_engine = RetrieverQueryEngine.from_args(
        retriever=final_retriever,
        node_postprocessors=[reranker],
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize", 
            llm=llm
        ),
        llm=llm,
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
        # Test
        test_question = "Catherine R. Clay's role transition in innovation derivatives at Cboe Global Markets Inc. (CBOE) showcases its competitiveness."
        
        answer, sources = run_rag_query(rag_query_engine, test_question)

        print("\n" + "="*70)
        print(f"| Final Answer:")
        print(f"| {answer.strip()}")
        print("="*70)
        
        if sources:
            print(f"| Retrieved & Re-ranked {len(sources)} Source Chunks:")
            for i, node in enumerate(sources):
                score = round(node.score, 4)
                cleaned_text = node.text[:200].replace('\n', ' ')
                print(f"| --- Rank {i+1} (Score: {score}) ---")
                print(f"| {cleaned_text}...")
        else:
            print("No source nodes retrieved")
        print("="*70)