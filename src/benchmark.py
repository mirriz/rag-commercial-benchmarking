import os
import pandas as pd
import json
import time

# Import your existing functions
from ingest.download_data import get_qa_benchmark_data
from pipelines.rag_pipeline import initialise_rag_system, run_rag_query


OUTPUT_DIR = "results/"
OUTPUT_FILE = "local_rag_v2_evaluation_dataset.json"
SAMPLE_SIZE = 500 # Set to None to run the FULL dataset

def run_benchmark():
    print("STARTING LOCAL RAG BENCHMARKING")

    # Load Benchmark Data 
    print("\nLoading FinDER QA Benchmark Data...")
    try:
        questions, ground_truths, ground_truth_contexts = get_qa_benchmark_data()
        
        if not questions:
            print("Error: No data found. Please run src/ingest/download_data.py")
            return

        if SAMPLE_SIZE:
            print(f"Limiting run to first {SAMPLE_SIZE}")
            questions = questions[:SAMPLE_SIZE]
            ground_truths = ground_truths[:SAMPLE_SIZE]
            ground_truth_contexts = ground_truth_contexts[:SAMPLE_SIZE]
        else:
            print(f"Running on full dataset.")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialise local RAG system
    print("\nInitialising Local RAG Pipeline...")
    rag_engine = initialise_rag_system()
    
    if not rag_engine:
        print("Failed to initialise RAG system")
        return

    print("\nRunning Queries w/ Local LLM...")
    
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    total_queries = len(questions)
    start_time = time.time()

    for i, query in enumerate(questions):

        print(f"Processing query {i+1}")

        try:
            # Run the query
            generated_answer, source_nodes = run_rag_query(rag_engine, query)
            
            # Retrieved context for RAGAs
            retrieved_contexts = [node.text for node in source_nodes]
            
            # Store data
            results["question"].append(query)
            results["answer"].append(generated_answer)
            results["contexts"].append(retrieved_contexts)
            results["ground_truth"].append(ground_truths[i])
            
        except Exception as e:
            print(f"\nError processing query {i}: {e}")
            results["question"].append(query)
            results["answer"].append("Error generating response")
            results["contexts"].append([])
            results["ground_truth"].append(ground_truths[i])
            continue

    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds ({elapsed_time/total_queries:.2f}s per query).")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Ensure Output Directory Exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Save as a standard JSON list
    df.to_json(output_path, orient="records", indent=4)
    
    print(f"Benchmark Complete!")
    print(f"Results saved to: {output_path}")
    
    # PReview
    print("\nOutput:")
    print(df[['question', 'answer']].head(2))

if __name__ == "__main__":
    run_benchmark()