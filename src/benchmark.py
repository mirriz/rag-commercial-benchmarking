import os
import pandas as pd
import json
import time
from ingest.download_data import get_qa_benchmark_data
from pipelines.rag_pipeline import initialise_rag_system, run_rag_query

OUTPUT_DIR = "results/"
OUTPUT_FILE = "commercial_rag_evaluation_dataset.json"
SAMPLE_SIZE = 500

def run_benchmark():
    try:
        questions, ground_truths, ground_truth_contexts = get_qa_benchmark_data()
        
        if not questions:
            return

        if SAMPLE_SIZE:
            questions = questions[:SAMPLE_SIZE]
            ground_truths = ground_truths[:SAMPLE_SIZE]
            ground_truth_contexts = ground_truth_contexts[:SAMPLE_SIZE]
    except Exception:
        return

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    processed_lookup = {}

    if os.path.exists(output_path):
        try:
            df_existing = pd.read_json(output_path, orient="records")
            if not df_existing.empty:
                for _, row in df_existing.iterrows():
                    if row["answer"] and row["answer"] != "Error generating response":
                        processed_lookup[row["question"]] = row.to_dict()
        except Exception:
            pass

    indices_to_run = [i for i, q in enumerate(questions) if q not in processed_lookup]

    rag_engine = None
    if indices_to_run:
        rag_engine = initialise_rag_system()

    final_results = []

    for i, query in enumerate(questions):
        if query in processed_lookup:
            final_results.append(processed_lookup[query])
            continue

        try:
            generated_answer, source_nodes = run_rag_query(rag_engine, query)
            retrieved_contexts = [node.text for node in source_nodes]
            
            new_entry = {
                "question": query,
                "answer": generated_answer,
                "contexts": retrieved_contexts,
                "ground_truth": ground_truths[i]
            }
            final_results.append(new_entry)

        except Exception:
            error_entry = {
                "question": query,
                "answer": "Error generating response",
                "contexts": [],
                "ground_truth": ground_truths[i]
            }
            final_results.append(error_entry)

        df = pd.DataFrame(final_results)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_json(output_path, orient="records", indent=4)

if __name__ == "__main__":
    run_benchmark()