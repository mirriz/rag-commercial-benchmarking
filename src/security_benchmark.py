import json
import os
from pipelines.rag_pipeline import initialise_rag_system, run_rag_query

# Settings
SECURITY_DATASET_PATH = "data/security_benchmark_dataset/security_benchmark_dataset.json"
OUTPUT_DIR = "results/"
OUTPUT_FILE = "commercial_rag_security_benchmark_results.json"


def run_security_benchmark():
    print("--- STARTING SECURITY BENCHMARK ---")
    
    # Load Security Dataset
    if not os.path.exists(SECURITY_DATASET_PATH):
        print(f"Error: Security dataset not found at {SECURITY_DATASET_PATH}")
        return

    with open(SECURITY_DATASET_PATH, 'r') as f:
        security_tests = json.load(f)
    
    print(f"Loaded {len(security_tests)} security test cases")

    # Initialise RAG System
    rag_engine = initialise_rag_system()
    if not rag_engine:
        return

    # Run Attacks
    print("\nRunning Attacks...")

    for test_case in security_tests:
        attack_id = test_case['id']
        question = test_case['question']
        
        print(f"Processing {attack_id}...")
        
        try:
            # Run the attack through the RAG pipeline
            generated_answer, source_nodes = run_rag_query(rag_engine, question)
        
            test_case['model_response'] = generated_answer

        except Exception as e:
            print(f"Error: {e}")
            test_case['model_response'] = f"Error: {str(e)}"


    # Save the Updated JSON 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    with open(output_path, 'w') as f:
        json.dump(security_tests, f, indent=4)
        

    print(f"Benchmark Complete")
    print(f"Results appended and saved to: {output_path}")

if __name__ == "__main__":
    run_security_benchmark()