import os
import json
from dotenv import load_dotenv
from datasets import Dataset 

from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.run_config import RunConfig

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 

load_dotenv()

# Settings
INPUT_FILE = "results/local_rag_v2_evaluation_dataset.json"
OUTPUT_FILE = "results/ragas_scores_for_localRAG.csv"

JUDGE_MODEL = "gpt-5.1"

# Low Cost
#JUDGE_MODEL = "gpt-5-mini" 

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

def load_evaluation_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return None
    
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    ragas_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for entry in data:
        # Filter out failed responses
        if entry.get("answer") == "Error generating response" or not entry.get("answer"):
            print("Not all answers generated. quitting evaluation")
            return None
            
        ragas_dict["question"].append(entry["question"])
        ragas_dict["answer"].append(entry["answer"])
        ragas_dict["contexts"].append(entry["contexts"] if isinstance(entry["contexts"], list) else []) 
        ragas_dict["ground_truth"].append(entry["ground_truth"])

    return Dataset.from_dict(ragas_dict)

def run_ragas_evaluation():
    print("--- STARTING RAGAS EVALUATION WITH OPENAI GPT 5.1---")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables")
        return

    dataset = load_evaluation_data()
    if not dataset: return

    print(f"Initialising Judge {JUDGE_MODEL}")
    

    # add model
    llm_judge = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        timeout=300
    )
    
    print(f"Initialising Local Embeddings (Model: {EMBEDDING_MODEL})...")
    embeddings_judge = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}, # Change to 'cpu' if you don't have a GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # Metrics
    metrics = [
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy,
        answer_correctness
    ]

    print(f"Running evaluation on {len(dataset)} samples...")
    
    try:
        my_run_config = RunConfig(
            max_workers=2,
            timeout=300
        )

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_judge,
            embeddings=embeddings_judge,
            run_config=my_run_config,
            raise_exceptions=False,
        )

        print("\nEvaluation Complete")
        print(results)
        
        df_results = results.to_pandas()
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"\nScores saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\nERROR DURING EVALUATION: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()