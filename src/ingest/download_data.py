from datasets import load_dataset, load_from_disk
import os

QA_DOWNLOAD_PATH = "data/finder_data"
QA_DATA_PATH = "data/finder_data/train"





def load_finder_qa_benchmark():
    """Loads the FinDER QA dataset"""

    if os.path.exists(QA_DOWNLOAD_PATH):
        print(f"FinDER QA data already exists at {QA_DOWNLOAD_PATH}")
        return load_from_disk(QA_DOWNLOAD_PATH)

    print("Loading FinDER QA dataset...")
    dataset = load_dataset("Linq-AI-Research/FinDER") 

    # Save to local machine
    dataset.save_to_disk(QA_DOWNLOAD_PATH)
    print(f"FinDER QA benchmark saved to {QA_DOWNLOAD_PATH}")
    return dataset





def get_qa_benchmark_data():
    """Extracts questions and answers from FinDER dataset"""

    # Check if the correct data folder exists
    if not os.path.exists(QA_DATA_PATH):
        print(f"Error: QA data not found at {QA_DATA_PATH}")
        print("Please run the full download_data.py to acquire data.")
        return None, None, None
    

    print(f"Loading QA benchmark from disk: {QA_DATA_PATH}")
    

    dataset_split = load_from_disk(QA_DATA_PATH) 
    
    # Extract Questions, Answers and Context from the Dataset object
    questions = dataset_split['text']
    answers = dataset_split['answer']
    contexts = dataset_split['references']
    
    # Use the length of the loaded data for verification
    print(f"Loaded {len(questions)} Q&A pairs directly from the split.")
    
    return questions, answers, contexts







if __name__ == "__main__":

    load_finder_qa_benchmark()

    questions, answers, contexts = get_qa_benchmark_data()

    print(f"Questions: {len(questions)}")
    print(f"Answers: {len(answers)}")
    print(f"Context: {len(contexts)}")