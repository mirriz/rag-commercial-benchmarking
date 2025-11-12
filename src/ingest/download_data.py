from datasets import load_dataset
import os

def load_finder_dataset(save_dir="data/raw"):
    """
    Loads the FinDER dataset from and saves it locally
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading FinDER dataset from Hugging Face...")
    dataset = load_dataset("Linq-AI-Research/FinDER")

    # Save to disk for offline work
    # dataset.save_to_disk(os.path.join(save_dir, "finder"))
    # print(f"FinDER dataset saved to {os.path.join(save_dir, 'finder')}")

if __name__ == "__main__":
    load_finder_dataset()
