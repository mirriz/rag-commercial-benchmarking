# RAG Commercial Benchmarking

This repository contains the benchmarking suite for evaluating a commercial Large Language Model (Gemini) augmented with a locally hosted, high-performance Retrieval-Augmented Generation (RAG) pipeline. It is specifically designed to query and reason over complex, unstructured financial data (SEC 10-K filings) using the FinDER QA dataset.

## Architecture & Tech Stack

The RAG pipeline is orchestrated using **LlamaIndex** and utilizes a hybrid retrieval approach to maximize context recall and precision:
* **LLM**: Gemini 3 Flash (`gemini-3-flash-preview`), Llama3:8B
* **Dense Retrieval**: ChromaDB with HuggingFace embeddings (`BAAI/bge-base-en-v1.5`,`all-MiniLM-L6-v2`)
* **Sparse Retrieval**: BM25 keyword matching (cached via Pickle)
* **Re-ranking**: Cross-encoder re-ranking using Sentence Transformers (`BAAI/bge-reranker-v2-m3`)
* **Data Processing**: Pandas, Nest Asyncio


## Getting Started

### Prerequisites
1. Ensure you have Python 3.11.8 installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
You must configure your API keys before running the pipeline. Create a .env file in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Running the Benchmark
To execute the evaluation benchmark across the QA dataset:
```bash
python src/benchmark.py
```
Results will be saved to results/VERSION_TYPE_dataset.json. If the process drops due to a network timeout, re-running the script will pick up exactly where it left off.

### Testing the Pipeline
You can test the engine locally with a single query by running the pipeline script directly:
```bash
python src/pipelines/rag_pipeline.py
```

## Disclaimer

This was developed as part of an academic dissertation. The codebase, evaluation metrics, and generated answers are intended for educational, research, and benchmarking purposes. 

Although this pipeline processes real-world financial data (SEC 10-K filings), **the outputs do not constitute financial, legal, or investment advice.** Large Language Models (LLMs) and RAG architectures can produce hallucinated results. Users should not rely on this software to make financial decisions.

## License

**© 2026 The University of Leeds and Alexander East**.

This project was submitted in accordance with the requirements for the degree of BSc Computer Science (Digital & Technology Solutions) at the University of Leeds. 