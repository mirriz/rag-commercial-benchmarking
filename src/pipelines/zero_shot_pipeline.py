import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

# Settings
MODEL_NAME = "gpt-5-mini"


def initialise_zero_shot_system():

    print("\n--- Initialising Zero-Shot System ---")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return None

    try:
        # Initialise LLM
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.0,
            max_retries=2,
            request_timeout=60
        )
        
        # Create Prompt Template
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant that provides accurate and concise answers based on your knowledge."
            ),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        # Create Chain
        chain = prompt_template | llm
        
        print(f"Zero-Shot System ({MODEL_NAME}) initialised.")
        return chain

    except Exception as e:
        print(f"Error initialising Zero-Shot system: {e}")
        return None


def run_zero_shot_query(chain, question):
    """
    Executes a Zero-Shot query.
    source_nodes will always be empty [] as there is no retrieval.
    """
    if chain is None:
        return "System failed to initialise.", []

    print(f"\n> Querying (Zero-Shot): {question}")
    
    try:
        # Execute the chain
        response = chain.invoke({"question": question})
        answer = response.content
        
        # Return empty list for sources to match RAG signature
        return answer, []
        
    except Exception as e:
        print(f"Error during query: {e}")
        return "Error generating response", []


if __name__ == "__main__":
    # Initialize
    zero_shot_engine = initialise_zero_shot_system()

    if zero_shot_engine:
        # Test Question
        test_question = "Delta in CBOE Data & Access Solutions rev from 2021-23."
        
        # Run Query
        answer, sources = run_zero_shot_query(zero_shot_engine, test_question)

        # Print Output
        print("\n" + "="*70)
        print(f"Final Answer (Zero-Shot):")
        print(f"{answer.strip()}")
        print("="*70)
        
        # Verify Sources (Should be empty)
        if sources:
            print(f"| Retrieved {len(sources)} Source Chunks (Unexpected)")
        else:
            print("No source nodes for Zero-Shot")
        print("="*70)