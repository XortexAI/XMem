import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure the root of the project is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.pipelines.retrieval import RetrievalPipeline

async def interactive_session():
    print("=" * 80)
    print("XMEM INTERACTIVE RETRIEVAL - LONGMEMEVAL DATASET")
    print("=" * 80)
    
    # Initialize the retrieval pipeline
    print("Initializing components...")
    pipeline = RetrievalPipeline()
    print("✓ Retrieval Pipeline Initialized")
    print("-" * 80)
    
    # Base user ID for the LongMemEval oracle dataset
    base_user_id = "longmemeval_longmemeval_oracle"
    
    while True:
        try:
            print("\n" + "=" * 80)
            
            # Get the question number to determine which memory namespace to search in
            q_num_input = input("Enter Question Number (e.g., 1, 100) to search against [or 'quit' to exit]: ").strip()
            
            if q_num_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting interactive session.")
                break
                
            if not q_num_input.isdigit():
                print("Please enter a valid number.")
                continue
                
            user_id = f"{base_user_id}_q{q_num_input}"
            
            # Get the query from the user
            query = input(f"\n[Searching for user: {user_id}]\nEnter your query: ")
            
            if not query.strip():
                print("Query cannot be empty.")
                continue
                
            print(f"\n🔍 Searching memories...")
            
            # Run the retrieval pipeline
            result = await pipeline.run(
                query=query,
                user_id=user_id,
                top_k=10  # Use the default for LongMemEval
            )
            
            print("\n" + "-" * 60)
            print("FETCHED CONTEXT (WHAT THE LLM SEES):")
            print("-" * 60)
            
            total_facts = 0
            total_events = 0
            
            # Iterate through the retrieved sources to see what was retrieved
            if not result.sources:
                print("No context retrieved from any databases.")
            else:
                for idx, record in enumerate(result.sources, 1):
                    print(f"\n➤ SOURCE: {record.domain.upper()}")
                    print(f"  [{idx}] {record.content}")
                    
                    if record.domain == "summary":
                        total_facts += 1
                    elif record.domain == "temporal":
                        total_events += 1
                    elif record.domain == "profile":
                        total_facts += 1

            print(f"\n[Stats: {total_facts} profiles/summaries from Pinecone, {total_events} events from Neo4j]")
            
            print("\n" + "-" * 60)
            print("FINAL GENERATED ANSWER:")
            print("-" * 60)
            print(result.answer)
            
        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the interactive session
    asyncio.run(interactive_session())
