"""Quick script to verify Pinecone and Neo4j connectivity."""

import os
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*70)
print("XMEM CONFIGURATION CHECK")
print("="*70 + "\n")

# Check environment variables
print("📋 Environment Variables:")
print(f"  ✓ GEMINI_API_KEY: {'set' if os.getenv('GEMINI_API_KEY') else '❌ MISSING'}")
print(f"  ✓ PINECONE_API_KEY: {'set' if os.getenv('PINECONE_API_KEY') else '❌ MISSING'}")
print(f"  ✓ PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME', '❌ MISSING')}")
print(f"  ✓ PINECONE_DIMENSION: {os.getenv('PINECONE_DIMENSION', '❌ MISSING')}")
print(f"  ✓ NEO4J_URI: {os.getenv('NEO4J_URI', '❌ MISSING')}")
print(f"  ✓ NEO4J_PASSWORD: {'set' if os.getenv('NEO4J_PASSWORD') else '❌ MISSING'}")
print()

# Test Pinecone connection
print("🔌 Testing Pinecone connection...")
try:
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    indexes = pc.list_indexes()
    
    index_name = os.getenv('PINECONE_INDEX_NAME')
    index_names = [idx.name for idx in indexes]
    
    print(f"  Available indexes: {index_names}")
    
    if index_name in index_names:
        print(f"  ✅ Index '{index_name}' exists")
        
        # Get index details
        index_info = pc.describe_index(index_name)
        print(f"  Dimension: {index_info.dimension}")
        print(f"  Metric: {index_info.metric}")
        print(f"  Status: {index_info.status.state}")
        
        # Check dimension matches
        expected_dim = int(os.getenv('PINECONE_DIMENSION', '768'))
        if index_info.dimension != expected_dim:
            print(f"  ⚠️  WARNING: Index dimension ({index_info.dimension}) doesn't match config ({expected_dim})")
    else:
        print(f"  ⚠️  Index '{index_name}' not found. Available: {index_names}")
        print(f"  You need to create it with dimension=768, metric=cosine")
        
except Exception as e:
    print(f"  ❌ Pinecone connection failed: {e}")

print()

# Test Neo4j connection
print("🔌 Testing Neo4j connection...")
try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
    driver.verify_connectivity()
    print("  ✅ Neo4j connected successfully")
    
    # Test a simple query
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        record = result.single()
        print(f"  Test query result: {record['test']}")
    
    driver.close()
    
except Exception as e:
    print(f"  ❌ Neo4j connection failed: {e}")

print()

# Test embedding model
print("🤖 Testing embedding model...")
try:
    from google import genai
    from google.genai import types
    
    # model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2') 
    model_name = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
    print(f"  Loading model: {model_name}")
    
    api_key_to_use = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key_to_use) if api_key_to_use else genai.Client()
    
    print(f"  ✅ Model loaded successfully")
    
    # Test embedding
    test_text = "Hello world"
    result = client.models.embed_content(
        model=model_name,
        contents=test_text,
        config=types.EmbedContentConfig(output_dimensionality=int(os.getenv('PINECONE_DIMENSION', '768')))
    )
    [embedding_obj] = result.embeddings
    dim = len(embedding_obj.values)
    
    print(f"  Dimension: {dim}")
    print(f"  Test embedding shape: ({dim},)")
    
except Exception as e:
    print(f"  ❌ Embedding model failed: {e}")

print("\n" + "="*70)
print("CONFIGURATION CHECK COMPLETE")
print("="*70 + "\n")
