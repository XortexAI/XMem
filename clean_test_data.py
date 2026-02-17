"""Clean up test data from Pinecone and Neo4j before re-testing."""

import os
from dotenv import load_dotenv
load_dotenv()

print("\n🧹 Cleaning test data...\n")

# Clean Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
namespace = os.getenv("PINECONE_NAMESPACE", "default")

# Delete all vectors for demo_user
# Pinecone serverless supports delete by metadata filter
try:
    index.delete(
        filter={"user_id": {"$eq": "demo_user"}},
        namespace=namespace,
    )
    print("  ✅ Pinecone: deleted demo_user vectors")
except Exception as e:
    print(f"  ⚠️  Pinecone delete failed: {e}")
    print("     Trying delete_all for namespace...")
    try:
        index.delete(delete_all=True, namespace=namespace)
        print("  ✅ Pinecone: cleared namespace")
    except Exception as e2:
        print(f"  ❌ Pinecone: {e2}")

# Clean Neo4j
from neo4j import GraphDatabase
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
with driver.session() as session:
    result = session.run("""
        MATCH (u:User {user_id: 'demo_user'})-[r:HAS_EVENT]->()
        DELETE r
        RETURN count(r) as deleted
    """)
    count = result.single()["deleted"]
    print(f"  ✅ Neo4j: deleted {count} event relationships for demo_user")

driver.close()
print("\n✨ Clean up complete!\n")
