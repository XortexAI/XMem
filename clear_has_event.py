
from neo4j import GraphDatabase

uri = "neo4j+ssc://98eed7c7.databases.neo4j.io"
user = "98eed7c7"
password = "tBWqVv82mwrYUxasnzDZS9LpYC2dRdobHvwqW-MxAcI"

driver = GraphDatabase.driver(uri, auth=(user, password))

def remove_has_events():
    query = "MATCH ()-[r:HAS_EVENT]->() DELETE r"
    with driver.session() as session:
        result = session.run(query)
        summary = result.consume()
        print(f"Deleted {summary.counters.relationships_deleted} HAS_EVENT relationships.")

if __name__ == "__main__":
    remove_has_events()
    driver.close()
