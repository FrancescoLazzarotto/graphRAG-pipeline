from __future__ import annotations

from graphrag.kg.manager import KnowledgeGraphManager


def inject_movie_dataset(kg_manager: KnowledgeGraphManager) -> None:
    cypher_query = """
    MERGE (m:Movie {title: 'The Matrix'})
    ON CREATE SET m.released = 1999, m.tagline = 'Welcome to the Real World'
    MERGE (k:Person {name: 'Keanu Reeves'}) ON CREATE SET k.born = 1964
    MERGE (c:Person {name: 'Carrie-Anne Moss'}) ON CREATE SET c.born = 1967
    MERGE (l:Person {name: 'Laurence Fishburne'}) ON CREATE SET l.born = 1961
    MERGE (h:Person {name: 'Hugo Weaving'})
    MERGE (lana:Person {name: 'Lana Wachowski'}) ON CREATE SET lana.born = 1965
    MERGE (lilly:Person {name: 'Lilly Wachowski'}) ON CREATE SET lilly.born = 1967

    MERGE (k)-[:ACTED_IN {roles: ['Neo']}]->(m)
    MERGE (c)-[:ACTED_IN {roles: ['Trinity']}]->(m)
    MERGE (l)-[:ACTED_IN {roles: ['Morpheus']}]->(m)
    MERGE (h)-[:ACTED_IN {roles: ['Agent Smith']}]->(m)
    MERGE (lana)-[:DIRECTED]->(m)
    MERGE (lilly)-[:DIRECTED]->(m)
    """
    kg_manager.run_query(cypher_query)
