from __future__ import annotations

from graphrag.kg.manager import KnowledgeGraphManager


def inject_movie_dataset(kg_manager: KnowledgeGraphManager) -> None:
    kg_manager.clear()

    cypher_query = """
    MERGE (m1:Movie {title: 'The Matrix'})
    ON CREATE SET m1.released = 1999, m1.tagline = 'Welcome to the Real World', m1.budget_usd = 63000000, m1.box_office_usd = 467222728
    MERGE (m2:Movie {title: 'The Matrix Reloaded'})
    ON CREATE SET m2.released = 2003, m2.tagline = 'Everything that has a beginning has an end', m2.budget_usd = 127000000, m2.box_office_usd = 741763379
    MERGE (m3:Movie {title: 'The Matrix Revolutions'})
    ON CREATE SET m3.released = 2003, m3.tagline = 'Everything ends', m3.budget_usd = 110000000, m3.box_office_usd = 427343830
    MERGE (m4:Movie {title: 'The Matrix Resurrections'})
    ON CREATE SET m4.released = 2021, m4.tagline = 'Return to the Source', m4.budget_usd = 160000000, m4.box_office_usd = 157451968
    MERGE (m5:Movie {title: 'John Wick'})
    ON CREATE SET m5.released = 2014, m5.tagline = 'Dont set him off', m5.budget_usd = 20000000, m5.box_office_usd = 426047111
    MERGE (m6:Movie {title: 'Speed'})
    ON CREATE SET m6.released = 1994, m6.tagline = 'Pop quiz hotshot', m6.budget_usd = 70000000, m6.box_office_usd = 274912974
    MERGE (k:Person {name: 'Keanu Reeves', profession: 'Actor'}) ON CREATE SET k.born = 1964, k.birthplace = 'Beirut'
    MERGE (c:Person {name: 'Carrie-Anne Moss', profession: 'Actress'}) ON CREATE SET c.born = 1967, c.birthplace = 'Vancouver'
    MERGE (l:Person {name: 'Laurence Fishburne', profession: 'Actor'}) ON CREATE SET l.born = 1961, l.birthplace = 'Augusta'
    MERGE (h:Person {name: 'Hugo Weaving', profession: 'Actor'}) ON CREATE SET h.born = 1960, h.birthplace = 'Nigeria'
    MERGE (j:Person {name: 'Jada Pinkett Smith', profession: 'Actress'}) ON CREATE SET j.born = 1971
    MERGE (m:Person {name: 'Matt Dillon', profession: 'Actor'}) ON CREATE SET m.born = 1962
    MERGE (y:Person {name: 'Yahya Abdul-Mateen II', profession: 'Actor'}) ON CREATE SET y.born = 1986
    MERGE (lana:Person {name: 'Lana Wachowski', profession: 'Director'}) ON CREATE SET lana.born = 1965, lana.birthplace = 'Chicago'
    MERGE (lilly:Person {name: 'Lilly Wachowski', profession: 'Director'}) ON CREATE SET lilly.born = 1967, lilly.birthplace = 'Chicago'
    MERGE (dc:Person {name: 'David Leitch', profession: 'Director'}) ON CREATE SET dc.born = 1975
    MERGE (jp:Person {name: 'Jan de Bont', profession: 'Director'}) ON CREATE SET jp.born = 1943
    MERGE (ls:Person {name: 'Laszlo Kovacs', profession: 'Cinematographer'}) ON CREATE SET ls.born = 1933
    MERGE (jc:Person {name: 'Joel Silver', profession: 'Producer'}) ON CREATE SET jc.born = 1952
    MERGE (bm:Person {name: 'Basil Iwanyk', profession: 'Producer'}) ON CREATE SET bm.born = 1969
    MERGE (sci_fi:Genre {name: 'Science Fiction'})
    MERGE (action:Genre {name: 'Action'})
    MERGE (thriller:Genre {name: 'Thriller'})
    MERGE (cyberpunk:Genre {name: 'Cyberpunk'})
    MERGE (k)-[:ACTED_IN {roles: ['Neo'], screen_time_minutes: 145}]->(m1)
    MERGE (c)-[:ACTED_IN {roles: ['Trinity'], screen_time_minutes: 98}]->(m1)
    MERGE (l)-[:ACTED_IN {roles: ['Morpheus'], screen_time_minutes: 105}]->(m1)
    MERGE (h)-[:ACTED_IN {roles: ['Agent Smith'], screen_time_minutes: 85}]->(m1)
    MERGE (lana)-[:DIRECTED]->(m1)
    MERGE (lilly)-[:DIRECTED]->(m1)
    MERGE (jc)-[:PRODUCED]->(m1)
    MERGE (k)-[:ACTED_IN {roles: ['Neo'], screen_time_minutes: 160}]->(m2)
    MERGE (c)-[:ACTED_IN {roles: ['Trinity'], screen_time_minutes: 120}]->(m2)
    MERGE (l)-[:ACTED_IN {roles: ['Morpheus'], screen_time_minutes: 95}]->(m2)
    MERGE (h)-[:ACTED_IN {roles: ['Agent Smith'], screen_time_minutes: 110}]->(m2)
    MERGE (j)-[:ACTED_IN {roles: ['Niobe'], screen_time_minutes: 70}]->(m2)
    MERGE (lana)-[:DIRECTED]->(m2)
    MERGE (lilly)-[:DIRECTED]->(m2)
    MERGE (jc)-[:PRODUCED]->(m2)
    MERGE (k)-[:ACTED_IN {roles: ['Neo'], screen_time_minutes: 155}]->(m3)
    MERGE (c)-[:ACTED_IN {roles: ['Trinity'], screen_time_minutes: 115}]->(m3)
    MERGE (l)-[:ACTED_IN {roles: ['Morpheus'], screen_time_minutes: 70}]->(m3)
    MERGE (h)-[:ACTED_IN {roles: ['Agent Smith'], screen_time_minutes: 105}]->(m3)
    MERGE (j)-[:ACTED_IN {roles: ['Niobe'], screen_time_minutes: 90}]->(m3)
    MERGE (lana)-[:DIRECTED]->(m3)
    MERGE (lilly)-[:DIRECTED]->(m3)
    MERGE (jc)-[:PRODUCED]->(m3)
    MERGE (k)-[:ACTED_IN {roles: ['Thomas/Neo'], screen_time_minutes: 168}]->(m4)
    MERGE (c)-[:ACTED_IN {roles: ['Trinity'], screen_time_minutes: 125}]->(m4)
    MERGE (y)-[:ACTED_IN {roles: ['Morpheus'], screen_time_minutes: 95}]->(m4)
    MERGE (jc)-[:PRODUCED]->(m4)
    MERGE (k)-[:ACTED_IN {roles: ['John Wick'], screen_time_minutes: 188}]->(m5)
    MERGE (dc)-[:DIRECTED]->(m5)
    MERGE (bm)-[:PRODUCED]->(m5)
    MERGE (k)-[:ACTED_IN {roles: ['Jack Traven'], screen_time_minutes: 156}]->(m6)
    MERGE (m)-[:ACTED_IN {roles: ['Madman'], screen_time_minutes: 15}]->(m6)
    MERGE (jp)-[:DIRECTED]->(m6)
    MERGE (m1)-[:HAS_GENRE]->(sci_fi)
    MERGE (m1)-[:HAS_GENRE]->(action)
    MERGE (m1)-[:HAS_GENRE]->(cyberpunk)
    MERGE (m2)-[:HAS_GENRE]->(sci_fi)
    MERGE (m2)-[:HAS_GENRE]->(action)
    MERGE (m2)-[:HAS_GENRE]->(cyberpunk)
    MERGE (m3)-[:HAS_GENRE]->(sci_fi)
    MERGE (m3)-[:HAS_GENRE]->(action)
    MERGE (m4)-[:HAS_GENRE]->(sci_fi)
    MERGE (m4)-[:HAS_GENRE]->(action)
    MERGE (m5)-[:HAS_GENRE]->(action)
    MERGE (m5)-[:HAS_GENRE]->(thriller)
    MERGE (m6)-[:HAS_GENRE]->(action)
    MERGE (m6)-[:HAS_GENRE]->(thriller)
    MERGE (m1)-[:HAS_SEQUEL]->(m2)
    MERGE (m2)-[:HAS_SEQUEL]->(m3)
    MERGE (m3)-[:HAS_SEQUEL]->(m4)
    MERGE (lana)-[:COLLABORATED_WITH]->(lilly)
    MERGE (lilly)-[:COLLABORATED_WITH]->(lana)
    MERGE (k)-[:WORKED_WITH]->(c)
    MERGE (k)-[:WORKED_WITH]->(l)
    """
    kg_manager.run_query(cypher_query)
