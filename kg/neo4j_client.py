from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase, NotificationDisabledCategory

class Neo4jClient:
    def __init__(self):
        from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_disabled_categories={
                NotificationDisabledCategory.UNRECOGNIZED,
            },
        )

    def query(self, q, params=None):
        with self.driver.session() as session:
            res = session.run(q, params or {})
            return [r.data() for r in res]