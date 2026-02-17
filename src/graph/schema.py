"""
Neo4j Graph Schema — labels, relationship types, and property names.

Centralises all magic strings used in Cypher queries so a typo in one
place won't silently break queries elsewhere.

Node layout
-----------
(:User {user_id})
(:Date {date})

Relationship layout
-------------------
(:User)-[:HAS_EVENT {event_name, year, desc, time, date_expression, embedding}]->(:Date)
"""


class GraphSchema:
    # ── Node labels ──────────────────────────────────────────────────
    LABEL_USER = "User"
    LABEL_DATE = "Date"

    # ── Relationship types ───────────────────────────────────────────
    REL_HAS_EVENT = "HAS_EVENT"

    # ── Property names ───────────────────────────────────────────────
    PROP_USER_ID = "user_id"
    PROP_DATE_VAL = "date"

    PROP_EVENT_NAME = "event_name"
    PROP_YEAR = "year"
    PROP_DESC = "desc"
    PROP_TIME = "time"
    PROP_DATE_EXPRESSION = "date_expression"
    PROP_EMBEDDING = "embedding"


def setup_constraints(driver) -> None:
    """Create uniqueness constraints for User and Date nodes.

    Safe to call repeatedly — uses ``IF NOT EXISTS``.
    """
    queries = [
        (
            f"CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
            f"FOR (u:{GraphSchema.LABEL_USER}) "
            f"REQUIRE u.{GraphSchema.PROP_USER_ID} IS UNIQUE"
        ),
        (
            f"CREATE CONSTRAINT date_val_unique IF NOT EXISTS "
            f"FOR (d:{GraphSchema.LABEL_DATE}) "
            f"REQUIRE d.{GraphSchema.PROP_DATE_VAL} IS UNIQUE"
        ),
    ]

    with driver.session() as session:
        for q in queries:
            session.run(q)
