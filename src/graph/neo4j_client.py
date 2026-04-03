"""
Neo4j client — connection management + CRUD operations for temporal events.

Provides:
- ``Neo4jClient``: manages the driver, exposes session and CRUD helpers.
- Graph operations: create/update/delete events, search by embedding
  (cosine similarity), initialise calendar date nodes.

Usage::

    from src.graph.neo4j_client import Neo4jClient

    client = Neo4jClient(uri, user, password)
    client.connect()
    client.initialize_date_nodes()
    client.create_user_node("user_123")
    client.create_event("user_123", "03-15", {"event_name": "Birthday", ...})
    results = client.search_events_by_embedding("user_123", query_embedding)
    client.close()
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from neo4j import GraphDatabase

from src.graph.schema import GraphSchema

logger = logging.getLogger("xmem.graph.neo4j")

F = TypeVar("F", bound=Callable)


# ---------------------------------------------------------------------------
# Retry decorator for transient connection errors
# ---------------------------------------------------------------------------

def _neo4j_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Retry on transient Neo4j connection failures with exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: "Neo4jClient", *args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as exc:
                    err = str(exc).lower()
                    retryable = any(
                        k in err
                        for k in ("ssl", "connection", "routing", "eof", "reset", "refused")
                    )
                    if not retryable:
                        raise
                    last_exc = exc
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Neo4j retry %d/%d for %s: %s (wait %.1fs)",
                        attempt + 1, max_retries, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    # Force reconnect
                    try:
                        self.close()
                        self.connect()
                    except Exception as conn_err:
                        logger.error("Reconnect failed: %s", conn_err)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Neo4jClient
# ---------------------------------------------------------------------------

class Neo4jClient:
    """Manages the Neo4j driver and provides CRUD helpers for temporal events."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None
        self._embedding_fn = embedding_fn

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self._uri)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def _session(self):
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialised. Call connect() first.")
        return self._driver.session()

    @property
    def driver(self):
        return self._driver

    # -- initialisation ----------------------------------------------------

    @_neo4j_retry()
    def initialize_date_nodes(self) -> None:
        """Ensure 366 Date nodes (01-01 … 12-31) exist."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        dates: List[str] = []
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%m-%d"))
            cur += timedelta(days=1)

        query = f"""
        UNWIND $dates AS date_str
        MERGE (d:{GraphSchema.LABEL_DATE} {{ {GraphSchema.PROP_DATE_VAL}: date_str }})
        """
        with self._session() as session:
            session.run(query, dates=dates)
        logger.info("Initialised %d date nodes.", len(dates))

    @_neo4j_retry()
    def create_user_node(self, user_id: str) -> None:
        query = f"""
        MERGE (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
        RETURN u
        """
        with self._session() as session:
            session.run(query, user_id=user_id)

    # -- CRUD: events ------------------------------------------------------

    @_neo4j_retry()
    def create_event(
        self,
        user_id: str,
        date_str: str,
        event_data: Dict[str, Any],
    ) -> None:
        """Create a HAS_EVENT relationship between User and Date node."""
        self.create_user_node(user_id)

        props = self._build_event_props(event_data)

        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
        MATCH (d:{GraphSchema.LABEL_DATE} {{ {GraphSchema.PROP_DATE_VAL}: $date_str }})
        CREATE (u)-[r:{GraphSchema.REL_HAS_EVENT}]->(d)
        SET r += $props
        RETURN r
        """
        with self._session() as session:
            session.run(query, user_id=user_id, date_str=date_str, props=props)
        logger.info("Created event for %s on %s", user_id, date_str)

    @_neo4j_retry()
    def update_event(
        self,
        user_id: str,
        date_str: str,
        event_data: Dict[str, Any],
    ) -> None:
        """Update an existing HAS_EVENT relationship."""
        props = self._build_event_props(event_data)

        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
              -[r:{GraphSchema.REL_HAS_EVENT}]->
              (d:{GraphSchema.LABEL_DATE} {{ {GraphSchema.PROP_DATE_VAL}: $date_str }})
        SET r += $props
        RETURN r
        """
        with self._session() as session:
            session.run(query, user_id=user_id, date_str=date_str, props=props)

    @_neo4j_retry()
    def delete_event(
        self,
        user_id: str,
        date_str: str,
        event_name: Optional[str] = None,
    ) -> int:
        """Delete HAS_EVENT relationship(s). Optionally filter by event_name."""
        if event_name:
            query = f"""
            MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
                  -[r:{GraphSchema.REL_HAS_EVENT} {{ {GraphSchema.PROP_EVENT_NAME}: $event_name }}]->
                  (d:{GraphSchema.LABEL_DATE} {{ {GraphSchema.PROP_DATE_VAL}: $date_str }})
            DELETE r
            RETURN count(r) as deleted
            """
            params = {"user_id": user_id, "date_str": date_str, "event_name": event_name}
        else:
            query = f"""
            MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
                  -[r:{GraphSchema.REL_HAS_EVENT}]->
                  (d:{GraphSchema.LABEL_DATE} {{ {GraphSchema.PROP_DATE_VAL}: $date_str }})
            DELETE r
            RETURN count(r) as deleted
            """
            params = {"user_id": user_id, "date_str": date_str}

        with self._session() as session:
            result = session.run(query, **params)
            record = result.single()
            return record["deleted"] if record else 0

    # -- Search: cosine similarity on embeddings stored in relationships ---

    @_neo4j_retry()
    def search_events_by_embedding(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 1,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Semantic search over event embeddings stored on HAS_EVENT relationships.

        Generates an embedding for *query_text*, then fetches all events for
        *user_id* that carry an embedding, computes cosine similarity, and
        returns the top-k results above the threshold.
        """
        if not self._embedding_fn:
            logger.warning("No embedding function — cannot search by embedding.")
            return []

        query_embedding = self._embedding_fn(query_text)

        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
              -[r:{GraphSchema.REL_HAS_EVENT}]->
              (d:{GraphSchema.LABEL_DATE})
        WHERE r.{GraphSchema.PROP_EMBEDDING} IS NOT NULL
        RETURN r.{GraphSchema.PROP_EVENT_NAME} as event_name,
               r.{GraphSchema.PROP_DESC} as desc,
               r.{GraphSchema.PROP_YEAR} as year,
               r.{GraphSchema.PROP_TIME} as time,
               r.{GraphSchema.PROP_DATE_EXPRESSION} as date_expression,
               r.{GraphSchema.PROP_EMBEDDING} as embedding,
               d.{GraphSchema.PROP_DATE_VAL} as date
        """

        results: List[Dict[str, Any]] = []
        with self._session() as session:
            records = session.run(query, user_id=user_id)
            for record in records:
                event_embedding = record["embedding"]
                if event_embedding:
                    q_vec = np.array(query_embedding, dtype=np.float32)
                    e_vec = np.array(event_embedding, dtype=np.float32)
                    similarity = float(np.dot(q_vec, e_vec))

                    if similarity >= similarity_threshold:
                        results.append({
                            "event_name": record["event_name"],
                            "desc": record["desc"],
                            "year": record["year"],
                            "time": record["time"],
                            "date": record["date"],
                            "date_expression": record["date_expression"],
                            "similarity_score": similarity,
                        })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    # -- Search: by event_name (exact match) — for Judge -------------------

    @_neo4j_retry()
    def search_events_by_name(
        self,
        event_name: str,
        user_id: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Find events by exact event_name match for a given user.

        Used by the Judge to check for duplicate temporal events.
        """
        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
              -[r:{GraphSchema.REL_HAS_EVENT}]->
              (d:{GraphSchema.LABEL_DATE})
        WHERE toLower(r.{GraphSchema.PROP_EVENT_NAME}) = toLower($event_name)
        RETURN r.{GraphSchema.PROP_EVENT_NAME} as event_name,
               r.{GraphSchema.PROP_DESC} as desc,
               r.{GraphSchema.PROP_YEAR} as year,
               r.{GraphSchema.PROP_TIME} as time,
               r.{GraphSchema.PROP_DATE_EXPRESSION} as date_expression,
               d.{GraphSchema.PROP_DATE_VAL} as date
        LIMIT $top_k
        """
        results: List[Dict[str, Any]] = []
        with self._session() as session:
            records = session.run(
                query, user_id=user_id, event_name=event_name, top_k=top_k,
            )
            for record in records:
                results.append({
                    "event_name": record["event_name"],
                    "desc": record["desc"],
                    "year": record["year"],
                    "time": record["time"],
                    "date": record["date"],
                    "date_expression": record["date_expression"],
                })
        return results

    # -- Utility -----------------------------------------------------------

    @_neo4j_retry()
    def get_all_events_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
              -[r:{GraphSchema.REL_HAS_EVENT}]->
              (d:{GraphSchema.LABEL_DATE})
        RETURN r.{GraphSchema.PROP_EVENT_NAME} as event_name,
               r.{GraphSchema.PROP_DESC} as desc,
               r.{GraphSchema.PROP_YEAR} as year,
               r.{GraphSchema.PROP_TIME} as time,
               r.{GraphSchema.PROP_DATE_EXPRESSION} as date_expression,
               d.{GraphSchema.PROP_DATE_VAL} as date
        """
        results: List[Dict[str, Any]] = []
        with self._session() as session:
            records = session.run(query, user_id=user_id)
            for record in records:
                results.append({
                    "event_name": record["event_name"],
                    "desc": record["desc"],
                    "year": record["year"],
                    "time": record["time"],
                    "date": record["date"],
                    "date_expression": record["date_expression"],
                })
        return results

    @_neo4j_retry()
    def clear_user_events(self, user_id: str) -> int:
        query = f"""
        MATCH (u:{GraphSchema.LABEL_USER} {{ {GraphSchema.PROP_USER_ID}: $user_id }})
              -[r:{GraphSchema.REL_HAS_EVENT}]->()
        DELETE r
        RETURN count(r) as deleted
        """
        with self._session() as session:
            result = session.run(query, user_id=user_id)
            record = result.single()
            return record["deleted"] if record else 0

    # -- internal helpers --------------------------------------------------

    def _build_event_props(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a clean property dict, optionally adding an embedding."""
        props: Dict[str, Any] = {
            GraphSchema.PROP_EVENT_NAME: event_data.get("event_name"),
            GraphSchema.PROP_YEAR: event_data.get("year"),
            GraphSchema.PROP_DESC: event_data.get("desc"),
            GraphSchema.PROP_TIME: event_data.get("time"),
            GraphSchema.PROP_DATE_EXPRESSION: event_data.get("date_expression"),
        }

        # Generate embedding for the event description
        if self._embedding_fn:
            desc = event_data.get("desc", "")
            event_name = event_data.get("event_name", "")
            searchable = f"{event_name}: {desc}" if desc else event_name
            if searchable:
                props[GraphSchema.PROP_EMBEDDING] = self._embedding_fn(searchable)

        # Remove None values — Neo4j doesn't like them in SET
        return {k: v for k, v in props.items() if v is not None}
