"""Local vector store implementations for XMem.

These backends implement the same BaseVectorStore contract as Pinecone so
local/dev deployments can switch storage with VECTOR_STORE_PROVIDER while
leaving the cloud path untouched.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings
from src.storage.base import BaseVectorStore, IndexStats, SearchResult
from src.storage.memory_lifecycle import (
    CONTENT_HASH_KEY,
    FORGET_REASON_KEY,
    FORGOTTEN_AT_KEY,
    IS_CURRENT_KEY,
    PARENT_MEMORY_ID_KEY,
    VERSION_KEY,
    build_lifecycle_metadata,
    compute_memory_hash,
    is_retrievable_memory,
    utc_now_iso,
)
from src.utils.exceptions import VectorStoreValidationError


def _metadata_matches(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise VectorStoreValidationError(
            f"Vector dimension mismatch: {len(a)} != {len(b)}",
            operation="search",
        )
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, (dot / (norm_a * norm_b) + 1.0) / 2.0))


_DEDUP_SCOPE_KEYS = ("user_id", "tenant_id", "org_id", "workspace_id", "project_id")


class SQLiteVectorStore(BaseVectorStore):
    """Small embedded vector store for single-user local testing.

    Embeddings are stored as JSON and ranked in Python. This is intentionally
    simple and portable; for larger local datasets use pgvector or Chroma.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
        create_if_not_exists: bool = True,
    ) -> None:
        self._path = Path(path or settings.sqlite_vector_path)
        self._namespace = namespace or settings.pinecone_namespace
        self._dimension = int(dimension or settings.pinecone_dimension)
        if create_if_not_exists:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS xmem_vectors (
                namespace TEXT NOT NULL,
                id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_xmem_vectors_namespace "
            "ON xmem_vectors(namespace)"
        )
        self._conn.commit()

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        self.validate_inputs(texts, embeddings, ids, metadata)
        is_valid, error = self.validate_embeddings(embeddings, self._dimension)
        if not is_valid:
            raise VectorStoreValidationError(error, operation="add")

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadata = metadata or [{} for _ in texts]
        output_ids: List[str] = []
        rows = []
        for text, embedding, vec_id, meta in zip(texts, embeddings, ids, metadata):
            lifecycle_meta = build_lifecycle_metadata(text, meta)
            existing_id = self._find_current_by_hash(
                lifecycle_meta[CONTENT_HASH_KEY],
                lifecycle_meta,
            )
            if existing_id:
                output_ids.append(existing_id)
                continue
            output_ids.append(vec_id)
            rows.append(
                (
                    self._namespace,
                    vec_id,
                    text,
                    json.dumps([float(v) for v in embedding]),
                    json.dumps(lifecycle_meta),
                )
            )
        if not rows:
            return output_ids
        self._conn.executemany(
            """
            INSERT INTO xmem_vectors(namespace, id, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(namespace, id) DO UPDATE SET
                content = excluded.content,
                embedding = excluded.embedding,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        self._conn.commit()
        return output_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if len(query_embedding) != self._dimension:
            raise VectorStoreValidationError(
                f"Query embedding dimension {len(query_embedding)} "
                f"doesn't match index dimension {self._dimension}",
                operation="search",
            )
        rows = self._conn.execute(
            "SELECT id, content, embedding, metadata FROM xmem_vectors WHERE namespace = ?",
            (self._namespace,),
        ).fetchall()
        results: List[SearchResult] = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            if not is_retrievable_memory(meta) or not _metadata_matches(meta, filters):
                continue
            embedding = json.loads(row["embedding"])
            results.append(
                SearchResult(
                    id=row["id"],
                    content=row["content"],
                    score=_cosine_similarity(query_embedding, embedding),
                    metadata=meta,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT content, embedding, metadata FROM xmem_vectors "
            "WHERE namespace = ? AND id = ?",
            (self._namespace, id),
        ).fetchone()
        if not row:
            return False
        current_meta = json.loads(row["metadata"] or "{}")
        current_meta.update(metadata or {})
        new_text = text if text is not None else row["content"]
        current_meta[CONTENT_HASH_KEY] = compute_memory_hash(new_text)
        new_embedding = embedding if embedding is not None else json.loads(row["embedding"])
        if len(new_embedding) != self._dimension:
            raise VectorStoreValidationError(
                f"Embedding dimension {len(new_embedding)} doesn't match {self._dimension}",
                operation="update",
            )
        self._conn.execute(
            """
            UPDATE xmem_vectors
            SET content = ?, embedding = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
            WHERE namespace = ? AND id = ?
            """,
            (
                new_text,
                json.dumps([float(v) for v in new_embedding]),
                json.dumps(current_meta),
                self._namespace,
                id,
            ),
        )
        self._conn.commit()
        return True

    def add_version(
        self,
        parent_id: str,
        text: str,
        embedding: List[float],
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a new current memory version and keep the parent as history."""

        parent = self.get([parent_id])
        if not parent:
            return None
        if not is_retrievable_memory(parent[0]["metadata"] or {}):
            return None
        if len(embedding) != self._dimension:
            raise VectorStoreValidationError(
                f"Embedding dimension {len(embedding)} doesn't match {self._dimension}",
                operation="add_version",
            )

        parent_meta = dict(parent[0]["metadata"] or {})
        root_parent_id = parent_meta.get(PARENT_MEMORY_ID_KEY) or parent_id
        next_version = int(parent_meta.get(VERSION_KEY) or 1) + 1
        new_id = id or str(uuid.uuid4())
        new_meta = build_lifecycle_metadata(
            text,
            metadata,
            parent_memory_id=root_parent_id,
            version=next_version,
            is_current=True,
        )
        existing_id = self._find_current_by_hash(new_meta[CONTENT_HASH_KEY], new_meta)
        if existing_id:
            if existing_id == parent_id:
                return existing_id
            parent_meta[IS_CURRENT_KEY] = False
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE xmem_vectors
                    SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE namespace = ? AND id = ?
                    """,
                    (json.dumps(parent_meta), self._namespace, parent_id),
                )
            return existing_id

        parent_meta[IS_CURRENT_KEY] = False
        with self._conn:
            self._conn.execute(
                """
                UPDATE xmem_vectors
                SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE namespace = ? AND id = ?
                """,
                (json.dumps(parent_meta), self._namespace, parent_id),
            )
            self._conn.execute(
                """
                INSERT INTO xmem_vectors(namespace, id, content, embedding, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self._namespace,
                    new_id,
                    text,
                    json.dumps([float(v) for v in embedding]),
                    json.dumps(new_meta),
                ),
            )
        return new_id

    def forget(
        self,
        ids: List[str],
        reason: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Soft-forget memories by default, preserving audit history."""

        if hard_delete:
            return self.delete(ids)
        if not ids:
            return True

        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, metadata FROM xmem_vectors "
            f"WHERE namespace = ? AND id IN ({placeholders})",
            [self._namespace, *ids],
        ).fetchall()

        now = utc_now_iso()
        updates = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            meta[IS_CURRENT_KEY] = False
            meta[FORGOTTEN_AT_KEY] = now
            meta[FORGET_REASON_KEY] = reason
            updates.append((json.dumps(meta), self._namespace, row["id"]))

        if updates:
            self._conn.executemany(
                "UPDATE xmem_vectors SET metadata = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE namespace = ? AND id = ?",
                updates,
            )
            self._conn.commit()
        return True

    def delete(self, ids: List[str]) -> bool:
        if not ids:
            return True
        self._conn.executemany(
            "DELETE FROM xmem_vectors WHERE namespace = ? AND id = ?",
            [(self._namespace, vec_id) for vec_id in ids],
        )
        self._conn.commit()
        return True

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, content, embedding, metadata FROM xmem_vectors "
            f"WHERE namespace = ? AND id IN ({placeholders})",
            [self._namespace, *ids],
        ).fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "embedding": json.loads(row["embedding"]),
            }
            for row in rows
        ]

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 10,
    ) -> List[SearchResult]:
        rows = self._conn.execute(
            "SELECT id, content, metadata FROM xmem_vectors WHERE namespace = ?",
            (self._namespace,),
        ).fetchall()
        results: List[SearchResult] = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            if is_retrievable_memory(meta) and _metadata_matches(meta, filters):
                results.append(SearchResult(id=row["id"], content=row["content"], score=1.0, metadata=meta))
        return results[:top_k]

    def _find_current_by_hash(
        self,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        clauses = [
            "namespace = ?",
            f"json_extract(metadata, '$.{CONTENT_HASH_KEY}') = ?",
            f"json_extract(metadata, '$.{IS_CURRENT_KEY}') = 1",
            f"json_extract(metadata, '$.{FORGOTTEN_AT_KEY}') IS NULL",
        ]
        params: List[Any] = [self._namespace, content_hash]
        scope = {
            key: (metadata or {}).get(key)
            for key in _DEDUP_SCOPE_KEYS
            if (metadata or {}).get(key) is not None
        }
        if scope:
            for key, value in scope.items():
                clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
        else:
            for key in _DEDUP_SCOPE_KEYS:
                clauses.append(f"json_type(metadata, '$.{key}') IS NULL")

        row = self._conn.execute(
            f"SELECT id FROM xmem_vectors WHERE {' AND '.join(clauses)} LIMIT 1",
            params,
        ).fetchone()
        return row["id"] if row else None

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        from src.pipelines.ingest import embed_text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(list(embed_text(query_text)), top_k=top_k, filters=filters),
        )

    def health_check(self) -> bool:
        try:
            self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def get_stats(self) -> IndexStats:
        rows = self._conn.execute(
            "SELECT namespace, count(*) AS count FROM xmem_vectors GROUP BY namespace"
        ).fetchall()
        namespaces = {row["namespace"]: int(row["count"]) for row in rows}
        return IndexStats(
            total_vector_count=sum(namespaces.values()),
            dimension=self._dimension,
            namespaces=namespaces,
        )


class ChromaVectorStore(BaseVectorStore):
    """Persistent Chroma backend for easy local development."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
        create_if_not_exists: bool = True,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install Chroma with: pip install chromadb") from exc

        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._namespace = (namespace or settings.pinecone_namespace).replace(":", "_")
        self._dimension = int(dimension or settings.pinecone_dimension)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=create_if_not_exists)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(name=self._namespace)

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        self.validate_inputs(texts, embeddings, ids, metadata)
        is_valid, error = self.validate_embeddings(embeddings, self._dimension)
        if not is_valid:
            raise VectorStoreValidationError(error, operation="add")
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadata = metadata or [{} for _ in texts]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
        )
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchResult(
                id=vec_id,
                content=doc or "",
                score=max(0.0, min(1.0, 1.0 - float(distance))),
                metadata=meta or {},
            )
            for vec_id, doc, meta, distance in zip(ids, docs, metadatas, distances)
        ]

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        existing = self.get([id])
        if not existing:
            return False
        doc = existing[0]
        merged_meta = dict(doc.get("metadata") or {})
        merged_meta.update(metadata or {})
        self._collection.update(
            ids=[id],
            documents=[text if text is not None else doc["content"]],
            embeddings=[embedding] if embedding is not None else None,
            metadatas=[merged_meta],
        )
        return True

    def delete(self, ids: List[str]) -> bool:
        if ids:
            self._collection.delete(ids=ids)
        return True

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        result = self._collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        return [
            {
                "id": vec_id,
                "content": doc or "",
                "metadata": meta or {},
                "embedding": emb,
            }
            for vec_id, doc, meta, emb in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
                result.get("embeddings", []),
            )
        ]

    def search_by_metadata(self, filters: Dict[str, Any], top_k: int = 10) -> List[SearchResult]:
        result = self._collection.get(where=filters or None, limit=top_k, include=["documents", "metadatas"])
        return [
            SearchResult(id=vec_id, content=doc or "", score=1.0, metadata=meta or {})
            for vec_id, doc, meta in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ]

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        from src.pipelines.ingest import embed_text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(list(embed_text(query_text)), top_k=top_k, filters=filters),
        )

    def health_check(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    def get_stats(self) -> IndexStats:
        count = int(self._collection.count())
        return IndexStats(
            total_vector_count=count,
            dimension=self._dimension,
            namespaces={self._namespace: count},
        )


class PGVectorStore(BaseVectorStore):
    """Postgres + pgvector backend for production-like local installs."""

    def __init__(
        self,
        url: Optional[str] = None,
        table: Optional[str] = None,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
        create_if_not_exists: bool = True,
    ) -> None:
        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise ImportError(
                "Install pgvector dependencies with: pip install psycopg[binary] pgvector"
            ) from exc

        self._psycopg = psycopg
        self._jsonb = Jsonb
        self._url = url or settings.pgvector_url
        self._table = table or settings.pgvector_table
        self._namespace = namespace or settings.pinecone_namespace
        self._dimension = int(dimension or settings.pinecone_dimension)
        self._conn = psycopg.connect(self._url)
        self._conn.autocommit = True
        if create_if_not_exists:
            self._setup()

    def _setup(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    namespace TEXT NOT NULL,
                    id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self._dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY(namespace, id)
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_namespace_idx "
                f"ON {self._table}(namespace)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_metadata_idx "
                f"ON {self._table} USING GIN(metadata)"
            )

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(str(float(v)) for v in vector) + "]"

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        self.validate_inputs(texts, embeddings, ids, metadata)
        is_valid, error = self.validate_embeddings(embeddings, self._dimension)
        if not is_valid:
            raise VectorStoreValidationError(error, operation="add")
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadata = metadata or [{} for _ in texts]
        rows = [
            (self._namespace, vec_id, text, self._vector_literal(embedding), self._jsonb(meta or {}))
            for text, embedding, vec_id, meta in zip(texts, embeddings, ids, metadata)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._table}(namespace, id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s)
                ON CONFLICT(namespace, id) DO UPDATE SET
                    content = excluded.content,
                    embedding = excluded.embedding,
                    metadata = excluded.metadata,
                    updated_at = now()
                """,
                rows,
            )
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        params: List[Any] = [self._vector_literal(query_embedding), self._namespace]
        filter_sql = ""
        if filters:
            filter_sql = " AND metadata @> %s"
            params.append(self._jsonb(filters))
        params.append(int(top_k))
        with self._conn.cursor(row_factory=self._psycopg.rows.dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata,
                       GREATEST(0.0, LEAST(1.0, 1.0 - (embedding <=> %s::vector))) AS score
                FROM {self._table}
                WHERE namespace = %s{filter_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [params[0], *params[1:-1], params[0], params[-1]],
            )
            rows = cur.fetchall()
        return [
            SearchResult(id=row["id"], content=row["content"], score=float(row["score"]), metadata=dict(row["metadata"] or {}))
            for row in rows
        ]

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        docs = self.get([id])
        if not docs:
            return False
        current = docs[0]
        merged_meta = dict(current.get("metadata") or {})
        merged_meta.update(metadata or {})
        new_embedding = embedding if embedding is not None else current["embedding"]
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {self._table}
                SET content = %s, embedding = %s::vector, metadata = %s, updated_at = now()
                WHERE namespace = %s AND id = %s
                """,
                (
                    text if text is not None else current["content"],
                    self._vector_literal(new_embedding),
                    self._jsonb(merged_meta),
                    self._namespace,
                    id,
                ),
            )
            return cur.rowcount > 0

    def delete(self, ids: List[str]) -> bool:
        if not ids:
            return True
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE namespace = %s AND id = ANY(%s)",
                (self._namespace, ids),
            )
        return True

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        with self._conn.cursor(row_factory=self._psycopg.rows.dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata, embedding::text AS embedding
                FROM {self._table}
                WHERE namespace = %s AND id = ANY(%s)
                """,
                (self._namespace, ids),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": dict(row["metadata"] or {}),
                "embedding": json.loads(row["embedding"].replace("[", "[").replace("]", "]")),
            }
            for row in rows
        ]

    def search_by_metadata(self, filters: Dict[str, Any], top_k: int = 10) -> List[SearchResult]:
        with self._conn.cursor(row_factory=self._psycopg.rows.dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata
                FROM {self._table}
                WHERE namespace = %s AND metadata @> %s
                LIMIT %s
                """,
                (self._namespace, self._jsonb(filters or {}), int(top_k)),
            )
            rows = cur.fetchall()
        return [
            SearchResult(id=row["id"], content=row["content"], score=1.0, metadata=dict(row["metadata"] or {}))
            for row in rows
        ]

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        from src.pipelines.ingest import embed_text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(list(embed_text(query_text)), top_k=top_k, filters=filters),
        )

    def health_check(self) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def get_stats(self) -> IndexStats:
        with self._conn.cursor(row_factory=self._psycopg.rows.dict_row) as cur:
            cur.execute(
                f"SELECT namespace, count(*) AS count FROM {self._table} GROUP BY namespace"
            )
            rows = cur.fetchall()
        namespaces = {row["namespace"]: int(row["count"]) for row in rows}
        return IndexStats(
            total_vector_count=sum(namespaces.values()),
            dimension=self._dimension,
            namespaces=namespaces,
        )
