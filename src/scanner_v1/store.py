"""
Unified store for scanner_v1 — one Neo4j client that owns the graph,
the raw code, the embeddings, and the scan state.

Replaces three things from v0:
  src/scanner/code_store.py        (Mongo: raw code + scan state)
  src/storage/pinecone.py          (Pinecone: vector search)
  src/graph/code_graph_client.py   (Neo4j: graph queries)

Why the merge works:
  Neo4j 5.11+ has native vector indexes on node properties.  A Symbol
  node can carry `raw_code`, `summary`, `summary_embedding`, and
  `code_embedding` all at once.  One MERGE writes everything
  transactionally — no partial states between stores, no cleanup job.

Identity model — composite keys are non-negotiable:
  Repository : (org_id, repo)
  Directory  : (org_id, repo, dir_path)
  File       : (org_id, repo, file_path)
  Symbol     : (org_id, repo, file_path, qualified_name)
  Scan       : (org_id, repo, scan_id)

  Without the (org_id, repo) prefix, two repos with src/utils.py collapse
  into the same node — the v0 bug we are explicitly fixing.

Labels all carry the "V1" suffix so v0 and v1 can share one Neo4j
instance during migration.  Nothing in this file hard-codes Cypher
strings for labels/rels — everything comes from schemas.py.

This file is the ONLY place in scanner_v1 that knows about Cypher.
Everything else (indexer, enricher, runner) talks to methods here.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from neo4j import GraphDatabase

from src.scanner_v1 import schemas as S
from src.scanner_v1.embedder import SymbolEmbedding

logger = logging.getLogger("xmem.scanner_v1.store")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SYMBOL_UPSERT_BATCH = 100      # UNWIND batch for symbol writes
FILE_UPSERT_BATCH   = 50
CALL_EDGE_BATCH     = 500
IMPORT_EDGE_BATCH   = 500


# ---------------------------------------------------------------------------
# CodeStoreV1
# ---------------------------------------------------------------------------

class CodeStoreV1:
    """
    Single-database store backed by Neo4j.

    Responsibilities:
      - Connection lifecycle (connect / close / session context).
      - One-time schema setup: constraints + vector indexes + fulltext.
      - Upserts for Repository / Directory / File / Symbol / Scan nodes.
      - Edge creation: CONTAINS_* / DEFINES / CALLS / IMPORTS.
      - Cascading deletes.
      - Incremental-scan helpers: get_file_hash, get_file_symbol_hashes.
      - Scan-state lifecycle: start_scan / complete_scan / fail_scan /
        get_last_commit_sha.
      - Read paths used later by retrieval_v1:
            get_file_content, get_symbol_code, vector_search_symbols,
            vector_search_files, fulltext_search_symbols.
    """

    # =================================================================
    # LIFECYCLE
    # =================================================================

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: Optional[str] = None,
        embedding_dimension: int = 384,
    ) -> None:
        # embedding_dimension is required because Neo4j vector indexes
        # are typed by dimension at creation time. Keep in sync with
        # the embedder model.
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.embedding_dimension = int(embedding_dimension)
        self.driver = None

    def connect(self) -> None:
        """Open the Neo4j driver. Idempotent."""
        if self.driver is not None:
            return
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password),
            max_connection_lifetime=200, # Kill connections before Aura load balancers drop them
            keep_alive=True,             # Enable TCP keep-alive
        )

    def close(self) -> None:
        """Close the driver. Idempotent."""
        if self.driver is None:
            return
        self.driver.close()
        self.driver = None

    @contextmanager
    def _session(self):
        """Yield a Neo4j session bound to self.database.
        Opens the driver lazily if the caller never explicitly connect()ed."""
        if self.driver is None:
            self.connect()
        with self.driver.session(database=self.database) as session:
            yield session

    # =================================================================
    # ONE-TIME SCHEMA SETUP
    # =================================================================
    # Called once per Neo4j instance. Safe to call repeatedly —
    # CREATE ... IF NOT EXISTS is idempotent.
    #
    # Neo4j requires one statement per session.run() call, so each
    # constraint / index goes out individually.
    # -----------------------------------------------------------------

    def setup_schema(self) -> None:
        """Create constraints + vector indexes + fulltext indexes."""
        self._create_constraints()
        self._create_vector_indexes()
        self._create_fulltext_indexes()

    def _create_constraints(self) -> None:
        """Composite uniqueness constraints for every node type."""
        statements = [
            f"""
            CREATE CONSTRAINT {S.IDX_REPO_UNIQUE} IF NOT EXISTS
            FOR (r:{S.LABEL_REPOSITORY})
            REQUIRE (r.org_id, r.repo) IS UNIQUE
            """,
            f"""
            CREATE CONSTRAINT {S.IDX_FILE_UNIQUE} IF NOT EXISTS
            FOR (f:{S.LABEL_FILE})
            REQUIRE (f.org_id, f.repo, f.file_path) IS UNIQUE
            """,
            f"""
            CREATE CONSTRAINT {S.IDX_SYMBOL_UNIQUE} IF NOT EXISTS
            FOR (s:{S.LABEL_SYMBOL})
            REQUIRE (s.org_id, s.repo, s.file_path, s.qualified_name) IS UNIQUE
            """,
            f"""
            CREATE CONSTRAINT dir_v1_unique_idx IF NOT EXISTS
            FOR (d:{S.LABEL_DIRECTORY})
            REQUIRE (d.org_id, d.repo, d.dir_path) IS UNIQUE
            """,
            f"""
            CREATE CONSTRAINT scan_v1_unique_idx IF NOT EXISTS
            FOR (sc:{S.LABEL_SCAN})
            REQUIRE (sc.org_id, sc.repo, sc.scan_id) IS UNIQUE
            """,
        ]
        with self._session() as session:
            for stmt in statements:
                session.run(stmt)

    def _create_vector_indexes(self) -> None:
        """Build every lane declared in schemas.ALL_VECTOR_INDEXES.
        Dimensions come from self.embedding_dimension so swapping the
        model only touches the constructor."""
        dim = self.embedding_dimension
        with self._session() as session:
            for spec in S.ALL_VECTOR_INDEXES:
                cypher = f"""
                CREATE VECTOR INDEX {spec.name} IF NOT EXISTS
                FOR (n:{spec.label}) ON (n.{spec.property})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dim},
                        `vector.similarity_function`: '{spec.similarity}'
                    }}
                }}
                """
                session.run(cypher)

    def _create_fulltext_indexes(self) -> None:
        """Build fulltext indexes for Symbol + File.

        `whitespace` analyzer (not `standard`) keeps identifiers intact.
        The default English analyzer stems `getUserById` → `getuserbyid`
        and drops stop-words like `is`/`as`/`do` — real function names.
        """
        analyzer = S.FULLTEXT_ANALYZER_CODE
        symbol_ft = f"""
        CREATE FULLTEXT INDEX {S.IDX_SYMBOL_FULLTEXT} IF NOT EXISTS
        FOR (s:{S.LABEL_SYMBOL})
        ON EACH [s.qualified_name, s.signature, s.raw_code, s.summary]
        OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: '{analyzer}' }} }}
        """
        file_ft = f"""
        CREATE FULLTEXT INDEX {S.IDX_FILE_FULLTEXT} IF NOT EXISTS
        FOR (f:{S.LABEL_FILE})
        ON EACH [f.file_path, f.summary, f.raw_content]
        OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: '{analyzer}' }} }}
        """
        with self._session() as session:
            session.run(symbol_ft)
            session.run(file_ft)

    # =================================================================
    # REPOSITORY / DIRECTORY
    # =================================================================

    def upsert_repository(
        self, org_id: str, repo: str, branch: str = "main",
    ) -> None:
        """MERGE a Repository node keyed by (org_id, repo)."""
        cypher = f"""
        MERGE (r:{S.LABEL_REPOSITORY} {{org_id: $org_id, repo: $repo}})
        SET r.branch = $branch,
            r.indexed_at = datetime(),
            r.scanner_version = $scanner_version
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, branch=branch,
                scanner_version=S.SCANNER_VERSION,
            )

    def upsert_directory(
        self,
        org_id: str,
        repo: str,
        dir_path: str,
        languages: Optional[List[str]] = None,
        file_count: int = 0,
        total_symbols: int = 0,
        summary: str = "",
    ) -> None:
        """MERGE a Directory node and link it under the Repository.

        v1 keeps directories structural only — no embeddings — so the
        parent link goes straight to Repository (flat model) instead of
        reconstructing a full Dir→Dir tree. The indexer can layer that
        on later if navigation needs it.
        """
        cypher = f"""
        MERGE (r:{S.LABEL_REPOSITORY} {{org_id: $org_id, repo: $repo}})
        MERGE (d:{S.LABEL_DIRECTORY} {{org_id: $org_id, repo: $repo, dir_path: $dir_path}})
        SET d.languages = $languages,
            d.file_count = $file_count,
            d.symbol_count = $total_symbols,
            d.summary = $summary,
            d.indexed_at = datetime(),
            d.scanner_version = $scanner_version
        MERGE (r)-[:{S.REL_CONTAINS_DIR}]->(d)
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, dir_path=dir_path,
                languages=list(languages or []),
                file_count=int(file_count),
                total_symbols=int(total_symbols),
                summary=summary,
                scanner_version=S.SCANNER_VERSION,
            )

    # =================================================================
    # FILE
    # =================================================================

    def upsert_file(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        language: str,
        raw_content: str,
        summary: str,
        summary_source: str,
        summary_embedding: List[float],
        total_lines: int,
        symbol_count: int,
        symbol_names: List[str],
        content_hash: str,
        commit_sha: str,
    ) -> None:
        """Single-file upsert. For full scans prefer upsert_files_batch."""
        row = {
            "org_id": org_id,
            "repo": repo,
            "file_path": file_path,
            "language": language,
            "raw_content": raw_content,
            "summary": summary,
            "summary_source": summary_source,
            "summary_embedding": summary_embedding,
            "total_lines": int(total_lines),
            "symbol_count": int(symbol_count),
            "symbol_names": list(symbol_names or []),
            "content_hash": content_hash,
            "commit_sha": commit_sha,
        }
        self.upsert_files_batch([row])

    def upsert_files_batch(self, rows: Sequence[Dict[str, Any]]) -> None:
        """UNWIND-based batch upsert for files. The main full-scan path.

        Each row must carry every key upsert_file's signature names.
        Missing keys here would leave File nodes with NULL properties —
        the indexer is responsible for building complete rows.
        """
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MERGE (r:{S.LABEL_REPOSITORY} {{org_id: row.org_id, repo: row.repo}})
        MERGE (f:{S.LABEL_FILE} {{org_id: row.org_id, repo: row.repo, file_path: row.file_path}})
        SET f.language = row.language,
            f.raw_content = row.raw_content,
            f.summary = row.summary,
            f.summary_source = row.summary_source,
            f.summary_embedding = row.summary_embedding,
            f.total_lines = row.total_lines,
            f.symbol_count = row.symbol_count,
            f.symbol_names = row.symbol_names,
            f.content_hash = row.content_hash,
            f.commit_sha = row.commit_sha,
            f.indexed_at = datetime(),
            f.scanner_version = $scanner_version
        MERGE (r)-[:{S.REL_CONTAINS_FILE}]->(f)
        """
        with self._session() as session:
            session.run(
                cypher, rows=list(rows),
                scanner_version=S.SCANNER_VERSION,
            )

    def get_file_hash(
        self, org_id: str, repo: str, file_path: str,
    ) -> Optional[str]:
        """Return the stored content_hash for a file, or None."""
        cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{org_id: $org_id, repo: $repo, file_path: $file_path}})
        RETURN f.content_hash AS hash
        """
        with self._session() as session:
            record = session.run(
                cypher, org_id=org_id, repo=repo, file_path=file_path,
            ).single()
            return record["hash"] if record else None

    def get_file_content(
        self, org_id: str, repo: str, file_path: str,
    ) -> Optional[str]:
        """Return raw file text stored on the File node."""
        cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{org_id: $org_id, repo: $repo, file_path: $file_path}})
        RETURN f.raw_content AS content
        """
        with self._session() as session:
            record = session.run(
                cypher, org_id=org_id, repo=repo, file_path=file_path,
            ).single()
            return record["content"] if record else None

    # =================================================================
    # SYMBOL
    # =================================================================

    def upsert_symbol(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        symbol: Dict[str, Any],
        embedding: SymbolEmbedding,
        commit_sha: str,
    ) -> None:
        """Single-symbol upsert. Forwards to the batch path."""
        row = build_symbol_row(org_id, repo, file_path, symbol, embedding, commit_sha)
        self.upsert_symbols_batch([row])

    def upsert_symbols_batch(self, rows: Sequence[Dict[str, Any]]) -> None:
        """UNWIND-based batch upsert for symbols — the scan hot path.

        Each row must already contain both lane vectors
        (summary_embedding, code_embedding). Callers build rows via
        build_symbol_row() below.

        Requires the File node to exist (MATCH, not MERGE) — the
        indexer MUST upsert files before symbols.
        """
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MATCH (f:{S.LABEL_FILE} {{org_id: row.org_id, repo: row.repo, file_path: row.file_path}})
        MERGE (s:{S.LABEL_SYMBOL} {{
            org_id: row.org_id,
            repo: row.repo,
            file_path: row.file_path,
            qualified_name: row.qualified_name
        }})
        SET s.symbol_name = row.symbol_name,
            s.symbol_type = row.symbol_type,
            s.language = row.language,
            s.signature = row.signature,
            s.docstring = row.docstring,
            s.summary = row.summary,
            s.summary_source = row.summary_source,
            s.raw_code = row.raw_code,
            s.content_hash = row.content_hash,
            s.signature_hash = row.signature_hash,
            s.start_line = row.start_line,
            s.end_line = row.end_line,
            s.line_count = row.line_count,
            s.parent_class = row.parent_class,
            s.is_public = row.is_public,
            s.is_entrypoint = row.is_entrypoint,
            s.complexity_bucket = row.complexity_bucket,
            s.commit_sha = row.commit_sha,
            s.summary_embedding = row.summary_embedding,
            s.code_embedding = row.code_embedding,
            s.indexed_at = datetime(),
            s.scanner_version = $scanner_version
        MERGE (f)-[:{S.REL_DEFINES}]->(s)
        """
        with self._session() as session:
            session.run(
                cypher, rows=list(rows),
                scanner_version=S.SCANNER_VERSION,
            )

    def get_file_symbol_hashes(
        self, org_id: str, repo: str, file_path: str,
    ) -> Dict[str, str]:
        """Return {qualified_name: content_hash} for all symbols in a file.
        The indexer diffs this against the newly parsed set to decide
        what to re-embed and what to delete."""
        cypher = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{org_id: $org_id, repo: $repo, file_path: $file_path}})
        RETURN s.qualified_name AS qname, s.content_hash AS hash
        """
        with self._session() as session:
            result = session.run(
                cypher, org_id=org_id, repo=repo, file_path=file_path,
            )
            return {r["qname"]: r["hash"] for r in result}

    def get_symbol_code(
        self, org_id: str, repo: str, file_path: str, symbol_name: str,
    ) -> Optional[str]:
        """Return raw code stored on the Symbol node."""
        cypher = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{
            org_id: $org_id, repo: $repo,
            file_path: $file_path, qualified_name: $symbol_name
        }})
        RETURN s.raw_code AS code
        """
        with self._session() as session:
            record = session.run(
                cypher, org_id=org_id, repo=repo,
                file_path=file_path, symbol_name=symbol_name,
            ).single()
            return record["code"] if record else None

    def update_symbol_summary(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        symbol_name: str,
        summary: str,
        summary_source: str,
        summary_embedding: List[float],
    ) -> None:
        """Enricher hook — rewrites the summary lane only.

        INVARIANT: never writes to code_embedding. The code lane is
        built from raw source at scan time and is considered permanent.
        """
        cypher = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{
            org_id: $org_id, repo: $repo,
            file_path: $file_path, qualified_name: $symbol_name
        }})
        SET s.summary = $summary,
            s.summary_source = $summary_source,
            s.summary_embedding = $summary_embedding,
            s.indexed_at = datetime()
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, file_path=file_path,
                symbol_name=symbol_name, summary=summary,
                summary_source=summary_source,
                summary_embedding=summary_embedding,
            )

    def update_file_summary(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        summary: str,
        summary_source: str,
        summary_embedding: List[float],
    ) -> None:
        """Overwrite the file summary lane with LLM output."""
        cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{
            org_id: $org_id, repo: $repo, file_path: $file_path
        }})
        SET f.summary = $summary,
            f.summary_source = $summary_source,
            f.indexed_at = datetime()
        """
        vector_cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{
            org_id: $org_id, repo: $repo, file_path: $file_path
        }})
        CALL db.create.setNodeVectorProperty(f, 'summary_embedding', $summary_embedding)
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, file_path=file_path,
                summary=summary, summary_source=summary_source,
            )
            session.run(
                vector_cypher,
                org_id=org_id, repo=repo, file_path=file_path,
                summary_embedding=summary_embedding,
            )

    def update_directory_summary(
        self,
        org_id: str,
        repo: str,
        dir_path: str,
        summary: str,
        summary_source: str,
    ) -> None:
        """Overwrite the directory summary with LLM output."""
        cypher = f"""
        MATCH (d:{S.LABEL_DIRECTORY} {{
            org_id: $org_id, repo: $repo, dir_path: $dir_path
        }})
        SET d.summary = $summary,
            d.summary_source = $summary_source
        """
        with self._session() as session:
            session.run(
                cypher, org_id=org_id, repo=repo, dir_path=dir_path,
                summary=summary, summary_source=summary_source,
            )

    # =================================================================
    # CASCADING DELETES
    # =================================================================
    # DETACH DELETE drops incident edges but not attached target nodes,
    # so file deletion explicitly pulls its Symbol children along with
    # it. Fixes the v0 bug where Neo4j leaked symbol nodes.
    # -----------------------------------------------------------------

    def delete_symbol(
        self, org_id: str, repo: str, file_path: str, symbol_name: str,
    ) -> None:
        cypher = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{
            org_id: $org_id, repo: $repo,
            file_path: $file_path, qualified_name: $symbol_name
        }})
        DETACH DELETE s
        """
        with self._session() as session:
            session.run(
                cypher, org_id=org_id, repo=repo,
                file_path=file_path, symbol_name=symbol_name,
            )

    def delete_file(
        self, org_id: str, repo: str, file_path: str,
    ) -> None:
        """DETACH DELETE the File AND every Symbol it defines.

        OPTIONAL MATCH on the symbols so a File with zero symbols still
        deletes cleanly. DETACH DELETE of a null is a no-op in Cypher.
        """
        cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{org_id: $org_id, repo: $repo, file_path: $file_path}})
        OPTIONAL MATCH (f)-[:{S.REL_DEFINES}]->(s:{S.LABEL_SYMBOL})
        DETACH DELETE s, f
        """
        with self._session() as session:
            session.run(
                cypher, org_id=org_id, repo=repo, file_path=file_path,
            )

    def delete_repository(self, org_id: str, repo: str) -> int:
        """Nuke everything for a repo. Returns the count of nodes removed.

        Uses a read-then-write transaction because RETURN-after-DELETE
        can't see the nodes it just dropped. Every v1 node carries
        org_id/repo so a flat filter matches Repository, Directory,
        File, Symbol, and Scan in one pass.
        """
        def _work(tx):
            count = tx.run(
                "MATCH (n) WHERE n.org_id = $org_id AND n.repo = $repo "
                "RETURN count(n) AS c",
                org_id=org_id, repo=repo,
            ).single()["c"]
            tx.run(
                "MATCH (n) WHERE n.org_id = $org_id AND n.repo = $repo "
                "DETACH DELETE n",
                org_id=org_id, repo=repo,
            )
            return int(count)

        with self._session() as session:
            return session.execute_write(_work)

    # =================================================================
    # EDGES
    # =================================================================

    def add_calls_edge(
        self,
        org_id: str,
        repo: str,
        caller_file: str,
        caller_qname: str,
        callee_file: str,
        callee_qname: str,
        is_direct: bool = True,
        is_ambiguous: bool = False,
    ) -> None:
        """Symbol → Symbol CALLS edge.

        is_ambiguous flags edges whose callee was matched by flat name
        alone (shortcoming #2 — resolved in a later AST pass).
        """
        cypher = f"""
        MATCH (caller:{S.LABEL_SYMBOL} {{
            org_id: $org_id, repo: $repo,
            file_path: $caller_file, qualified_name: $caller_qname
        }})
        MATCH (callee:{S.LABEL_SYMBOL} {{
            org_id: $org_id, repo: $repo,
            file_path: $callee_file, qualified_name: $callee_qname
        }})
        MERGE (caller)-[c:{S.REL_CALLS}]->(callee)
        SET c.is_direct = $is_direct,
            c.is_ambiguous = $is_ambiguous
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo,
                caller_file=caller_file, caller_qname=caller_qname,
                callee_file=callee_file, callee_qname=callee_qname,
                is_direct=is_direct, is_ambiguous=is_ambiguous,
            )

    def add_calls_edges_batch(
        self, rows: Sequence[Dict[str, Any]],
    ) -> None:
        """Batched version — full scans create thousands of edges."""
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MATCH (caller:{S.LABEL_SYMBOL} {{
            org_id: row.org_id, repo: row.repo,
            file_path: row.caller_file, qualified_name: row.caller_qname
        }})
        MATCH (callee:{S.LABEL_SYMBOL} {{
            org_id: row.org_id, repo: row.repo,
            file_path: row.callee_file, qualified_name: row.callee_qname
        }})
        MERGE (caller)-[c:{S.REL_CALLS}]->(callee)
        SET c.is_direct = coalesce(row.is_direct, true),
            c.is_ambiguous = coalesce(row.is_ambiguous, false)
        """
        with self._session() as session:
            session.run(cypher, rows=list(rows))

    def add_imports_edge(
        self,
        org_id: str,
        repo: str,
        importer_file: str,
        imported_file: str,
        import_type: str = S.IMPORT_TYPE_MODULE,
    ) -> None:
        """File → File IMPORTS edge."""
        cypher = f"""
        MATCH (importer:{S.LABEL_FILE} {{
            org_id: $org_id, repo: $repo, file_path: $importer_file
        }})
        MATCH (imported:{S.LABEL_FILE} {{
            org_id: $org_id, repo: $repo, file_path: $imported_file
        }})
        MERGE (importer)-[i:{S.REL_IMPORTS}]->(imported)
        SET i.import_type = $import_type
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo,
                importer_file=importer_file,
                imported_file=imported_file,
                import_type=import_type,
            )

    def add_imports_edges_batch(
        self, rows: Sequence[Dict[str, Any]],
    ) -> None:
        if not rows:
            return
        cypher = f"""
        UNWIND $rows AS row
        MATCH (importer:{S.LABEL_FILE} {{
            org_id: row.org_id, repo: row.repo, file_path: row.importer_file
        }})
        MATCH (imported:{S.LABEL_FILE} {{
            org_id: row.org_id, repo: row.repo, file_path: row.imported_file
        }})
        MERGE (importer)-[i:{S.REL_IMPORTS}]->(imported)
        SET i.import_type = coalesce(row.import_type, '{S.IMPORT_TYPE_MODULE}')
        """
        with self._session() as session:
            session.run(cypher, rows=list(rows))

    # =================================================================
    # SCAN STATE
    # =================================================================
    # v0 kept scan state in Mongo. v1 puts it in Neo4j as Scan nodes
    # attached to the Repository. Same shape, one less database.
    # -----------------------------------------------------------------

    def start_scan(
        self, org_id: str, repo: str, commit_sha: str,
    ) -> str:
        """Create a Scan node with status='running'. Returns scan_id."""
        scan_id = str(uuid4())
        cypher = f"""
        MATCH (r:{S.LABEL_REPOSITORY} {{org_id: $org_id, repo: $repo}})
        CREATE (sc:{S.LABEL_SCAN} {{
            org_id: $org_id,
            repo: $repo,
            scan_id: $scan_id,
            commit_sha: $commit_sha,
            scan_status: $status,
            started_at: datetime(),
            scanner_version: $scanner_version
        }})
        MERGE (r)-[:{S.REL_HAS_SCAN}]->(sc)
        RETURN sc.scan_id AS scan_id
        """
        with self._session() as session:
            record = session.run(
                cypher,
                org_id=org_id, repo=repo,
                commit_sha=commit_sha, scan_id=scan_id,
                status=S.SCAN_STATUS_RUNNING,
                scanner_version=S.SCANNER_VERSION,
            ).single()
            return record["scan_id"]

    def complete_scan(
        self,
        org_id: str,
        repo: str,
        scan_id: str,
        commit_sha: str,
        stats: Dict[str, Any],
        duration_seconds: float,
    ) -> None:
        """Mark a specific scan complete and record stats.

        stats is JSON-encoded — Neo4j does not accept Map values as
        node properties.
        """
        cypher = f"""
        MATCH (sc:{S.LABEL_SCAN} {{org_id: $org_id, repo: $repo, scan_id: $scan_id}})
        SET sc.scan_status = $status,
            sc.commit_sha = $commit_sha,
            sc.finished_at = datetime(),
            sc.duration_seconds = $duration_seconds,
            sc.stats = $stats
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, scan_id=scan_id,
                commit_sha=commit_sha,
                status=S.SCAN_STATUS_COMPLETE,
                duration_seconds=float(duration_seconds),
                stats=json.dumps(stats or {}),
            )

    def fail_scan(
        self,
        org_id: str,
        repo: str,
        scan_id: str,
        error_message: str,
    ) -> None:
        """Mark a specific scan failed and record the error."""
        cypher = f"""
        MATCH (sc:{S.LABEL_SCAN} {{org_id: $org_id, repo: $repo, scan_id: $scan_id}})
        SET sc.scan_status = $status,
            sc.finished_at = datetime(),
            sc.error_message = $error_message
        """
        with self._session() as session:
            session.run(
                cypher,
                org_id=org_id, repo=repo, scan_id=scan_id,
                status=S.SCAN_STATUS_FAILED,
                error_message=error_message,
            )

    def get_last_commit_sha(
        self, org_id: str, repo: str,
    ) -> Optional[str]:
        """Return the commit_sha of the most recent completed scan,
        or None if this repo has never been successfully scanned."""
        cypher = f"""
        MATCH (sc:{S.LABEL_SCAN} {{
            org_id: $org_id, repo: $repo, scan_status: $status
        }})
        RETURN sc.commit_sha AS commit_sha
        ORDER BY sc.finished_at DESC
        LIMIT 1
        """
        with self._session() as session:
            record = session.run(
                cypher,
                org_id=org_id, repo=repo,
                status=S.SCAN_STATUS_COMPLETE,
            ).single()
            return record["commit_sha"] if record else None

    # =================================================================
    # READ PATHS (used later by retrieval_v1)
    # =================================================================
    # Stubbed here so the scanner side knows what it is feeding.
    # Actual retrieval_v1 calls them. Included now so schema changes
    # don't need a second pass later.
    # -----------------------------------------------------------------

    def vector_search_symbols(
        self,
        query_vector: List[float],
        lane: str,                      # "summary" | "code"
        org_id: str,
        repo: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """ANN search over one symbol lane, filtered by org_id (and repo).

        Neo4j vector indexes don't accept structured predicates natively,
        so we over-fetch then post-filter. k is bumped above top_k to
        leave headroom after the org/repo filter.
        """
        if lane == "summary":
            index_name = S.IDX_SYMBOL_SUMMARY_VEC
        elif lane == "code":
            index_name = S.IDX_SYMBOL_CODE_VEC
        else:
            raise ValueError(
                f"unknown lane: {lane!r} (expected 'summary' or 'code')"
            )

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
        YIELD node, score
        WHERE node.org_id = $org_id
          AND ($repo IS NULL OR node.repo = $repo)
        RETURN node.qualified_name  AS qualified_name,
               node.symbol_type     AS symbol_type,
               node.signature       AS signature,
               node.docstring       AS docstring,
               node.summary         AS summary,
               node.file_path       AS file_path,
               node.repo            AS repo,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        k = max(top_k * 4, 50)
        with self._session() as session:
            result = session.run(
                cypher,
                index_name=index_name,
                k=k,
                query_vector=query_vector,
                org_id=org_id, repo=repo, top_k=top_k,
            )
            return [r.data() for r in result]

    def vector_search_files(
        self,
        query_vector: List[float],
        org_id: str,
        repo: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """ANN search over the file summary lane."""
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
        YIELD node, score
        WHERE node.org_id = $org_id
          AND ($repo IS NULL OR node.repo = $repo)
        RETURN node.file_path AS file_path,
               node.language  AS language,
               node.summary   AS summary,
               node.repo      AS repo,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        k = max(top_k * 4, 50)
        with self._session() as session:
            result = session.run(
                cypher,
                index_name=S.IDX_FILE_SUMMARY_VEC,
                k=k,
                query_vector=query_vector,
                org_id=org_id, repo=repo, top_k=top_k,
            )
            return [r.data() for r in result]

    def fulltext_search_symbols(
        self,
        query_text: str,
        org_id: str,
        repo: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        import re
        """BM25 lane over (qualified_name, signature, raw_code, summary)."""
        # Sanitize Lucene special characters to prevent parse errors
        query_text = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', ' ', query_text)
        query_text = re.sub(r'\s+', ' ', query_text).strip()
        if not query_text:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $query_text)
        YIELD node, score
        WHERE node.org_id = $org_id
          AND ($repo IS NULL OR node.repo = $repo)
        RETURN node.qualified_name  AS qualified_name,
               node.symbol_type     AS symbol_type,
               node.signature       AS signature,
               node.docstring       AS docstring,
               node.summary         AS summary,
               node.file_path       AS file_path,
               node.repo            AS repo,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self._session() as session:
            result = session.run(
                cypher,
                index_name=S.IDX_SYMBOL_FULLTEXT,
                query_text=query_text,
                org_id=org_id, repo=repo, top_k=top_k,
            )
            return [r.data() for r in result]

    def get_repo_directories(
        self,
        org_id: str,
        repo: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of all directories in the repo, primarily for overview/structure queries."""
        cypher = f"""
        MATCH (d:{S.LABEL_DIRECTORY} {{org_id: $org_id}})
        WHERE ($repo IS NULL OR d.repo = $repo)
        RETURN d.dir_path AS dir_path, d.summary AS summary, d.file_count AS file_count
        ORDER BY d.dir_path
        """
        with self._session() as session:
            result = session.run(cypher, org_id=org_id, repo=repo)
            return [r.data() for r in result]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
# Kept at module scope so the indexer can construct rows directly when it
# wants to call upsert_symbols_batch without going through the single-row
# path.  Pure function, no Neo4j, no I/O.
# ---------------------------------------------------------------------------

def build_symbol_row(
    org_id: str,
    repo: str,
    file_path: str,
    symbol: Dict[str, Any],
    embedding: SymbolEmbedding,
    commit_sha: str,
) -> Dict[str, Any]:
    """Convert a ParsedSymbol-as-dict + SymbolEmbedding into a write row.

    Missing fields fall back to safe defaults so a minimally-populated
    symbol dict still upserts cleanly. The indexer is expected to pass
    a rich dict (raw_code, content_hash, line ranges, ...) — the
    defaults are the graceful-failure path, not the main path.
    """
    qualified_name = symbol["qualified_name"]
    return {
        "org_id": org_id,
        "repo": repo,
        "file_path": file_path,
        "qualified_name": qualified_name,
        "symbol_name": symbol.get("symbol_name")
                       or qualified_name.rsplit(".", 1)[-1],
        "symbol_type": symbol.get("symbol_type", S.SYMBOL_TYPE_FUNCTION),
        "language": symbol.get("language", ""),
        "signature": symbol.get("signature", ""),
        "docstring": symbol.get("docstring", ""),
        "summary": symbol.get("summary", ""),
        "summary_source": symbol.get("summary_source", S.SUMMARY_SOURCE_AST),
        "raw_code": symbol.get("raw_code", ""),
        "content_hash": symbol.get("content_hash", ""),
        "signature_hash": symbol.get("signature_hash", ""),
        "start_line": int(symbol.get("start_line", 0)),
        "end_line": int(symbol.get("end_line", 0)),
        "line_count": int(symbol.get("line_count", 0)),
        "parent_class": symbol.get("parent_class"),
        "is_public": bool(symbol.get("is_public", True)),
        "is_entrypoint": bool(symbol.get("is_entrypoint", False)),
        "complexity_bucket": symbol.get("complexity_bucket", S.COMPLEXITY_LOW),
        "commit_sha": commit_sha,
        "summary_embedding": embedding.summary_vector,
        "code_embedding": embedding.code_vector,
    }
