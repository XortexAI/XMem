"""
Neo4j Code Graph Client — CRUD + traversal for the code knowledge graph.

Manages Repository → Directory → File → Symbol hierarchy and
Annotation → ANNOTATES edges.  Also provides traversal queries for
impact analysis (callers/callees, importers, inheritance chains).

Usage::

    from src.graph.code_graph_client import CodeGraphClient

    client = CodeGraphClient(uri, user, password)
    client.connect()
    client.setup()
    client.upsert_repository("razorpay", "payment-service")
    client.upsert_symbol({...})
    callers = client.get_callers("razorpay", "payment-service", "PaymentProcessor.process")
    client.close()
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from neo4j import GraphDatabase

from src.graph.code_schema import CodeGraphSchema as S, setup_code_constraints

logger = logging.getLogger("xmem.graph.code")

F = TypeVar("F", bound=Callable)


# ---------------------------------------------------------------------------
# Retry decorator (same pattern as the temporal client)
# ---------------------------------------------------------------------------

def _retry(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: "CodeGraphClient", *args, **kwargs):
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
                    try:
                        self.close()
                        self.connect()
                    except Exception as conn_err:
                        logger.error("Reconnect failed: %s", conn_err)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


class CodeGraphClient:
    """Manages the Neo4j driver and provides CRUD + traversal for code knowledge."""

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
        logger.info("Code graph connected to Neo4j at %s", self._uri)

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

    def setup(self) -> None:
        """Create constraints and indexes for the code graph."""
        if self._driver:
            setup_code_constraints(self._driver)
            logger.info("Code graph constraints created.")

    # ======================================================================
    # STRUCTURAL NODES — Repository / Directory / File / Symbol
    # ======================================================================

    @_retry()
    def upsert_repository(self, org_id: str, repo_name: str) -> None:
        query = f"""
        MERGE (r:{S.LABEL_REPOSITORY} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO_NAME}: $repo_name
        }})
        RETURN r
        """
        with self._session() as session:
            session.run(query, org_id=org_id, repo_name=repo_name)

    @_retry()
    def upsert_directory(
        self,
        org_id: str,
        repo: str,
        dir_path: str,
        summary: str = "",
        languages: Optional[List[str]] = None,
        file_count: int = 0,
        total_symbols: int = 0,
    ) -> None:
        self.upsert_repository(org_id, repo)
        query = f"""
        MERGE (d:{S.LABEL_DIRECTORY} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_DIR_PATH}: $dir_path
        }})
        SET d.{S.PROP_DIR_SUMMARY} = $summary,
            d.{S.PROP_DIR_LANGUAGES} = $languages,
            d.{S.PROP_DIR_FILE_COUNT} = $file_count,
            d.{S.PROP_DIR_TOTAL_SYMBOLS} = $total_symbols,
            d.{S.PROP_INDEXED_AT} = datetime()
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo, dir_path=dir_path,
                summary=summary, languages=languages or [],
                file_count=file_count, total_symbols=total_symbols,
            )

        self._link_directory_to_parent(org_id, repo, dir_path)

    @_retry()
    def upsert_file(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        language: str = "",
        summary: str = "",
        symbol_count: int = 0,
        symbol_names: Optional[List[str]] = None,
        total_lines: int = 0,
        commit_sha: str = "",
    ) -> None:
        dir_path = "/".join(file_path.split("/")[:-1]) + "/"
        if dir_path == "/":
            dir_path = "/"
        self.upsert_directory(org_id, repo, dir_path)

        query = f"""
        MERGE (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        SET f.{S.PROP_LANGUAGE} = $language,
            f.{S.PROP_FILE_SUMMARY} = $summary,
            f.{S.PROP_SYMBOL_COUNT} = $symbol_count,
            f.{S.PROP_SYMBOL_NAMES} = $symbol_names,
            f.{S.PROP_TOTAL_LINES} = $total_lines,
            f.{S.PROP_COMMIT_SHA} = $commit_sha,
            f.{S.PROP_INDEXED_AT} = datetime()
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo, file_path=file_path,
                language=language, summary=summary,
                symbol_count=symbol_count, symbol_names=symbol_names or [],
                total_lines=total_lines, commit_sha=commit_sha,
            )

        self._link_file_to_directory(org_id, repo, file_path, dir_path)

    @_retry()
    def upsert_symbol(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        symbol_name: str,
        symbol_type: str,
        summary: str = "",
        signature: str = "",
        docstring: str = "",
        start_line: int = 0,
        end_line: int = 0,
        commit_sha: str = "",
        branch: str = "main",
        parent_class: Optional[str] = None,
        imports: Optional[List[str]] = None,
        signature_hash: str = "",
        is_public: bool = True,
        is_entrypoint: bool = False,
        complexity_bucket: str = "medium",
        line_count: int = 0,
        language: str = "",
    ) -> None:
        self.upsert_file(org_id, repo, file_path, language=language)

        query = f"""
        MERGE (s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path,
            {S.PROP_SYMBOL_NAME}: $symbol_name,
            {S.PROP_SIGNATURE_HASH}: $signature_hash
        }})
        SET s.{S.PROP_SYMBOL_TYPE} = $symbol_type,
            s.{S.PROP_SUMMARY} = $summary,
            s.{S.PROP_SIGNATURE} = $signature,
            s.{S.PROP_DOCSTRING} = $docstring,
            s.{S.PROP_START_LINE} = $start_line,
            s.{S.PROP_END_LINE} = $end_line,
            s.{S.PROP_COMMIT_SHA} = $commit_sha,
            s.{S.PROP_BRANCH} = $branch,
            s.{S.PROP_PARENT_CLASS} = $parent_class,
            s.{S.PROP_IMPORTS} = $imports,
            s.{S.PROP_IS_PUBLIC} = $is_public,
            s.{S.PROP_IS_ENTRYPOINT} = $is_entrypoint,
            s.{S.PROP_COMPLEXITY} = $complexity_bucket,
            s.{S.PROP_LINE_COUNT} = $line_count,
            s.{S.PROP_LANGUAGE} = $language,
            s.{S.PROP_INDEXED_AT} = datetime()
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo, file_path=file_path,
                symbol_name=symbol_name, symbol_type=symbol_type,
                summary=summary, signature=signature, docstring=docstring,
                start_line=start_line, end_line=end_line,
                commit_sha=commit_sha, branch=branch,
                parent_class=parent_class, imports=imports or [],
                signature_hash=signature_hash,
                is_public=is_public, is_entrypoint=is_entrypoint,
                complexity_bucket=complexity_bucket, line_count=line_count,
                language=language,
            )

        self._link_symbol_to_file(org_id, repo, file_path, symbol_name, signature_hash)

    # ======================================================================
    # DEPENDENCY EDGES (scanner-populated)
    # ======================================================================

    @_retry()
    def add_calls_edge(
        self,
        org_id: str,
        repo: str,
        caller_name: str,
        caller_file: str,
        callee_name: str,
        callee_file: str,
        call_count: int = 1,
        is_direct: bool = True,
    ) -> None:
        """Create a CALLS edge between two symbols."""
        query = f"""
        MATCH (caller:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $caller_name, {S.PROP_FILE_PATH}: $caller_file
        }})
        MATCH (callee:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $callee_name, {S.PROP_FILE_PATH}: $callee_file
        }})
        MERGE (caller)-[r:{S.REL_CALLS}]->(callee)
        SET r.{S.PROP_CALL_COUNT} = $call_count,
            r.{S.PROP_IS_DIRECT} = $is_direct
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo,
                caller_name=caller_name, caller_file=caller_file,
                callee_name=callee_name, callee_file=callee_file,
                call_count=call_count, is_direct=is_direct,
            )

    @_retry()
    def add_imports_edge(
        self,
        org_id: str,
        repo: str,
        importer_file: str,
        imported_file: str,
        import_type: str = "direct",
    ) -> None:
        query = f"""
        MATCH (importer:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $importer_file
        }})
        MATCH (imported:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $imported_file
        }})
        MERGE (importer)-[r:{S.REL_IMPORTS}]->(imported)
        SET r.{S.PROP_IMPORT_TYPE} = $import_type
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo,
                importer_file=importer_file, imported_file=imported_file,
                import_type=import_type,
            )

    @_retry()
    def add_inherits_edge(
        self,
        org_id: str,
        repo: str,
        child_name: str,
        child_file: str,
        parent_name: str,
        parent_file: str,
    ) -> None:
        query = f"""
        MATCH (child:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $child_name, {S.PROP_FILE_PATH}: $child_file
        }})
        MATCH (parent:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $parent_name, {S.PROP_FILE_PATH}: $parent_file
        }})
        MERGE (child)-[:{S.REL_INHERITS}]->(parent)
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo,
                child_name=child_name, child_file=child_file,
                parent_name=parent_name, parent_file=parent_file,
            )

    @_retry()
    def add_implements_edge(
        self,
        org_id: str,
        repo: str,
        implementor_name: str,
        implementor_file: str,
        interface_name: str,
        interface_file: str,
    ) -> None:
        query = f"""
        MATCH (impl:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $implementor_name, {S.PROP_FILE_PATH}: $implementor_file
        }})
        MATCH (iface:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $interface_name, {S.PROP_FILE_PATH}: $interface_file
        }})
        MERGE (impl)-[:{S.REL_IMPLEMENTS}]->(iface)
        """
        with self._session() as session:
            session.run(
                query,
                org_id=org_id, repo=repo,
                implementor_name=implementor_name, implementor_file=implementor_file,
                interface_name=interface_name, interface_file=interface_file,
            )

    # ======================================================================
    # ANNOTATIONS
    # ======================================================================

    @_retry()
    def create_annotation(
        self,
        org_id: str,
        content: str,
        annotation_type: str = "explanation",
        severity: Optional[str] = None,
        author_id: Optional[str] = None,
        author_name: Optional[str] = None,
        repo: Optional[str] = None,
        target_file: Optional[str] = None,
        target_symbol: Optional[str] = None,
        status: str = "active",
    ) -> str:
        """Create an annotation node and link it to its target (symbol/file/directory)."""
        annotation_id = str(uuid.uuid4())

        query = f"""
        CREATE (a:{S.LABEL_ANNOTATION} {{
            {S.PROP_ANNOTATION_ID}: $annotation_id,
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_CONTENT}: $content,
            {S.PROP_ANNOTATION_TYPE}: $annotation_type,
            {S.PROP_SEVERITY}: $severity,
            {S.PROP_AUTHOR_ID}: $author_id,
            {S.PROP_AUTHOR_NAME}: $author_name,
            {S.PROP_STATUS}: $status,
            {S.PROP_CREATED_AT}: datetime()
        }})
        RETURN a.{S.PROP_ANNOTATION_ID} as id
        """
        with self._session() as session:
            session.run(
                query,
                annotation_id=annotation_id, org_id=org_id, repo=repo,
                content=content, annotation_type=annotation_type,
                severity=severity, author_id=author_id,
                author_name=author_name, status=status,
            )

        # Link to target
        if target_symbol and repo and target_file:
            self._link_annotation_to_symbol(
                annotation_id, org_id, repo, target_file, target_symbol,
            )
        elif target_file and repo:
            self._link_annotation_to_file(annotation_id, org_id, repo, target_file)

        logger.info(
            "Created annotation %s targeting %s",
            annotation_id, target_symbol or target_file or "(org-level)",
        )
        return annotation_id

    @_retry()
    def update_annotation_status(
        self,
        annotation_id: str,
        status: str,
        resolved_by: Optional[str] = None,
    ) -> bool:
        query = f"""
        MATCH (a:{S.LABEL_ANNOTATION} {{ {S.PROP_ANNOTATION_ID}: $annotation_id }})
        SET a.{S.PROP_STATUS} = $status
        """
        if status == "resolved":
            query += f", a.{S.PROP_RESOLVED_AT} = datetime()"
            if resolved_by:
                query += f", a.{S.PROP_RESOLVED_BY} = $resolved_by"
        query += " RETURN a"

        with self._session() as session:
            result = session.run(
                query,
                annotation_id=annotation_id, status=status,
                resolved_by=resolved_by,
            )
            return result.single() is not None

    @_retry()
    def get_annotations_for_symbol(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        status_filter = f"AND a.{S.PROP_STATUS} = $status" if status else ""
        query = f"""
        MATCH (a:{S.LABEL_ANNOTATION})-[:{S.REL_ANNOTATES}]->(s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        WHERE a.{S.PROP_ORG_ID} = $org_id {status_filter}
        RETURN a.{S.PROP_ANNOTATION_ID} as id,
               a.{S.PROP_CONTENT} as content,
               a.{S.PROP_ANNOTATION_TYPE} as annotation_type,
               a.{S.PROP_SEVERITY} as severity,
               a.{S.PROP_AUTHOR_NAME} as author_name,
               a.{S.PROP_STATUS} as status,
               a.{S.PROP_CREATED_AT} as created_at
        ORDER BY a.{S.PROP_CREATED_AT} DESC
        """
        results = []
        with self._session() as session:
            records = session.run(
                query, org_id=org_id, repo=repo,
                symbol_name=symbol_name, status=status,
            )
            for r in records:
                results.append(dict(r))
        return results

    @_retry()
    def get_annotations_for_file(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        status_filter = f"AND a.{S.PROP_STATUS} = $status" if status else ""
        query = f"""
        MATCH (a:{S.LABEL_ANNOTATION})-[:{S.REL_ANNOTATES}]->(f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        WHERE a.{S.PROP_ORG_ID} = $org_id {status_filter}
        RETURN a.{S.PROP_ANNOTATION_ID} as id,
               a.{S.PROP_CONTENT} as content,
               a.{S.PROP_ANNOTATION_TYPE} as annotation_type,
               a.{S.PROP_SEVERITY} as severity,
               a.{S.PROP_AUTHOR_NAME} as author_name,
               a.{S.PROP_STATUS} as status,
               a.{S.PROP_CREATED_AT} as created_at
        ORDER BY a.{S.PROP_CREATED_AT} DESC
        """
        results = []
        with self._session() as session:
            records = session.run(
                query, org_id=org_id, repo=repo,
                file_path=file_path, status=status,
            )
            for r in records:
                results.append(dict(r))
        return results

    # ======================================================================
    # GRAPH TRAVERSAL — Impact Analysis
    # ======================================================================

    @_retry()
    def get_callers(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Find all symbols that call the given symbol (up to N hops)."""
        query = f"""
        MATCH (target:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        MATCH (caller:{S.LABEL_SYMBOL})-[r:{S.REL_CALLS}*1..{depth}]->(target)
        RETURN DISTINCT
            caller.{S.PROP_SYMBOL_NAME} as symbol_name,
            caller.{S.PROP_FILE_PATH} as file_path,
            caller.{S.PROP_SYMBOL_TYPE} as symbol_type,
            caller.{S.PROP_SUMMARY} as summary,
            length(r) as distance
        ORDER BY distance, caller.{S.PROP_SYMBOL_NAME}
        """
        results = []
        with self._session() as session:
            records = session.run(
                query, org_id=org_id, repo=repo,
                symbol_name=symbol_name,
            )
            for r in records:
                results.append(dict(r))
        return results

    @_retry()
    def get_callees(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Find all symbols that the given symbol calls (up to N hops)."""
        query = f"""
        MATCH (source:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        MATCH (source)-[r:{S.REL_CALLS}*1..{depth}]->(callee:{S.LABEL_SYMBOL})
        RETURN DISTINCT
            callee.{S.PROP_SYMBOL_NAME} as symbol_name,
            callee.{S.PROP_FILE_PATH} as file_path,
            callee.{S.PROP_SYMBOL_TYPE} as symbol_type,
            callee.{S.PROP_SUMMARY} as summary,
            length(r) as distance
        ORDER BY distance, callee.{S.PROP_SYMBOL_NAME}
        """
        results = []
        with self._session() as session:
            records = session.run(
                query, org_id=org_id, repo=repo,
                symbol_name=symbol_name,
            )
            for r in records:
                results.append(dict(r))
        return results

    @_retry()
    def get_inheritance_chain(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
    ) -> List[Dict[str, Any]]:
        """Get the full inheritance/implementation chain for a symbol."""
        query = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        OPTIONAL MATCH (s)-[:{S.REL_INHERITS}*1..5]->(parent:{S.LABEL_SYMBOL})
        OPTIONAL MATCH (child:{S.LABEL_SYMBOL})-[:{S.REL_INHERITS}*1..5]->(s)
        RETURN
            collect(DISTINCT {{
                name: parent.{S.PROP_SYMBOL_NAME},
                file: parent.{S.PROP_FILE_PATH},
                relation: 'parent'
            }}) as parents,
            collect(DISTINCT {{
                name: child.{S.PROP_SYMBOL_NAME},
                file: child.{S.PROP_FILE_PATH},
                relation: 'child'
            }}) as children
        """
        with self._session() as session:
            result = session.run(
                query, org_id=org_id, repo=repo,
                symbol_name=symbol_name,
            )
            record = result.single()
            if not record:
                return []

            chain = []
            for p in record["parents"]:
                if p["name"]:
                    chain.append(p)
            for c in record["children"]:
                if c["name"]:
                    chain.append(c)
            return chain

    @_retry()
    def get_file_imports(
        self,
        org_id: str,
        repo: str,
        file_path: str,
    ) -> Dict[str, List[str]]:
        """Get files imported by and files that import the given file."""
        query = f"""
        MATCH (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        OPTIONAL MATCH (f)-[:{S.REL_IMPORTS}]->(imported:{S.LABEL_FILE})
        OPTIONAL MATCH (importer:{S.LABEL_FILE})-[:{S.REL_IMPORTS}]->(f)
        RETURN
            collect(DISTINCT imported.{S.PROP_FILE_PATH}) as imports,
            collect(DISTINCT importer.{S.PROP_FILE_PATH}) as imported_by
        """
        with self._session() as session:
            result = session.run(
                query, org_id=org_id, repo=repo, file_path=file_path,
            )
            record = result.single()
            if not record:
                return {"imports": [], "imported_by": []}

            return {
                "imports": [f for f in record["imports"] if f],
                "imported_by": [f for f in record["imported_by"] if f],
            }

    @_retry()
    def get_symbol_details(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get full details for a symbol including its relationships."""
        query = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        RETURN s.{S.PROP_SYMBOL_NAME} as symbol_name,
               s.{S.PROP_SYMBOL_TYPE} as symbol_type,
               s.{S.PROP_FILE_PATH} as file_path,
               s.{S.PROP_SUMMARY} as summary,
               s.{S.PROP_SIGNATURE} as signature,
               s.{S.PROP_DOCSTRING} as docstring,
               s.{S.PROP_START_LINE} as start_line,
               s.{S.PROP_END_LINE} as end_line,
               s.{S.PROP_PARENT_CLASS} as parent_class,
               s.{S.PROP_IS_PUBLIC} as is_public,
               s.{S.PROP_IS_ENTRYPOINT} as is_entrypoint,
               s.{S.PROP_COMPLEXITY} as complexity_bucket,
               s.{S.PROP_LINE_COUNT} as line_count,
               s.{S.PROP_LANGUAGE} as language,
               s.{S.PROP_IMPORTS} as imports
        LIMIT 1
        """
        with self._session() as session:
            result = session.run(
                query, org_id=org_id, repo=repo,
                symbol_name=symbol_name,
            )
            record = result.single()
            return dict(record) if record else None

    @_retry()
    def get_file_symbols(
        self,
        org_id: str,
        repo: str,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Get all symbols defined in a file."""
        query = f"""
        MATCH (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id,
            {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})-[:{S.REL_DEFINES}]->(s:{S.LABEL_SYMBOL})
        RETURN s.{S.PROP_SYMBOL_NAME} as symbol_name,
               s.{S.PROP_SYMBOL_TYPE} as symbol_type,
               s.{S.PROP_SUMMARY} as summary,
               s.{S.PROP_SIGNATURE} as signature,
               s.{S.PROP_IS_PUBLIC} as is_public
        ORDER BY s.{S.PROP_START_LINE}
        """
        results = []
        with self._session() as session:
            records = session.run(
                query, org_id=org_id, repo=repo, file_path=file_path,
            )
            for r in records:
                results.append(dict(r))
        return results

    @_retry()
    def impact_analysis(
        self,
        org_id: str,
        repo: str,
        symbol_name: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Full impact analysis: callers, callees, inheritance, annotations."""
        callers = self.get_callers(org_id, repo, symbol_name, depth=depth)
        callees = self.get_callees(org_id, repo, symbol_name, depth=depth)
        inheritance = self.get_inheritance_chain(org_id, repo, symbol_name)
        annotations = self.get_annotations_for_symbol(org_id, repo, symbol_name)

        return {
            "symbol": symbol_name,
            "callers": callers,
            "callees": callees,
            "inheritance": inheritance,
            "annotations": annotations,
            "total_impact": len(callers),
        }

    # ======================================================================
    # INTERNAL HELPERS — Structural linking
    # ======================================================================

    @_retry()
    def _link_directory_to_parent(
        self, org_id: str, repo: str, dir_path: str,
    ) -> None:
        """Link directory to its parent (Repository or parent Directory)."""
        parts = dir_path.rstrip("/").split("/")
        if len(parts) <= 1:
            query = f"""
            MATCH (r:{S.LABEL_REPOSITORY} {{
                {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO_NAME}: $repo
            }})
            MATCH (d:{S.LABEL_DIRECTORY} {{
                {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
                {S.PROP_DIR_PATH}: $dir_path
            }})
            MERGE (r)-[:{S.REL_CONTAINS_DIR}]->(d)
            """
            with self._session() as session:
                session.run(query, org_id=org_id, repo=repo, dir_path=dir_path)
        else:
            parent_path = "/".join(parts[:-1]) + "/"
            self.upsert_directory(org_id, repo, parent_path)
            query = f"""
            MATCH (parent:{S.LABEL_DIRECTORY} {{
                {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
                {S.PROP_DIR_PATH}: $parent_path
            }})
            MATCH (child:{S.LABEL_DIRECTORY} {{
                {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
                {S.PROP_DIR_PATH}: $dir_path
            }})
            MERGE (parent)-[:{S.REL_CONTAINS_DIR}]->(child)
            """
            with self._session() as session:
                session.run(
                    query, org_id=org_id, repo=repo,
                    parent_path=parent_path, dir_path=dir_path,
                )

    @_retry()
    def _link_file_to_directory(
        self, org_id: str, repo: str, file_path: str, dir_path: str,
    ) -> None:
        query = f"""
        MATCH (d:{S.LABEL_DIRECTORY} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_DIR_PATH}: $dir_path
        }})
        MATCH (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        MERGE (d)-[:{S.REL_CONTAINS_FILE}]->(f)
        """
        with self._session() as session:
            session.run(
                query, org_id=org_id, repo=repo,
                file_path=file_path, dir_path=dir_path,
            )

    @_retry()
    def _link_symbol_to_file(
        self, org_id: str, repo: str, file_path: str,
        symbol_name: str, signature_hash: str,
    ) -> None:
        query = f"""
        MATCH (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        MATCH (s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path,
            {S.PROP_SYMBOL_NAME}: $symbol_name,
            {S.PROP_SIGNATURE_HASH}: $signature_hash
        }})
        MERGE (f)-[:{S.REL_DEFINES}]->(s)
        """
        with self._session() as session:
            session.run(
                query, org_id=org_id, repo=repo,
                file_path=file_path, symbol_name=symbol_name,
                signature_hash=signature_hash,
            )

    @_retry()
    def _link_annotation_to_symbol(
        self, annotation_id: str, org_id: str, repo: str,
        file_path: str, symbol_name: str,
    ) -> None:
        query = f"""
        MATCH (a:{S.LABEL_ANNOTATION} {{ {S.PROP_ANNOTATION_ID}: $annotation_id }})
        MATCH (s:{S.LABEL_SYMBOL} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path,
            {S.PROP_SYMBOL_NAME}: $symbol_name
        }})
        MERGE (a)-[r:{S.REL_ANNOTATES}]->(s)
        SET r.{S.PROP_RELEVANCE} = 'direct'
        """
        with self._session() as session:
            session.run(
                query, annotation_id=annotation_id,
                org_id=org_id, repo=repo,
                file_path=file_path, symbol_name=symbol_name,
            )

    @_retry()
    def _link_annotation_to_file(
        self, annotation_id: str, org_id: str, repo: str, file_path: str,
    ) -> None:
        query = f"""
        MATCH (a:{S.LABEL_ANNOTATION} {{ {S.PROP_ANNOTATION_ID}: $annotation_id }})
        MATCH (f:{S.LABEL_FILE} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO}: $repo,
            {S.PROP_FILE_PATH}: $file_path
        }})
        MERGE (a)-[:{S.REL_ANNOTATES}]->(f)
        """
        with self._session() as session:
            session.run(
                query, annotation_id=annotation_id,
                org_id=org_id, repo=repo, file_path=file_path,
            )

    # ======================================================================
    # CLEANUP / UTILITY
    # ======================================================================

    @_retry()
    def delete_repository(self, org_id: str, repo: str) -> int:
        """Delete a repository and all its children (directories, files, symbols)."""
        query = f"""
        MATCH (r:{S.LABEL_REPOSITORY} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO_NAME}: $repo
        }})
        OPTIONAL MATCH (r)-[*]->(n)
        DETACH DELETE r, n
        RETURN count(n) as deleted
        """
        with self._session() as session:
            result = session.run(query, org_id=org_id, repo=repo)
            record = result.single()
            return record["deleted"] if record else 0

    @_retry()
    def get_repo_stats(self, org_id: str, repo: str) -> Dict[str, int]:
        """Get counts of directories, files, and symbols for a repository."""
        query = f"""
        MATCH (r:{S.LABEL_REPOSITORY} {{
            {S.PROP_ORG_ID}: $org_id, {S.PROP_REPO_NAME}: $repo
        }})
        OPTIONAL MATCH (r)-[*]->(d:{S.LABEL_DIRECTORY})
        WITH r, count(DISTINCT d) as dir_count
        OPTIONAL MATCH (r)-[*]->(f:{S.LABEL_FILE})
        WITH r, dir_count, count(DISTINCT f) as file_count
        OPTIONAL MATCH (r)-[*]->(s:{S.LABEL_SYMBOL})
        RETURN dir_count, file_count, count(DISTINCT s) as symbol_count
        """
        with self._session() as session:
            result = session.run(query, org_id=org_id, repo=repo)
            record = result.single()
            if not record:
                return {"directories": 0, "files": 0, "symbols": 0}
            return {
                "directories": record["dir_count"],
                "files": record["file_count"],
                "symbols": record["symbol_count"],
            }
