"""
Neo4j Code Graph Schema — labels, relationship types, and property names
for the code knowledge graph.

Node layout
-----------
(:Repository {org_id, repo_name})
(:Directory  {org_id, repo, dir_path})
(:File       {org_id, repo, file_path, language})
(:Symbol     {org_id, repo, file_path, symbol_name, symbol_type, ...})
(:Annotation {annotation_id, org_id, content, annotation_type, ...})

Relationship layout
-------------------
Structural:
  (:Repository)-[:CONTAINS_DIR]->(:Directory)
  (:Directory)-[:CONTAINS_DIR]->(:Directory)
  (:Directory)-[:CONTAINS_FILE]->(:File)
  (:File)-[:DEFINES]->(:Symbol)

Dependency (built by scanner):
  (:Symbol)-[:CALLS {call_count, is_direct}]->(:Symbol)
  (:File)-[:IMPORTS {import_type}]->(:File)
  (:Symbol)-[:INHERITS]->(:Symbol)
  (:Symbol)-[:IMPLEMENTS]->(:Symbol)

Knowledge:
  (:Annotation)-[:ANNOTATES {relevance}]->(:Symbol)
  (:Annotation)-[:ANNOTATES]->(:File)
  (:Annotation)-[:ANNOTATES]->(:Directory)
"""


class CodeGraphSchema:
    # ── Node labels ──────────────────────────────────────────────────
    LABEL_REPOSITORY = "Repository"
    LABEL_DIRECTORY = "Directory"
    LABEL_FILE = "File"
    LABEL_SYMBOL = "Symbol"
    LABEL_ANNOTATION = "Annotation"

    # ── Structural relationships ─────────────────────────────────────
    REL_CONTAINS_DIR = "CONTAINS_DIR"
    REL_CONTAINS_FILE = "CONTAINS_FILE"
    REL_DEFINES = "DEFINES"

    # ── Dependency relationships (scanner-populated) ─────────────────
    REL_CALLS = "CALLS"
    REL_IMPORTS = "IMPORTS"
    REL_INHERITS = "INHERITS"
    REL_IMPLEMENTS = "IMPLEMENTS"

    # ── Knowledge relationships ──────────────────────────────────────
    REL_ANNOTATES = "ANNOTATES"

    # ── Repository properties ────────────────────────────────────────
    PROP_ORG_ID = "org_id"
    PROP_REPO_NAME = "repo_name"

    # ── Directory properties ─────────────────────────────────────────
    PROP_DIR_PATH = "dir_path"
    PROP_DIR_LANGUAGES = "languages"
    PROP_DIR_FILE_COUNT = "file_count"
    PROP_DIR_TOTAL_SYMBOLS = "total_symbols"
    PROP_DIR_SUMMARY = "summary"

    # ── File properties ──────────────────────────────────────────────
    PROP_FILE_PATH = "file_path"
    PROP_LANGUAGE = "language"
    PROP_FILE_SUMMARY = "summary"
    PROP_SYMBOL_COUNT = "symbol_count"
    PROP_SYMBOL_NAMES = "symbol_names"
    PROP_TOTAL_LINES = "total_lines"
    PROP_COMMIT_SHA = "commit_sha"

    # ── Symbol properties ────────────────────────────────────────────
    PROP_SYMBOL_NAME = "symbol_name"
    PROP_SYMBOL_TYPE = "symbol_type"
    PROP_SUMMARY = "summary"
    PROP_SIGNATURE = "signature"
    PROP_DOCSTRING = "docstring"
    PROP_START_LINE = "start_line"
    PROP_END_LINE = "end_line"
    PROP_BRANCH = "branch"
    PROP_PARENT_CLASS = "parent_class"
    PROP_IMPORTS = "imports"
    PROP_SIGNATURE_HASH = "signature_hash"
    PROP_IS_PUBLIC = "is_public"
    PROP_IS_ENTRYPOINT = "is_entrypoint"
    PROP_COMPLEXITY = "complexity_bucket"
    PROP_LINE_COUNT = "line_count"

    # ── Annotation properties ────────────────────────────────────────
    PROP_ANNOTATION_ID = "annotation_id"
    PROP_CONTENT = "content"
    PROP_ANNOTATION_TYPE = "annotation_type"
    PROP_SEVERITY = "severity"
    PROP_AUTHOR_ID = "author_id"
    PROP_AUTHOR_NAME = "author_name"
    PROP_STATUS = "status"
    PROP_SUPERSEDED_BY = "superseded_by"
    PROP_CREATED_AT = "created_at"
    PROP_RESOLVED_AT = "resolved_at"
    PROP_RESOLVED_BY = "resolved_by"

    # ── Relationship properties ──────────────────────────────────────
    PROP_CALL_COUNT = "call_count"
    PROP_IS_DIRECT = "is_direct"
    PROP_IMPORT_TYPE = "import_type"
    PROP_RELEVANCE = "relevance"

    # ── Common ───────────────────────────────────────────────────────
    PROP_INDEXED_AT = "indexed_at"
    PROP_REPO = "repo"
    PROP_EMBEDDING = "embedding"


def setup_code_constraints(driver) -> None:
    """Create uniqueness constraints for code graph nodes.

    Safe to call repeatedly — uses ``IF NOT EXISTS``.
    """
    S = CodeGraphSchema
    queries = [
        (
            f"CREATE CONSTRAINT repo_unique IF NOT EXISTS "
            f"FOR (r:{S.LABEL_REPOSITORY}) "
            f"REQUIRE (r.{S.PROP_ORG_ID}, r.{S.PROP_REPO_NAME}) IS UNIQUE"
        ),
        (
            f"CREATE CONSTRAINT file_unique IF NOT EXISTS "
            f"FOR (f:{S.LABEL_FILE}) "
            f"REQUIRE (f.{S.PROP_ORG_ID}, f.{S.PROP_REPO}, f.{S.PROP_FILE_PATH}) IS UNIQUE"
        ),
        (
            f"CREATE CONSTRAINT dir_unique IF NOT EXISTS "
            f"FOR (d:{S.LABEL_DIRECTORY}) "
            f"REQUIRE (d.{S.PROP_ORG_ID}, d.{S.PROP_REPO}, d.{S.PROP_DIR_PATH}) IS UNIQUE"
        ),
        (
            f"CREATE CONSTRAINT symbol_unique IF NOT EXISTS "
            f"FOR (s:{S.LABEL_SYMBOL}) "
            f"REQUIRE (s.{S.PROP_ORG_ID}, s.{S.PROP_REPO}, s.{S.PROP_FILE_PATH}, "
            f"s.{S.PROP_SYMBOL_NAME}, s.{S.PROP_SIGNATURE_HASH}) IS UNIQUE"
        ),
        (
            f"CREATE CONSTRAINT annotation_id_unique IF NOT EXISTS "
            f"FOR (a:{S.LABEL_ANNOTATION}) "
            f"REQUIRE a.{S.PROP_ANNOTATION_ID} IS UNIQUE"
        ),
    ]

    with driver.session() as session:
        for q in queries:
            session.run(q)
