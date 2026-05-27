from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "xmem-context-v1"


def normalize_user_id(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.@-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:256]


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise SystemExit(f"XMem .env not found at {path}. Run npm run setup first.")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def project_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    xmem_dir = root
    return root, xmem_dir, xmem_dir / ".env"


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("format") != FORMAT:
        raise SystemExit(f"Unsupported context bundle format: {bundle.get('format')!r}")
    return bundle


def connect_postgres(env: dict[str, str]):
    import psycopg

    return psycopg.connect(env.get("PGVECTOR_URL") or "postgresql://xmem:xmem@localhost:15432/xmem")


def pgvector_table_identifier(env: dict[str, str]):
    from psycopg import sql

    table = env.get("PGVECTOR_TABLE") or "xmem_vectors"
    parts = table.split(".")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise SystemExit(f"Invalid PGVECTOR_TABLE name: {table}")
    return sql.Identifier(*parts)


def user_filter_values(user_id: str | None) -> list[str]:
    if not user_id:
        return []
    values = [user_id.strip(), normalize_user_id(user_id)]
    return sorted({value for value in values if value})


def export_pgvector(env: dict[str, str], user_id: str | None) -> list[dict[str, Any]]:
    from psycopg import sql

    filters = user_filter_values(user_id)
    params: list[Any] = []
    where_sql = sql.SQL("")
    if filters:
        where_sql = sql.SQL("WHERE metadata->>'user_id' = ANY(%s)")
        params.append(filters)

    with connect_postgres(env) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                SELECT namespace, id, content, embedding::text AS embedding,
                       metadata, created_at, updated_at
                FROM {table}
                {where}
                ORDER BY created_at, namespace, id
                """
                ).format(table=pgvector_table_identifier(env), where=where_sql),
                params,
            )
            rows = cur.fetchall()

    return [
        {
            "namespace": row[0],
            "id": row[1],
            "content": row[2],
            "embedding": row[3],
            "metadata": row[4] or {},
            "created_at": row[5],
            "updated_at": row[6],
        }
        for row in rows
    ]


def export_neo4j_events(env: dict[str, str], user_id: str | None) -> list[dict[str, Any]]:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return []

    filters = user_filter_values(user_id)
    where = "WHERE size($users) = 0 OR u.user_id IN $users"
    query = f"""
    MATCH (u:User)-[r:HAS_EVENT]->(d:Date)
    {where}
    RETURN u.user_id AS user_id, d.date AS date, properties(r) AS properties
    ORDER BY user_id, date, properties(r).event_name
    """

    driver = GraphDatabase.driver(
        env.get("NEO4J_URI") or "bolt://localhost:17687",
        auth=(env.get("NEO4J_USERNAME") or "neo4j", env.get("NEO4J_PASSWORD") or "local-password"),
    )
    try:
        with driver.session() as session:
            rel_types = [record["relationshipType"] for record in session.run("CALL db.relationshipTypes()")]
            if "HAS_EVENT" not in rel_types:
                return []
            records = session.run(query, users=filters)
            return [
                {
                    "user_id": record["user_id"],
                    "date": record["date"],
                    "properties": dict(record["properties"] or {}),
                }
                for record in records
            ]
    finally:
        driver.close()


def export_context(args: argparse.Namespace) -> None:
    root, _, env_path = project_paths()
    env = read_env(env_path)
    vectors = export_pgvector(env, args.user_id)
    events = export_neo4j_events(env, args.user_id)
    users = sorted(
        {
            str(row.get("metadata", {}).get("user_id") or "")
            for row in vectors
            if row.get("metadata", {}).get("user_id")
        }
        | {str(event.get("user_id") or "") for event in events if event.get("user_id")}
    )

    out = Path(args.out) if args.out else root / "exports" / f"xmem-context-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": FORMAT,
        "exported_at": datetime.now(timezone.utc),
        "source": {
            "workspace": str(root),
            "vector_store": "pgvector",
            "graph_store": "neo4j",
        },
        "filter": {"user_id": args.user_id or None},
        "users": users,
        "stores": {
            "pgvector": {
                "table": env.get("PGVECTOR_TABLE") or "xmem_vectors",
                "rows": vectors,
            },
            "neo4j": {
                "temporal_events": events,
            },
        },
    }
    out.write_text(json.dumps(bundle, indent=2, default=json_default), encoding="utf-8")
    print(f"[xmem] Exported {len(vectors)} vector memories and {len(events)} temporal events.")
    print(f"[xmem] Context bundle: {out}")


def import_pgvector(env: dict[str, str], rows: list[dict[str, Any]], user_id: str | None) -> int:
    if not rows:
        return 0
    from psycopg import sql
    from psycopg.types.json import Jsonb

    with connect_postgres(env) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for row in rows:
                metadata = dict(row.get("metadata") or {})
                if user_id:
                    metadata["user_id"] = normalize_user_id(user_id)
                cur.execute(
                    sql.SQL(
                        """
                    INSERT INTO {table}(namespace, id, content, embedding, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s::vector, %s, COALESCE(%s::timestamptz, now()), COALESCE(%s::timestamptz, now()))
                    ON CONFLICT(namespace, id) DO UPDATE SET
                        content = excluded.content,
                        embedding = excluded.embedding,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """
                    ).format(table=pgvector_table_identifier(env)),
                    (
                        row["namespace"],
                        row["id"],
                        row["content"],
                        row["embedding"],
                        Jsonb(metadata),
                        row.get("created_at"),
                        row.get("updated_at"),
                    ),
                )
    return len(rows)


def import_neo4j_events(env: dict[str, str], events: list[dict[str, Any]], user_id: str | None) -> int:
    if not events:
        return 0
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        env.get("NEO4J_URI") or "bolt://localhost:17687",
        auth=(env.get("NEO4J_USERNAME") or "neo4j", env.get("NEO4J_PASSWORD") or "local-password"),
    )
    query = """
    MERGE (u:User {user_id: $user_id})
    MERGE (d:Date {date: $date})
    MERGE (u)-[r:HAS_EVENT {event_name: $event_name}]->(d)
    SET r += $properties
    """
    try:
        with driver.session() as session:
            for event in events:
                props = dict(event.get("properties") or {})
                target_user = normalize_user_id(user_id) if user_id else event.get("user_id")
                session.run(
                    query,
                    user_id=target_user,
                    date=event.get("date"),
                    event_name=props.get("event_name") or "",
                    properties=props,
                )
    finally:
        driver.close()
    return len(events)


def import_context(args: argparse.Namespace) -> None:
    _, _, env_path = project_paths()
    env = read_env(env_path)
    bundle = load_bundle(Path(args.file))
    rows = bundle.get("stores", {}).get("pgvector", {}).get("rows", [])
    events = bundle.get("stores", {}).get("neo4j", {}).get("temporal_events", [])
    row_count = import_pgvector(env, rows, args.user_id)
    event_count = import_neo4j_events(env, events, args.user_id)
    print(f"[xmem] Imported {row_count} vector memories and {event_count} temporal events.")


def api_post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Remote sync failed: HTTP {exc.code}\n{body}") from exc


def sync_context(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.file))
    rows = bundle.get("stores", {}).get("pgvector", {}).get("rows", [])
    if not rows:
        print("[xmem] No vector memories found in bundle.")
        return

    server = args.server.rstrip("/")
    api_key = args.api_key
    if not api_key:
        raise SystemExit("Missing --api-key for remote sync.")

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        target_user = normalize_user_id(args.user_id) if args.user_id else metadata.get("user_id") or "xmem-imported-user"
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        key = (target_user, content)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "user_query": content,
                "agent_response": "Imported from an XMem local context bundle.",
                "user_id": target_user,
                "effort_level": "low",
            }
        )

    if args.dry_run:
        print(f"[xmem] Dry run: would sync {len(items)} memories to {server}.")
        return

    synced = 0
    for index in range(0, len(items), args.batch_size):
        batch = items[index : index + args.batch_size]
        response = api_post_json(
            f"{server}/v1/memory/batch-ingest",
            api_key,
            {"items": batch},
            args.timeout,
        )
        if response.get("status") != "ok":
            raise SystemExit(f"Remote sync failed: {response}")
        synced += len(batch)
        print(f"[xmem] Synced {synced}/{len(items)} memories")

    print(f"[xmem] Remote sync complete: {synced} memories sent to {server}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="XMem local context export/import/sync")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export local XMem context to a JSON bundle")
    export_parser.add_argument("--user-id", default="", help="Optional user id/name filter")
    export_parser.add_argument("--out", default="", help="Output JSON path")
    export_parser.set_defaults(func=export_context)

    import_parser = subparsers.add_parser("import", help="Import a JSON context bundle into local XMem storage")
    import_parser.add_argument("--file", required=True, help="Context bundle JSON path")
    import_parser.add_argument("--user-id", default="", help="Optional target user id override")
    import_parser.set_defaults(func=import_context)

    sync_parser = subparsers.add_parser("sync", help="Send a context bundle to a remote XMem API")
    sync_parser.add_argument("--file", required=True, help="Context bundle JSON path")
    sync_parser.add_argument("--server", required=True, help="Remote XMem server URL")
    sync_parser.add_argument("--api-key", required=True, help="Remote XMem API key")
    sync_parser.add_argument("--user-id", default="", help="Optional target user id override")
    sync_parser.add_argument("--batch-size", type=int, default=20, help="Batch size for remote ingest")
    sync_parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout seconds per batch")
    sync_parser.add_argument("--dry-run", action="store_true", help="Show what would be synced without sending")
    sync_parser.set_defaults(func=sync_context)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
