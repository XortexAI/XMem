import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_scanner_module(monkeypatch):
    api_package = types.ModuleType("src.api")
    api_package.__path__ = []

    dependencies = types.ModuleType("src.api.dependencies")

    async def require_api_key():
        return {"id": "authenticated-user"}

    dependencies.require_api_key = require_api_key

    config = types.ModuleType("src.config")
    config.settings = types.SimpleNamespace(
        mongodb_uri="mongodb://localhost:27017",
        mongodb_database="xmem_test",
    )

    monkeypatch.setitem(sys.modules, "src.api", api_package)
    monkeypatch.setitem(sys.modules, "src.api.dependencies", dependencies)
    monkeypatch.setitem(sys.modules, "src.config", config)

    scanner_path = (
        Path(__file__).resolve().parents[1] / "src" / "api" / "routes" / "scanner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scanner_route_under_test",
        scanner_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeScannerStore:
    def __init__(self):
        self.jobs = {}
        self.upserted_jobs = []
        self.user_repos = []

    def get_scanner_job(self, job_id):
        return self.jobs.get(job_id)

    def upsert_scanner_job(self, **kwargs):
        self.upserted_jobs.append(kwargs)
        self.jobs[kwargs["job_id"]] = kwargs

    def upsert_user_repo_entry(
        self,
        username,
        github_org,
        repo,
        branch,
        last_seen_commit=None,
    ):
        self.user_repos.append({
            "username": username,
            "github_org": github_org,
            "repo": repo,
            "branch": branch,
            "last_seen_commit": last_seen_commit,
        })


def test_start_scan_route_requires_api_key(monkeypatch):
    scanner = _load_scanner_module(monkeypatch)

    scan_route = next(
        route
        for route in scanner.router.routes
        if getattr(route, "endpoint", None) is scanner.start_scan
    )

    assert any(
        dependency.call is scanner.require_api_key
        for dependency in scan_route.dependant.dependencies
    )


def test_scan_request_username_is_optional(monkeypatch):
    scanner = _load_scanner_module(monkeypatch)

    req = scanner.ScanRequest(github_url="https://github.com/acme/payments")

    assert req.username == ""


def test_start_scan_uses_authenticated_user_not_body_username(monkeypatch):
    scanner = _load_scanner_module(monkeypatch)
    store = FakeScannerStore()
    scheduled_tasks = []

    monkeypatch.setattr(scanner, "_get_code_store", lambda: store)
    monkeypatch.setattr(scanner, "_get_branch_tip_sha", lambda *args: "abc123")
    monkeypatch.setattr(scanner, "_can_reuse_index", lambda *args: (False, False))
    monkeypatch.setattr(scanner, "_run_scan_pipeline", lambda *args: ("scan", args))
    monkeypatch.setattr(scanner.asyncio, "create_task", scheduled_tasks.append)

    req = scanner.ScanRequest(
        github_url="https://github.com/acme/payments",
        username="spoofed-user",
        branch="main",
    )
    response = asyncio.run(
        scanner.start_scan(
            req,
            user={"id": "user-1", "username": "alice", "name": "Alice"},
        ),
    )

    payload = json.loads(response.body)

    assert payload["status"] == "ok"
    assert payload["job_id"] == "alice:acme:payments"
    assert store.upserted_jobs[0]["username"] == "alice"
    assert store.upserted_jobs[0]["job_id"] == "alice:acme:payments"
    assert store.user_repos[0]["username"] == "alice"
    assert scheduled_tasks
