"""
Git operations — clone, pull, diff via subprocess.

Uses the system ``git`` binary for maximum reliability and compatibility.
No PyGithub dependency required for core operations.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("xmem.scanner.git")


class FileChangeType(str, Enum):
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"


@dataclass
class FileChange:
    change_type: FileChangeType
    file_path: str
    old_path: Optional[str] = None  # for renames


@dataclass
class DiffResult:
    """Result of a git diff between two commits."""
    changes: List[FileChange] = field(default_factory=list)
    from_sha: str = ""
    to_sha: str = ""

    @property
    def added(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == FileChangeType.ADDED]

    @property
    def modified(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == FileChangeType.MODIFIED]

    @property
    def deleted(self) -> List[str]:
        return [c.file_path for c in self.changes if c.change_type == FileChangeType.DELETED]

    @property
    def changed_files(self) -> List[str]:
        """All files that need re-processing (added + modified + rename targets)."""
        return self.added + self.modified + [
            c.file_path for c in self.changes if c.change_type == FileChangeType.RENAMED
        ]


def _run_git(args: List[str], cwd: str, timeout: int = 600) -> str:
    """Run a git command and return stdout."""
    cmd = ["git"] + args
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {args[0]} failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def clone_repo(
    repo_url: str,
    local_path: str,
    branch: str = "main",
    depth: Optional[int] = None,
    token: Optional[str] = None,
) -> str:
    """Clone a repository. Returns the HEAD commit SHA.

    If ``token`` is provided, it is injected into the URL for private repos
    (format: ``https://{token}@github.com/org/repo.git``).
    """
    if token and repo_url.startswith("https://"):
        repo_url = repo_url.replace("https://", f"https://{token}@")

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    args = ["clone", "--branch", branch, "--single-branch"]
    if depth:
        args += ["--depth", str(depth)]
    args += [repo_url, local_path]

    _run_git(args, cwd=str(Path(local_path).parent), timeout=1800)
    sha = get_head_sha(local_path)
    logger.info("Cloned %s → %s (HEAD=%s)", repo_url, local_path, sha[:8])
    return sha


def pull_latest(local_path: str, branch: str = "main") -> str:
    """Pull the latest changes. Returns the new HEAD SHA."""
    _run_git(["checkout", branch], cwd=local_path)
    _run_git(["pull", "origin", branch], cwd=local_path, timeout=600)
    sha = get_head_sha(local_path)
    logger.info("Pulled latest in %s (HEAD=%s)", local_path, sha[:8])
    return sha


def clone_or_pull(
    repo_url: str,
    local_path: str,
    branch: str = "main",
    token: Optional[str] = None,
) -> str:
    """Clone if not exists, pull if exists. Returns HEAD SHA."""
    if Path(local_path, ".git").exists():
        return pull_latest(local_path, branch)
    return clone_repo(repo_url, local_path, branch=branch, token=token)


def get_head_sha(local_path: str) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=local_path)


def get_diff(
    local_path: str,
    from_sha: str,
    to_sha: str = "HEAD",
) -> DiffResult:
    """Get the list of changed files between two commits."""
    raw = _run_git(
        ["diff", "--name-status", from_sha, to_sha],
        cwd=local_path,
    )

    result = DiffResult(from_sha=from_sha, to_sha=to_sha)
    if not raw:
        return result

    for line in raw.splitlines():
        parts = line.split("\t")
        if not parts:
            continue

        status = parts[0]
        if status.startswith("R"):
            # Rename: R100\told_path\tnew_path
            old_path = parts[1] if len(parts) > 1 else ""
            new_path = parts[2] if len(parts) > 2 else ""
            result.changes.append(FileChange(
                change_type=FileChangeType.RENAMED,
                file_path=new_path,
                old_path=old_path,
            ))
            result.changes.append(FileChange(
                change_type=FileChangeType.DELETED,
                file_path=old_path,
            ))
        elif status in ("A", "M", "D"):
            result.changes.append(FileChange(
                change_type=FileChangeType(status),
                file_path=parts[1] if len(parts) > 1 else "",
            ))

    logger.info(
        "Diff %s..%s: %d added, %d modified, %d deleted",
        from_sha[:8], to_sha[:8],
        len(result.added), len(result.modified), len(result.deleted),
    )
    return result


def list_all_files(local_path: str) -> List[str]:
    """List all tracked files in the repo."""
    raw = _run_git(["ls-files"], cwd=local_path)
    return [f for f in raw.splitlines() if f.strip()]


# ---------------------------------------------------------------------------
# Supported file extensions (which files to parse)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".php": "php",
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "vendor", "target",
    ".tox", ".mypy_cache", ".pytest_cache", "egg-info",
}

SKIP_PATTERNS = {
    "package-lock.json", "yarn.lock", "poetry.lock",
    "Pipfile.lock", "go.sum",
}


def get_language(file_path: str) -> Optional[str]:
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def should_skip_file(file_path: str) -> bool:
    parts = Path(file_path).parts
    if any(d in SKIP_DIRS for d in parts):
        return True
    if Path(file_path).name in SKIP_PATTERNS:
        return True
    return get_language(file_path) is None
