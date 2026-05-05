# Contributing to XMem

Thanks for helping improve XMem. This guide explains how to set up the project,
make changes safely, and open a pull request that maintainers can review quickly.

## Before You Start

- Check the issue list and comment on the issue you want to work on.
- For larger changes, share your approach before opening a PR.
- Keep each PR focused on one bug, feature, or documentation task.
- Do not commit secrets, local `.env` files, generated caches, or API keys.

## Project Areas

This repository contains the Python backend, agents, storage integrations, API
routes, scanner, and tests.

Common areas:

- `src/agents/` - classifier, memory extraction, judge, and domain agents.
- `src/pipelines/` - ingestion, retrieval, and weaver workflows.
- `src/api/` - FastAPI application, routes, dependencies, and middleware.
- `src/storage/` - vector store interface and Pinecone implementation.
- `src/graph/` - Neo4j clients and graph schemas.
- `src/scanner/` and `src/scanner_v1/` - repository indexing and AST parsing.
- `src/schemas/` - Pydantic models used across agents and APIs.
- `tests/` - unit and integration tests.
- `docs/` - user and developer documentation.

Related ecosystem pieces such as SDKs, MCP integrations, extensions, and landing
pages may live in separate repositories. Follow the README in that component
when working outside this backend repo.

## Development Setup

Requirements:

- Python 3.11 or newer.
- Access to any external services needed by the area you are testing, such as
  Pinecone, Neo4j, MongoDB, or an LLM provider.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux, use:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

## Linting and Formatting

Run Ruff before opening a PR:

```bash
ruff check src tests
```

Use the existing project style:

- Prefer small, focused functions.
- Keep domain schemas in `src/schemas/`.
- Keep prompt text in `src/prompts/`.
- Keep storage writes inside the Weaver or store clients.
- Avoid broad refactors in bug-fix PRs.

## Coding Standards

- Use Pydantic models for structured data crossing agent, API, or pipeline
  boundaries.
- Keep agent outputs explicit and validated where possible.
- Make Judge logic deterministic when the domain has stable identity keys.
- Keep external service calls behind small wrappers so tests can use fakes.
- Add tests for new behavior, regressions, and failure paths.
- Preserve backward compatibility for existing stored records and API responses.

## Documentation Standards

- Keep setup instructions copy-pasteable.
- Prefer short examples over long explanations.
- Update docs when changing public behavior, configuration, or setup steps.
- Do not include real credentials, private URLs, or personal tokens in docs.

## Pull Request Checklist

Before submitting:

- Rebase or sync with the latest main branch.
- Keep the PR scoped to the issue.
- Add or update tests when behavior changes.
- Run relevant tests and lint checks locally.
- Update docs if setup, API behavior, or configuration changes.
- Describe what changed, why it changed, and how you verified it.

Suggested PR description:

```md
## Summary
- ...

## Testing
- ...

## Notes
- ...
```
