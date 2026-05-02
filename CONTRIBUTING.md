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

Create a local environment file:

```powershell
copy .env.example .env
```

On macOS or Linux, use:

```bash
cp .env.example .env
```

Fill in only the keys needed for your work. For unit tests that do not call real
services, dummy values are often enough.

## Running Locally

Start the API server:

```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

If you need the full local stack, use the Docker Compose files in `docker/`.

## Testing

Run the test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_deterministic_memory_layer.py
```

The project config enables coverage by default. If your local environment does
not have `pytest-cov`, install the dev dependencies again:

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

## Review Process

Maintainers may ask for changes to scope, tests, naming, or compatibility. Please
respond in the PR thread and push follow-up commits to the same branch.

For security issues, do not open a public issue with exploit details. Contact the
maintainers privately first.
