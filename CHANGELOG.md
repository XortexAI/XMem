# Changelog

## Unreleased

- Add `forget: bool` flag to `POST /v2/memory/ingest`: memories with `forget=true` get a TTL (`expires_at`) and are excluded from retrieval (`_search_summary` + profile catalog) once expired. Read-time enforcement; no sweeper.
- Add `memory_forget_default_ttl_days` setting (env `MEMORY_FORGET_DEFAULT_TTL_DAYS`, default 30). Known limitation: changing it does not refresh an already-cached (idempotent) forget job's TTL; resolved when TTL becomes a client-supplied field.
- `POST /v2/memory/batch-ingest` rejects `forget=true` with HTTP 400 (per-item forget not yet supported in batch).
- Add modular Razorpay billing, credit wallets, ledger reservations, and v2 memory workflow metering.
- Add durable Temporal-backed v2 memory and scanner workflow APIs with job status, retry, cancel, and dead-letter endpoints.
- Add modular LoCoMo and BEAM benchmark runners for the Python XMem API.
- Add local XMem setup through `npx create-xmem@latest` and `npm run dev`.
- Add local Docker storage, Chrome extension build patching, diagnostics, verification, and context export/import/sync commands.
