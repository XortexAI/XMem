# Changelog

## Unreleased

- Add `forget: bool` flag to the v2 ingest API (`POST /v2/memory/ingest`): memories ingested with `forget=true` are tagged with a TTL (`expires_at`) and automatically excluded from all retrieval results after the TTL elapses (default 30 days, configurable via `MEMORY_FORGET_DEFAULT_TTL_DAYS`). No background sweeper required — enforcement is at read time.
- Add modular Razorpay billing, credit wallets, ledger reservations, and v2 memory workflow metering.
- Add durable Temporal-backed v2 memory and scanner workflow APIs with job status, retry, cancel, and dead-letter endpoints.
- Add modular LoCoMo and BEAM benchmark runners for the Python XMem API.
- Add local XMem setup through `npx create-xmem@latest` and `npm run dev`.
- Add local Docker storage, Chrome extension build patching, diagnostics, verification, and context export/import/sync commands.
