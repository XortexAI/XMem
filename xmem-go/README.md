# XMem Go

Self-contained Go rewrite of the core XMem memory API. It is intentionally isolated from the production Python service in `../Xmem`, which remains the live production system on port `8000`.

This project migrates only the main memory product:

- `GET /health`
- `POST /v1/memory/ingest`
- `GET /v1/memory/ingest/{job_id}/status`
- `POST /v1/memory/batch-ingest`
- `GET /v1/memory/jobs/{job_id}/status`
- `POST /v1/memory/retrieve`
- `POST /v1/memory/search`

Not migrated:

- scanner/repo indexing
- transcript parsing
- scrape endpoints
- `/context` endpoints
- code retrieval routes
- enterprise/admin/telemetry routes

## Run Locally

```bash
cd xmem-go
cp .env.example .env
go test ./...
go run ./cmd/xmem
```

The server listens on `http://localhost:8081` by default. Use the dev API key from `.env.example`:

```bash
curl http://localhost:8081/health

curl -X POST http://localhost:8081/v1/memory/ingest \
  -H 'Authorization: Bearer dev-xmem-go-key' \
  -H 'Content-Type: application/json' \
  -d '{"user_query":"My name is Alice and I work at XMem.","user_id":"alice"}'

# The ingest response returns a durable job_id immediately. Poll the returned
# status_url until status is "succeeded" before retrieving the newly written memory.

curl -X POST http://localhost:8081/v1/memory/retrieve \
  -H 'Authorization: Bearer dev-xmem-go-key' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where do I work?","user_id":"alice","top_k":5}'
```

## Adapter Status

The server supports real adapters and safe local fallbacks:

- Router: `chi`
- API key/user store: MongoDB when `APP_STORE_PROVIDER=mongo`, memory fallback outside production
- Durable job store: MongoDB when `APP_STORE_PROVIDER=mongo`, memory fallback outside production
- Vector store: Pinecone when `VECTOR_STORE_PROVIDER=pinecone`, memory fallback outside production
- Temporal graph: Neo4j when `NEO4J_PASSWORD` is set, memory fallback outside production
- LLMs: OpenAI, OpenRouter, Gemini, Claude, Ollama via direct HTTP clients, with local deterministic fallback

Dev defaults use `PINECONE_NAMESPACE=xmem-go-dev` to avoid production data. Do not point this service at production stores until parity tests and an explicit cutover are complete.

To enable real services:

```bash
APP_STORE_PROVIDER=mongo
MONGODB_URI=...
MONGODB_DATABASE=xmem

VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
PINECONE_NAMESPACE=xmem-go-dev
# Optional, avoids control-plane host lookup:
PINECONE_HOST=...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=...

NEO4J_URI=<bolt-or-neo4j-uri>
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>

FALLBACK_ORDER='["gemini","claude","openai","openrouter","ollama"]'
GEMINI_API_KEY=...
```

## Production Status

Production remains Python in `../Xmem` until an explicit cutover. Do not edit `../Xmem/.github/workflows/deploy-aws.yml`, and do not restart or replace the `xmem` system service for this Go server.
