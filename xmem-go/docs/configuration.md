# Configuration

`xmem-go` loads `xmem-go/.env` and then environment variables. Environment variables win over `.env` values.

Important defaults:

- `API_PORT=8081`
- `VECTOR_STORE_PROVIDER=memory`
- `APP_STORE_PROVIDER=memory` in `.env.example`; the code can use Mongo when set to `mongo`
- `PINECONE_NAMESPACE=xmem-go-dev`
- `API_KEYS='["dev-xmem-go-key"]'`

Core variables mirrored from Python:

- LLM: `GEMINI_API_KEY`, `GEMINI_MODEL`, `CLAUDE_API_KEY`, `CLAUDE_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `FALLBACK_ORDER`, `TEMPERATURE`
- Embeddings/vector: `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `OPENAI_EMBEDDING_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_HOST`, `PINECONE_NAMESPACE`, `PINECONE_DIMENSION`, `PINECONE_METRIC`, `PINECONE_CLOUD`, `PINECONE_REGION`
- Graph/store: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `MONGODB_URI`, `MONGODB_DATABASE`, `APP_STORE_PROVIDER`
- API/auth: `API_HOST`, `API_PORT`, `CORS_ORIGINS`, `RATE_LIMIT`, `API_KEYS`, `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRATION_DAYS`

Scanner, transcript parsing, scrape, and context endpoint configuration is intentionally absent from this Go migration.

## Adapter Selection

- Set `APP_STORE_PROVIDER=mongo` to validate `xmem_...` API keys against MongoDB collections `api_keys` and `users`, and persist durable ingest jobs in `durable_jobs`.
- Set `VECTOR_STORE_PROVIDER=pinecone` to write profile/summary/image/snippet vectors to Pinecone.
- Set `EMBEDDING_PROVIDER=openai` with `OPENAI_API_KEY` to generate OpenAI embeddings. `OPENAI_EMBEDDING_MODEL` defaults to `EMBEDDING_MODEL`.
- Set `NEO4J_PASSWORD` with `NEO4J_URI` and `NEO4J_USERNAME` to enable the Neo4j temporal store.
- In non-production environments, failed Mongo/Pinecone/Neo4j initialization falls back to memory so local development still starts.
- In `ENVIRONMENT=production`, failed Mongo/Pinecone/Neo4j initialization exits instead of silently falling back.
