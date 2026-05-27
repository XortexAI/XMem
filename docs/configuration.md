# Configuration

## Vector Store Provider

Cloud Pinecone remains the default:

```env
VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=xmem-index
PINECONE_NAMESPACE=default
PINECONE_DIMENSION=384
```

For local testing, switch only the provider-specific settings:

```env
VECTOR_STORE_PROVIDER=pgvector
PGVECTOR_URL=postgresql://xmem:xmem@localhost:5432/xmem
PGVECTOR_TABLE=xmem_vectors
```

Use the same Postgres instance for app metadata such as users, API keys,
projects, and team members:

```env
APP_STORE_PROVIDER=postgres
APP_POSTGRES_URL=postgresql://xmem:xmem@localhost:5432/xmem
```

If `APP_POSTGRES_URL` is omitted, XMem uses `PGVECTOR_URL`.

The generated local `.env` points Postgres settings at Docker-managed Postgres
on `localhost:15432`. If you set `PGVECTOR_URL` or `APP_POSTGRES_URL` to your
own Postgres URL, `npm run setup` and `npm run dev` skip the Docker Postgres
container and use that database instead. Keep the custom Postgres server running
before starting XMem.

The same rule applies to MongoDB. The generated local `.env` points
`MONGODB_URI` at Docker-managed Mongo on `localhost:27018`. If you set
`MONGODB_URI` to your own local or external MongoDB URI, XMem skips the Docker
Mongo container and uses that database instead.

For throwaway local testing without persistence:

```env
APP_STORE_PROVIDER=memory
```

Legacy admin analytics endpoints still read the old analytics collection shape.
For a no-Mongo local setup, disable analytics collection:

```env
ENABLE_ANALYTICS=false
```

```env
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=.xmem/chroma
```

```env
VECTOR_STORE_PROVIDER=sqlite
SQLITE_VECTOR_PATH=.xmem/xmem_vectors.db
```

Neo4j is configured independently and is used for temporal memory plus the
scanner v1 code graph:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=local-password
```

Run the local database stack with:

```bash
docker compose -f docker/docker-compose.dev.yml up -d
```

Install local vector backends with:

```bash
pip install -e ".[local]"
```

## Embedding Provider

Embedding generation is configured independently from vector storage. The
default preserves the current cloud behavior:

```env
EMBEDDING_PROVIDER=auto
EMBEDDING_MODEL=gemini-embedding-001
```

`auto` uses Bedrock when `EMBEDDING_MODEL` starts with `amazon.`, otherwise it
uses Gemini.

For Ollama local embeddings:

```env
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
PINECONE_DIMENSION=768
```

For an in-process local embedding model through FastEmbed:

```env
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
PINECONE_DIMENSION=384
```

The vector dimension must match the selected embedding model before creating
pgvector, Chroma, SQLite, Pinecone, or Neo4j vector indexes.

## Local Chat Models

Cloud chat models remain the default through `FALLBACK_ORDER`. To use Ollama
for the normal LLM agents instead of Gemini/OpenAI/OpenRouter/etc.:

```env
FALLBACK_ORDER=["ollama"]
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llava:latest
```

Pull the model before starting XMem:

```bash
ollama pull llama3.1:8b
```

You can also mix local and cloud fallback:

```env
FALLBACK_ORDER=["ollama", "openrouter", "gemini"]
```

Agent-specific model overrides still work. For example:

```env
CLASSIFIER_MODEL=llama3.1:8b
RETRIEVAL_MODEL=llama3.1:8b
CODE_MODEL=qwen2.5-coder:7b
```
