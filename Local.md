# Local Setup Guide

Use this guide to run XMem on your own machine.

Choose the view that matches you:

- **Every User**: the quickest path to run XMem locally and use the Chrome extension.
- **Developer View**: details for contributors, debugging, manual startup, ports, and environment setup.

## Every User

This path is for anyone who wants to install XMem locally and start using it.

### What You Need

Install these first:

- Git
- Node.js 20 or newer
- Python 3.11 or newer
- Docker Desktop
- Chrome or another Chromium-based browser
- Ollama, unless you plan to use a cloud LLM key

### 1. Clone XMem

```bash
git clone https://github.com/XortexAI/XMem.git
cd XMem
```

If you already have the repository, open a terminal in the XMem repository root.

### 2. Pick Your LLM Setup

#### Fully Local Setup

Use this if you want XMem to run without cloud LLM calls.

Start Docker Desktop and Ollama, then run:

```bash
npm run setup
```

XMem will use:

- `qwen2.5:1.5b` for chat
- `nomic-embed-text` for embeddings
- Local Postgres with pgvector for vector storage

#### Cloud LLM Setup

Use this if you want XMem to use Gemini, Claude, OpenAI, OpenRouter, or Bedrock.

Create `.env` from the local template.

PowerShell:

```powershell
Copy-Item templates\xmem.env.local .env
notepad .env
```

macOS or Linux:

```bash
cp templates/xmem.env.local .env
${EDITOR:-nano} .env
```

Add at least one real provider key:

```env
GEMINI_API_KEY=
CLAUDE_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

Then run:

```bash
npm run setup
```

When a cloud key is present, XMem uses that provider for LLM calls and uses
FastEmbed locally for embeddings. Ollama is not required in this mode.

### 3. Start XMem

```bash
npm run start
```

The API runs at:

```text
http://localhost:8000
```

You can check the API in your browser:

```text
http://localhost:8000/health
```

The setup is ready when the health response includes:

```json
{
  "pipelines_ready": true
}
```

### 4. Verify Everything

In another terminal, run:

```bash
npm run verify
```

If anything fails, run:

```bash
npm run doctor
```

### 5. Load the Chrome Extension

After setup, the built extension is available at:

```text
repos/xmem-extension/dist
```

Load it in Chrome:

1. Open `chrome://extensions`
2. Enable Developer mode
3. Click Load unpacked
4. Select `repos/xmem-extension/dist`

Use these extension settings:

```text
API URL: http://localhost:8000
API Key: dev-xmem-key
User ID: any stable local user id
```

### Everyday Commands

```bash
npm run setup
npm run start
npm run verify
npm run doctor
```

Use this shortcut on the first run:

```bash
npm run dev
```

`npm run dev` runs setup automatically if needed, then starts the API.

## Developer View

This path is for contributors and anyone who wants to understand or customize
the local environment.

### What Setup Does

`npm run setup` prepares the full local workspace:

- Creates `.env` from `templates/xmem.env.local` if it does not exist
- Detects cloud LLM keys and configures routing
- Starts local Postgres, MongoDB, and Neo4j containers
- Creates a Python virtual environment in `.venv`
- Installs XMem with local and development dependencies
- Clones and builds the Chrome extension in `repos/xmem-extension`
- Pulls Ollama models when local Ollama mode is enabled

### Local Services and Ports

The default local setup uses these ports:

| Service | URL or Port | Purpose |
| --- | --- | --- |
| XMem API | `http://localhost:8000` | Backend API |
| Postgres/pgvector | `localhost:15432` | Vector store and app metadata |
| MongoDB | `localhost:27018` | Document storage |
| Neo4j Browser | `http://localhost:17474` | Graph database UI |
| Neo4j Bolt | `localhost:17687` | Graph database connection |

Neo4j local credentials:

```text
Username: neo4j
Password: local-password
```

### Important Local Environment Values

The local template lives at:

```text
templates/xmem.env.local
```

Common values:

```env
API_PORT=8000
API_KEYS='["dev-xmem-key"]'
VECTOR_STORE_PROVIDER=pgvector
PGVECTOR_URL=postgresql://xmem:xmem@localhost:15432/xmem
APP_STORE_PROVIDER=postgres
APP_POSTGRES_URL=postgresql://xmem:xmem@localhost:15432/xmem
MONGODB_URI=mongodb://localhost:27018
NEO4J_URI=bolt://localhost:17687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=local-password
```

Fully local LLM routing:

```env
FALLBACK_ORDER='["ollama"]'
EMBEDDING_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
PINECONE_DIMENSION=768
```

Cloud LLM routing switches embeddings to FastEmbed:

```env
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
PINECONE_DIMENSION=384
```

Restart the API after changing `.env`.

### Manual Backend Startup

Most users should use `npm run setup` and `npm run start`. For manual backend
startup, run:

```bash
docker compose -f docker-compose.local.yml up -d
python -m venv .venv
```

Activate the virtual environment.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the API:

```bash
pip install -e ".[local,dev]"
python -m uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Developer Commands

```bash
npm run setup
npm run dev
npm run start
npm run verify
npm run doctor
```

Skip Docker startup if the containers are already running:

```bash
npm run start -- --skip-docker
```

Skip Ollama model pulls during setup:

```bash
npm run setup -- --skip-model-pull
```

Skip extension dependency installation and build:

```bash
npm run setup -- --skip-node-install
```

Skip Python dependency installation:

```bash
npm run setup -- --skip-python-install
```

### Troubleshooting

If Docker containers do not start, make sure Docker Desktop is running, then run:

```bash
npm run setup
```

If Ollama is required but missing models, run:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

If the API starts but `/health` stays in `loading`, wait a minute for pipelines
to initialize, then run:

```bash
npm run doctor
```

If the Chrome extension cannot connect, confirm:

- The API is running at `http://localhost:8000`
- The extension API URL is `http://localhost:8000`
- The extension API key is `dev-xmem-key`
- `/health` reports `pipelines_ready: true`

If you changed `.env`, stop and restart the API.
