<div align="center">
  <h1>XMem</h1>
  <p><strong>The Memory Layer for AI That Never Forgets</strong></p>
  <p>Give every AI agent and LLM interface persistent, cross-platform memory — out of the box.</p>

  <br/>

  <a href="#quickstart"><strong>Quickstart</strong></a> · <a href="#benchmarks">Benchmarks</a> · <a href="#architecture">Architecture</a> · <a href="#sdks">SDKs</a> · <a href="docs/api-reference.md">API Docs</a>

  <br/><br/>

  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/LongMemEval--S-97.1%25-brightgreen" alt="LongMemEval-S"/>
  <img src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Claude%20%7C%20GPT%20%7C%20Bedrock-orange" alt="Multi-LLM"/>
</div>

<br/>

## The Problem

LLMs have **goldfish memory**. Every conversation starts from zero. Switch from ChatGPT to Claude? Context gone. Move from your IDE to a browser? Context gone. Ask about something you discussed last week? Context gone.

This isn't just annoying — it's a fundamental bottleneck for anyone building AI agents, personal assistants, or any application that needs to *know* its user over time.

Companies like Mem0, Zep, and others have raised **tens of millions** trying to solve this. XMem takes a different approach.

## What XMem Does Differently

XMem is a **unified memory system** that sits behind every AI interface you use. It silently captures, classifies, and stores your interactions — and then surfaces the right memories at the right time, across any platform.

What makes it different:

- **Multi-domain memory, not a flat key-value store.** XMem doesn't just dump everything into one vector database. It has specialized agents that understand the *type* of information — personal facts, time-based events, code context, conversation summaries, images — and routes each to purpose-built storage.
- **Judge-before-write architecture.** Every piece of memory passes through a Judge agent that checks it against existing data and decides: add, update, delete, or skip. No duplicates. No stale data. Memory stays clean.
- **Works everywhere.** Chrome extension for ChatGPT/Claude/Gemini/DeepSeek/Perplexity. Python/TypeScript/Go SDKs for your own agents. One memory layer, every interface.

---

## Watch the Demo


https://github.com/user-attachments/assets/8e3349ab-63c9-4046-821d-ca8097948440

https://github.com/user-attachments/assets/60a1d5c3-2efe-4ef1-abb3-e334f5cc5fb7

---

## Benchmarks

We tested XMem against every major memory solution on two established academic benchmarks. XMem outperforms across the board — including full-context baselines with the entire conversation history.

### LongMemEval-S

The industry-standard benchmark for long-term conversational memory. Tests whether a system can recall facts, track preference changes, reason about time, and maintain context across sessions.

| Category | XMem | Zep | Full Context | Mem0 | Memobase | Supermemory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Session User** | **97.1** | 92.9 | 81.4 | 74.2 | 68.5 | 71.8 |
| **Single-Session Assistant** | **96.4** | 80.4 | 94.6 | 72.1 | 65.3 | 69.7 |
| **Single-Session Preference** | **70.0** | 56.7 | 20.0 | 42.3 | 38.1 | 45.6 |
| **Knowledge Update** | **88.4** | 83.3 | 78.2 | 62.8 | 58.4 | 60.1 |
| **Temporal Reasoning** | **76.7** | 62.4 | 45.1 | 48.9 | 42.7 | 51.3 |
| **Multi-Session** | **71.4** | 57.9 | 44.3 | 39.5 | 35.2 | 41.8 |

> XMem scores **97.1%** on single-session user recall — near-perfect memory. On preference tracking, XMem achieves **70.0%** vs the full-context baseline's 20.0%, proving that structured memory massively outperforms brute-force context stuffing.

### LoCoMo

Tests compositional reasoning over memory — can the system connect facts across conversations, reason about temporal relationships, and answer open-ended questions?

| Category | XMem | Zep | Full Context | Mem0 | Memobase | Supermemory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Hop** | **65.6** | 52.3 | 58.1 | 45.7 | 40.2 | 48.9 |
| **Multi-Hop** | **69.2** | 54.8 | 61.5 | 43.1 | 38.6 | 46.2 |
| **Temporal** | **73.0** | 58.4 | 49.7 | 51.2 | 44.8 | 53.5 |
| **Open Domain** | **55.7** | 44.1 | 52.3 | 38.6 | 33.9 | 41.4 |

> On multi-hop reasoning (connecting facts from different conversations), XMem beats the next best system by **14.4 points** and outperforms full-context by 7.7 points. On temporal reasoning, XMem leads by **23.3 points** over full-context — the biggest gap in any category.

### How We Benchmark
- **Evaluation**: LLM-as-Judge using Gemini with structured rubrics
- **Fairness**: All systems tested with identical conversation histories and queries
- **Reproducibility**: Full benchmark suite included in [`benchmarks/`](benchmarks/) and [`LongMemEval/`](LongMemEval/)

---

## Core Features

### Chrome Extension — Memory Where You Already Work
Stop copy-pasting context between AI tools. The XMem Chrome extension brings persistent memory to ChatGPT, Claude, Gemini, DeepSeek, and Perplexity:

- **Live Search & Inject**: As you type a prompt, XMem searches your memory in real-time and shows a floating chip. One click injects relevant context directly into your input — zero friction, no workflow change.
- **Background Auto-Save (Xingest)**: When you hit "Send", XMem asynchronously captures the conversation turn. A background queue extracts facts and summaries without touching your UI.

### Intelligent Multi-Domain Classification
Not all memory is the same, and treating it that way is why other solutions underperform. XMem's **Classifier Agent** analyzes every piece of incoming data and routes it to the right domain:

| Domain | What It Stores | Example | Storage |
| :--- | :--- | :--- | :--- |
| **Profile** | Permanent user facts — identity, preferences, traits | *"I prefer Go over Python for backends"* | Pinecone |
| **Temporal** | Time-anchored events with date resolution | *"I got promoted to Staff Engineer yesterday"* | Neo4j |
| **Summary** | Compressed conversation takeaways | *"Discussed migration from REST to gRPC"* | Pinecone |
| **Code** | Annotations, bugs, explanations tied to symbols | *"This retry logic has a race condition"* | Neo4j + Pinecone |
| **Snippet** | Personal code patterns and utilities | *"Here's my standard error handler in Go"* | Pinecone |
| **Image** | Visual observations and descriptions | *Screenshot of architecture diagram* | Pinecone |

### Agentic Ingestion Pipeline
Every conversation turn flows through a **7-stage LangGraph pipeline**:

```
Input → Classify → Extract (parallel) → Judge → Weave → Store
```

1. **Classifier** routes input to the relevant domains
2. **Domain Agents** (Profiler, Temporal, Summarizer, Code, Snippet, Image) extract structured data in parallel
3. **Judge Agent** compares each extraction against existing memory and decides: `ADD`, `UPDATE`, `DELETE`, or `NOOP`
4. **Weaver** deterministically executes the Judge's decisions across all storage backends

This means XMem doesn't just append — it **maintains** memory. Tell it you switched from Python to Go? The Judge updates your profile. Mention a meeting got rescheduled? The temporal record is corrected, not duplicated.

### Two-Step Agentic Retrieval
When you query XMem, retrieval is not a simple vector search. The LLM itself decides *what* to look up:

1. **Tool Selection**: The retrieval LLM analyzes your query and calls the appropriate search tools — `SearchProfile`, `SearchTemporal`, `SearchSummary`, `SearchSnippet` — potentially multiple in parallel
2. **Synthesis**: Results from all search tools are aggregated and the LLM generates a cited answer with source references

This means asking *"What's my preferred tech stack and when did I last refactor the auth module?"* triggers both a profile lookup and a temporal search — automatically.

### Code Scanner (XIDE)
XMem can index entire Git repositories and build a queryable knowledge graph of your codebase:

- **AST Parsing**: Deterministic parsing (no LLM needed) for Python, TypeScript, and JavaScript. Extracts functions, classes, methods, imports, and call graphs.
- **Incremental Scanning**: Uses `git diff` to only re-process changed files
- **Knowledge Graph**: Builds a Neo4j graph with `IMPORTS`, `CALLS`, and `ANNOTATES` relationships between symbols
- **Chat With Your Code**: Stream-based chat interface that retrieves relevant code context from your indexed repos

### Multi-LLM Orchestration with Fallback
XMem isn't locked to one provider. It orchestrates across **Gemini, Claude, OpenAI, OpenRouter, and Amazon Bedrock** with automatic failover:

```
gemini → claude → openai → bedrock
```

If your primary LLM API rate-limits or goes down, XMem silently falls back to the next provider. Your memory pipeline never breaks. Each agent can even be pinned to a specific model — use Gemini for classification but Claude for retrieval synthesis.

### Multi-Storage Backend
Each memory domain maps to the storage engine best suited for it:

| Engine | Purpose | Used For |
| :--- | :--- | :--- |
| **Pinecone** | High-speed vector similarity search | Profiles, summaries, snippets, code symbols |
| **Neo4j** | Graph traversal + temporal reasoning | Events, code knowledge graph, annotations |
| **MongoDB** | Raw document storage | Scanned code, file metadata, scan state |

---

## Quickstart

### 1. Start the XMem Server

```bash
git clone https://github.com/XortexLabs/xmem.git
cd xmem

# Install (requires Python 3.11+)
pip install -e .

# Configure environment
cp .env.example .env  # Add your API keys

# Start
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Minimum `.env` configuration:**
```ini
# At least one LLM provider
GEMINI_API_KEY=your_key
# Or: CLAUDE_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY

# Vector store (required)
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=xmem-index

# Graph store (required for temporal + code features)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Document store (required for code scanner)
MONGODB_URI=mongodb://localhost:27017
```

### 2. Install the Chrome Extension

```bash
cd xmem-extension
npm install && npm run build
```

Load `dist/` in Chrome via `chrome://extensions` → "Load unpacked". Point it to your server URL.

### 3. Use the SDKs

<a id="sdks"></a>

Every SDK exposes the same three core operations: **`ingest`**, **`retrieve`**, and **`search`**.

#### Python SDK (`client/xmem`)
```python
from xmem import XMemClient

client = XMemClient(api_url="http://localhost:8000")

# Ingest a conversation turn
client.ingest(
    user_query="I switched from Python to Go for all new backend services.",
    agent_response="That's a solid choice for performance-critical services.",
    user_id="dev_42"
)

# Retrieve with LLM-generated answer
result = client.retrieve(
    query="What language do I prefer for backends?",
    user_id="dev_42"
)
print(result.answer)  # "You prefer Go for backend services..."

# Raw search (no LLM generation)
hits = client.search(
    query="backend architecture decisions",
    domains=["profile", "summary"],
    user_id="dev_42"
)
```

#### TypeScript SDK (`@xmem/sdk`)
```typescript
import { XMemClient } from "@xmem/sdk";

const client = new XMemClient("http://localhost:8000");

const hits = await client.search({
  query: "python backend architecture",
  domains: ["code", "summary"],
  user_id: "dev_42"
});
```

#### Go SDK (`github.com/xmem/sdk-go`)
```go
client := xmem.NewClient("http://localhost:8000", "")

answer, _ := client.Retrieve(xmem.RetrieveParams{
    Query:  "Did I ever mention my dog?",
    UserID: "dev_42",
})
```

---

## Architecture

![Architecture](architecture.png)

XMem is built as a **pipeline of specialized AI agents** coordinated by LangGraph, backed by three purpose-built storage engines.

### Ingestion Flow

```
User Input (SDK / Chrome Extension / API)
         │
         ▼
   ┌─────────────┐
   │  Classifier  │ ── Analyzes text, routes to domains
   └──────┬──────┘
          │
    ┌─────┼─────┬──────┬─────────┐
    ▼     ▼     ▼      ▼         ▼
 Profile Temporal Summary Code  Snippet   ◄── Domain agents extract
 Agent   Agent   Agent  Agent   Agent        structured data in parallel
    │     │      │      │        │
    ▼     ▼      ▼      ▼        ▼
   ┌─────────────────────────────────┐
   │          Judge Agent            │ ── Compares against existing memory
   │   (ADD / UPDATE / DELETE / NOOP)│    Prevents duplicates & staleness
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │            Weaver               │ ── Deterministic executor
   │  Pinecone │ Neo4j │ MongoDB    │    Writes to the right backends
   └─────────────────────────────────┘
```

**High-effort mode** automatically splits long inputs into overlapping chunks (~200 tokens) and processes them in parallel, then merges results — ensuring nothing is lost in lengthy conversations.

### Retrieval Flow

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│       Retrieval LLM              │
│  Decides which tools to call:    │
│  SearchProfile, SearchTemporal,  │
│  SearchSummary, SearchSnippet    │
└──────────────┬───────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Pinecone    Neo4j    Pinecone     ◄── Parallel search execution
 (profiles)  (events)  (summaries)
    │          │          │
    └──────────┼──────────┘
               ▼
┌──────────────────────────────────┐
│   Answer Synthesis + Citations   │ ── LLM generates answer with sources
└──────────────────────────────────┘
```

### Resilience

Every LLM call in the pipeline passes through the **Model Registry**. If a provider fails or rate-limits, the request is automatically rerouted to the next provider in the fallback chain. No data loss. No downtime.

---

## Configuration

XMem is highly configurable. Override any agent's model, tune the fallback chain, or adjust quality/speed tradeoffs.

| Setting | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_MODEL_MODE` | `gemini-2.5-flash-lite` | Default LLM for all agents |
| `FALLBACK_ORDER` | `openrouter,gemini,claude,openai` | Provider failover sequence |
| `CLASSIFIER_MODEL` | — | Override model for classifier agent |
| `JUDGE_MODEL` | — | Override model for judge agent |
| `RETRIEVAL_MODEL` | — | Override model for retrieval synthesis |
| `PINECONE_DIMENSION` | `768` | Embedding vector dimension |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Text embedding model |
| `RATE_LIMIT` | `60` | API requests per minute |
| `TEMPERATURE` | `0.4` | LLM generation temperature |

See [docs/configuration.md](docs/configuration.md) for the full reference.

---

## Docker

```bash
docker build -t xmem .
docker run -p 8000:8000 --env-file .env xmem
```

Or with Docker Compose for the full stack (XMem + Neo4j + MongoDB):
```bash
cd docker && docker-compose up
```

---

## Project Structure

```
xmem/
├── src/
│   ├── agents/        # Classifier, Profiler, Temporal, Summarizer,
│   │                  # Judge, Weaver, Code, Snippet, Image agents
│   ├── pipelines/     # LangGraph ingestion & retrieval workflows
│   ├── api/           # FastAPI routes, middleware, rate limiting
│   ├── storage/       # Pinecone vector store client
│   ├── graph/         # Neo4j graph client + schema definitions
│   ├── scanner/       # Git ops, AST parser, incremental indexer
│   ├── models/        # Multi-LLM registry + provider builders
│   ├── schemas/       # Pydantic models for all memory domains
│   ├── config/        # Settings, effort levels, constants
│   └── prompts/       # System prompts for each agent
├── tests/             # Unit, integration, and E2E tests
├── benchmarks/        # LongMemEval + LoCoMo evaluation suite
├── frontend/          # Ingestion/retrieval visualization UI
├── docker/            # Docker Compose for full stack
└── pyproject.toml
```

---

## Contributing

https://github.com/user-attachments/assets/07b93914-4fd2-47ca-b048-c3cd390455e3

We welcome contributions:

```bash
# Setup dev environment
pip install -e ".[dev]"

# Run tests
GEMINI_API_KEY=dummy pytest

# Lint
ruff check src/
```

PRs for new IDE extensions (VSCode, JetBrains), additional language support in the AST parser, and new storage backends are especially welcome.

---

<div align="center">
  <strong>Forget forgetting. Build with XMem.</strong>
  <br/><br/>
  <a href="#quickstart">Get Started</a> · <a href="https://github.com/XortexLabs/xmem/issues">Report Bug</a> · <a href="https://github.com/XortexLabs/xmem/issues">Request Feature</a>
</div>
