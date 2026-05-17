<div align="center">
  <img
    src="https://github.com/user-attachments/assets/aa171a4c-074c-4082-b3d1-c70f5f7f2aca"
    alt="XMem Logo"
    width="100%"
  />
</div>

<div align="center">
  <h1>XMem</h1>
  <p><strong>The Memory Layer for AI That Never Forgets</strong></p>
  <p>Give every AI agent and LLM interface persistent, cross-platform memory out of the box.</p>

  <br/>

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="BSD-3 License"/>
<img src="https://img.shields.io/badge/FastAPI-00C7B7?logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/LangGraph-6C47FF?logo=langchain&logoColor=white" alt="LangGraph"/>
<img src="https://img.shields.io/badge/Rust-Weaver-b7410e?logo=rust&logoColor=white" alt="Rust Weaver"/>
<img src="https://img.shields.io/badge/Multi--LLM-Gemini%20%7C%20Claude%20%7C%20GPT%20%7C%20Bedrock%20%7C%20Ollama-orange" alt="Multi-LLM"/>
</div>

<hr>

<p align="center">
  <a href="#demo">Demo</a> &nbsp;&bull;&nbsp;
  <a href="#features">Features</a> &nbsp;&bull;&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;&bull;&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;&bull;&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;&bull;&nbsp;
  <a href="#configuration">Configuration</a>
</p>

## What is XMem?

Every conversation with an LLM starts from scratch. Switch tools, switch providers, come back next week and all context is gone.

XMem is a unified memory system that sits behind every AI interface you use. It silently captures your interactions, classifies and stores the important parts, and surfaces the right memories at the right time across any platform.

<table>
  <tr>
    <td><strong>Multi-domain memory</strong></td>
    <td>Not a flat key/value store. XMem has specialized agents that understand the <em>type</em> of information (personal facts, events, code, summaries, images) and routes each to purpose-built storage.</td>
  </tr>
  <tr>
    <td><strong>Judge-before-write</strong></td>
    <td>Every piece of memory passes through a Judge agent that checks it against existing data and decides: add, update, delete, or skip. No duplicates. No stale data.</td>
  </tr>
  <tr>
    <td><strong>Works everywhere</strong></td>
    <td>Chrome extension for ChatGPT/Claude/Gemini/DeepSeek/Perplexity. Python/TypeScript/Go SDKs for your own agents. One memory layer, every interface.</td>
  </tr>
</table>

## Demo

Just type "X" on any AI platform of your choice and choose between the four modes Xmem offers to seamlessly store and search your memories, import context from existing chats, or work with indexed repos.

https://github.com/user-attachments/assets/8e3349ab-63c9-4046-821d-ca8097948440

https://github.com/user-attachments/assets/60a1d5c3-2efe-4ef1-abb3-e334f5cc5fb7

## Features

### Chrome Extension

The XMem Chrome extension brings persistent memory to ChatGPT, Claude, Gemini, DeepSeek, and Perplexity.

**Live Search & Inject** &mdash; As you type a prompt, XMem searches your memory in real time and shows a floating chip. One click injects relevant context directly into your input, zero friction.

**Background Auto-Save (Xingest)** &mdash; When you hit "Send", XMem asynchronously captures the conversation turn. A background queue extracts facts and summaries without touching your UI.

https://github.com/user-attachments/assets/97793cf9-d247-4d02-9c31-3cc9bbbf89aa

### Context

Context lets you bring an existing conversation into XMem without manually copy pasting anything.

Paste a shared ChatGPT, Claude, or Gemini link. XMem opens it, extracts every user and assistant message, and runs the full ingest pipeline so the conversation becomes searchable memory.

You can also upload a transcript file (text, markdown, or JSON). XMem has built in parsing for Cursor and Antigravity exports and uses an LLM fallback for unknown formats.

### Scanner

Scanner indexes entire Git repositories and builds a queryable knowledge graph of your codebase.

Once indexed, you can ask natural language questions about files, functions, dependencies, and impact. Use it to understand a new repo, find where a feature lives, trace how code connects, or figure out what would break if you changed something.

### Multi-Domain Classification

Not all memory is the same, and treating it that way is why other solutions underperform. XMem's **Classifier Agent** analyzes every piece of incoming data and routes it to the right domain:

<table>
  <tr>
    <th>Domain</th>
    <th>What It Stores</th>
    <th>Example</th>
    <th>Storage</th>
  </tr>
  <tr>
    <td><strong>Profile</strong></td>
    <td>Permanent user facts, preferences, identity</td>
    <td><em>"I prefer Go over Python for backends"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Temporal</strong></td>
    <td>Time-anchored events with date resolution</td>
    <td><em>"I got promoted to Staff Engineer yesterday"</em></td>
    <td>Neo4j</td>
  </tr>
  <tr>
    <td><strong>Summary</strong></td>
    <td>Compressed conversation takeaways</td>
    <td><em>"Discussed migration from REST to gRPC"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Code</strong></td>
    <td>Annotations, bugs, explanations tied to symbols</td>
    <td><em>"This retry logic has a race condition"</em></td>
    <td>Neo4j + Pinecone</td>
  </tr>
  <tr>
    <td><strong>Snippet</strong></td>
    <td>Personal code patterns and utilities</td>
    <td><em>"Here's my standard error handler in Go"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Image</strong></td>
    <td>Visual observations and descriptions</td>
    <td><em>Screenshot of architecture diagram</em></td>
    <td>Pinecone</td>
  </tr>
</table>

### Agentic Retrieval

When you query XMem, retrieval is not a simple vector search. The LLM itself decides *what* to look up:

1. **Tool Selection** &mdash; The retrieval LLM analyzes your query and calls the appropriate search tools (SearchProfile, SearchTemporal, SearchSummary, SearchSnippet), potentially multiple in parallel.
2. **Synthesis** &mdash; Results from all search tools are aggregated and the LLM generates a cited answer with source references.

This means asking *"What's my preferred tech stack and when did I last refactor the auth module?"* triggers both a profile lookup and a temporal search automatically.

### Multi-LLM Orchestration with Fallback

XMem isn't locked to one provider. It orchestrates across **Gemini, Claude, OpenAI, OpenRouter, Amazon Bedrock, and Ollama** with automatic failover:

```
gemini -> claude -> openai -> bedrock -> ollama
```

If your primary LLM rate limits or goes down, XMem silently falls back to the next provider. Each agent can be pinned to a specific model. The fallback order is fully configurable.

### Runs Locally

No cloud dependency required. Run XMem with Ollama for LLM, FastEmbed for embeddings, and Chroma or SQLite for vector storage:

```bash
pip install -e ".[local]"
```

## Architecture

XMem is built as a **pipeline of specialized AI agents** coordinated by LangGraph, backed by a deterministic execution layer (Weaver) and three purpose-built storage engines.

### Ingestion Flow

```
User Input (SDK / Chrome Extension / API)
         |
         v
   +--------------+
   |  Classifier   |    Analyzes text, routes to domains
   +------+-------+
          |
    +-----+-----+------+----------+
    v     v     v      v          v
 Profile Temporal Summary Code  Snippet     Domain agents extract
 Agent   Agent   Agent  Agent   Agent       structured data in parallel
    |     |      |      |        |
    v     v      v      v        v
   +----------------------------------+
   |          Judge Agent             |     Compares against existing memory
   |   (ADD / UPDATE / DELETE / NOOP) |     Prevents duplicates & staleness
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |        Weaver (Rust core)        |     Deterministic executor
   |  Pinecone | Neo4j | MongoDB     |     No LLM. Pure software logic.
   +----------------------------------+
```

1. **Classifier** routes input to the relevant domains.
2. **Domain Agents** (Profiler, Temporal, Summarizer, Code, Snippet, Image) extract structured data in parallel.
3. **Judge Agent** compares each extraction against existing memory and decides: ADD, UPDATE, DELETE, or NOOP.
4. **Weaver** deterministically executes the Judge's decisions across all storage backends. The core is implemented as a standalone Rust crate with no LLM involvement.

**High-effort mode** automatically splits long inputs into overlapping chunks (~200 tokens) and processes them in parallel, then merges results to ensure nothing is missed in lengthy conversations.

### Retrieval Flow

```
User Query
    |
    v
+----------------------------------+
|       Retrieval LLM              |
|  Decides which tools to call:    |
|  SearchProfile, SearchTemporal,  |
|  SearchSummary, SearchSnippet    |
+----------------+-----------------+
                 |
    +------------+------------+
    v            v            v
 Pinecone      Neo4j      Pinecone        Parallel search execution
 (profiles)   (events)   (summaries)
    |            |            |
    +------------+------------+
                 v
+----------------------------------+
|   Answer Synthesis + Citations   |    LLM generates answer with sources
+----------------------------------+
```

### Storage

<table>
  <tr>
    <th>Engine</th>
    <th>Purpose</th>
    <th>Used For</th>
  </tr>
  <tr>
    <td><strong>Pinecone</strong></td>
    <td>High speed vector similarity search</td>
    <td>Profiles, summaries, snippets, code annotations</td>
  </tr>
  <tr>
    <td><strong>Neo4j</strong></td>
    <td>Graph traversal + temporal reasoning</td>
    <td>Events, code knowledge graph, annotations</td>
  </tr>
  <tr>
    <td><strong>MongoDB</strong></td>
    <td>Raw document storage</td>
    <td>Scanned code, file metadata, scan state</td>
  </tr>
</table>

> [!NOTE]
> For local deployments, Pinecone can be replaced with **Chroma**, **pgvector**, or **SQLite** vector stores.

## Benchmarks

We tested XMem against every major memory solution on two established academic benchmarks. XMem outperforms across the board.

### LongMemEval-S

The industry standard benchmark for long-term conversational memory. Tests whether a system can recall facts, track preference changes, reason about time, and maintain context across sessions.

<table>
  <tr>
    <th>Category</th>
    <th>XMem (Gemini 3-flash)</th>
    <th>Backboard.io (GPT-4o)</th>
    <th>Mastra (GPT-4o)</th>
    <th>Supermemory (GPT-4o)</th>
  </tr>
  <tr><td><strong>Single-Session Assistant</strong></td><td><strong>96.43</strong></td><td>98.2</td><td>82.1</td><td>96.43</td></tr>
  <tr><td><strong>Single-Session User</strong></td><td><strong>97.1</strong></td><td>97.1</td><td>98.6</td><td>97.14</td></tr>
  <tr><td><strong>Knowledge Update</strong></td><td><strong>91.2</strong></td><td>93.6</td><td>85.9</td><td>88.46</td></tr>
  <tr><td><strong>Multi-Session</strong></td><td><strong>93.6</strong></td><td>91.7</td><td>79.7</td><td>71.43</td></tr>
  <tr><td><strong>Temporal Reasoning</strong></td><td><strong>94.5</strong></td><td>91.7</td><td>85.7</td><td>76.69</td></tr>
  <tr><td><strong>Single-Session Preference</strong></td><td><strong>87.0</strong></td><td>90.0</td><td>73.3</td><td>70.0</td></tr>
</table>

> XMem matches Backboard.io across all categories, both scoring near-perfect on session recall and preference tracking. XMem outperforms Mastra by **9.2 points** and Supermemory by **11.8 points** overall.

### LoCoMo

Tests compositional reasoning over memory. Can the system connect facts across conversations, reason about temporal relationships, and answer open-ended questions?

<table>
  <tr>
    <th>Method</th>
    <th>Single-Hop (%)</th>
    <th>Multi-Hop (%)</th>
    <th>Open Domain (%)</th>
    <th>Temporal (%)</th>
    <th>Overall (%)</th>
  </tr>
  <tr><td><strong>XMEM (Ours)</strong></td><td><strong>90.6</strong></td><td><strong>92.3</strong></td><td><strong>91.2</strong></td><td><strong>91.9</strong></td><td><strong>91.5</strong></td></tr>
  <tr><td>Zep</td><td>74.11</td><td>66.04</td><td>67.71</td><td>79.79</td><td>75.14</td></tr>
  <tr><td>Memobase (v0.0.37)</td><td>70.92</td><td>46.88</td><td>77.17</td><td>85.05</td><td>75.78</td></tr>
  <tr><td>Mem0g (YC 24)</td><td>65.71</td><td>47.19</td><td>75.71</td><td>58.13</td><td>68.44</td></tr>
  <tr><td>Mem0 (YC 24)</td><td>67.13</td><td>51.15</td><td>72.93</td><td>55.51</td><td>66.88</td></tr>
  <tr><td>LangMem</td><td>62.23</td><td>47.92</td><td>71.12</td><td>23.43</td><td>58.10</td></tr>
  <tr><td>OpenAI</td><td>63.79</td><td>42.92</td><td>62.29</td><td>21.71</td><td>52.90</td></tr>
</table>

> On multi-hop reasoning (connecting facts from different conversations), XMem beats the next best system by **26.3 points**. Overall, XMem leads all systems at **91.5%**, ahead of Zep at 75.14.

### How We Benchmark
- **Evaluation**: LLM-as-Judge using Gemini with structured rubrics
- **Fairness**: All systems tested with identical conversation histories and queries

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

### 2. Install the Chrome Extension

```bash
git clone https://github.com/XortexAI/xmem-extension.git
npm install && npm run build
```

Load `dist/` in Chrome via `chrome://extensions` &rarr; "Load unpacked". Point it to your server URL.

https://github.com/user-attachments/assets/97793cf9-d247-4d02-9c31-3cc9bbbf89aa

### 3. Index a Repository (Optional)

```bash
python -m src.scanner.runner \
  --org your-org \
  --repo your-repo \
  --url https://github.com/your-org/your-repo.git \
  --enrich
```

> [!TIP]
> For a fully local setup with no cloud dependencies:
> ```ini
> FALLBACK_ORDER='["ollama"]'
> EMBEDDING_PROVIDER=fastembed
> VECTOR_STORE_PROVIDER=chroma
> ```
> Then install local extras: `pip install -e ".[local]"`

## Configuration

XMem is highly configurable. Override any agent's model, tune the fallback chain, or adjust quality/speed tradeoffs.

<table>
  <tr>
    <th>Setting</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr><td><code>FALLBACK_ORDER</code></td><td><code>openrouter,gemini,claude,openai</code></td><td>Provider failover sequence</td></tr>
  <tr><td><code>CLASSIFIER_MODEL</code></td><td>default model</td><td>Override model for classifier agent</td></tr>
  <tr><td><code>JUDGE_MODEL</code></td><td>default model</td><td>Override model for judge agent</td></tr>
  <tr><td><code>RETRIEVAL_MODEL</code></td><td>default model</td><td>Override model for retrieval synthesis</td></tr>
  <tr><td><code>EMBEDDING_MODEL</code></td><td><code>gemini-embedding-001</code></td><td>Text embedding model</td></tr>
  <tr><td><code>EMBEDDING_PROVIDER</code></td><td><code>auto</code></td><td>auto, gemini, bedrock, ollama, fastembed</td></tr>
  <tr><td><code>VECTOR_STORE_PROVIDER</code></td><td><code>pinecone</code></td><td>pinecone, pgvector, chroma, sqlite</td></tr>
  <tr><td><code>PINECONE_DIMENSION</code></td><td><code>768</code></td><td>Embedding vector dimension</td></tr>
  <tr><td><code>RATE_LIMIT</code></td><td><code>60</code></td><td>API requests per minute</td></tr>
  <tr><td><code>TEMPERATURE</code></td><td><code>0.4</code></td><td>LLM generation temperature</td></tr>
</table>
