# Xmem Migration Plan: v0 → Production

Comprehensive migration plan from `Xmem_v0` prototype to production-ready `Xmem` with modular architecture.

## User Review Required

> [!IMPORTANT]
> **Architecture Decisions (CONFIRMED):**
>
> - **Vector Store**: Pinecone (primary)
> - **Graph Database**: Neo4j (required for temporal/relational memory)
> - **MongoDB**: Session storage + User profiles
> - **LLM Providers**: Multi-provider support (Gemini, Claude, OpenAI)

> [!WARNING]
> **Breaking Changes:**
>
> - Complete restructure from `packages/server/*` to `src/*`
> - All import paths will change
> - API endpoints may need versioning (`/v1/`, `/v2/`)
> - Environment variables will be centralized in `pydantic-settings`

## Proposed Changes

### Phase 1: Project Foundation 🏗️

#### 1.1 Dependency Management

**[NEW]** [pyproject.toml](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/pyproject.toml)

Create modern Python project configuration with:

- **Build system**: `hatchling` or `poetry-core`
- **Dependencies**: Migrate from [requirements.txt](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/requirements.txt)
- **Dev dependencies**: `pytest`, `mypy`, `ruff`, `black`
- **Optional dependencies**: Groups for different vector stores

**Tasks:**

- [✓] Define project metadata (name, version, description, authors)
- [✓] Add core dependencies (FastAPI, Pydantic, LangChain, LangGraph)
- [✓] Add LLM dependencies (`langchain-google-genai`, `langchain-anthropic`, `langchain-openai`)
- [✓] Add vector store dependencies (Pinecone, sentence-transformers)
- [✓] Add database dependencies (pymongo, neo4j)
- [✓] Add dev dependencies (pytest, mypy, ruff, black, pre-commit)
- [✓] Configure build system and entry points

---

#### 1.2 Configuration Management

**[NEW]** [src/config/settings.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/config/settings.py)

Centralized configuration using `pydantic-settings`:

**Tasks:**

- [✓] Create `BaseSettings` class with environment variable loading
- [✓] Define LLM configuration (API keys for Gemini/Claude/OpenAI, model names, temperature, fallback order)
- [✓] Define vector store configuration (Pinecone API key, index name, namespace)
- [✓] Define database configuration (MongoDB URI for sessions/profiles, Neo4j URI/credentials)
- [✓] Define API configuration (CORS origins, rate limits)
- [N/A] Define logging configuration (level, format, handlers) - Skipped, using Opik for observability
- [✓] Add validation for required vs optional settings
- [✓] Create [.env.example](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/.env.example) with all configuration options (Updated with Opik)

**[NEW]** [src/config/**init**.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/config/__init__.py)

**Tasks:**

- [✓] Export singleton `settings` instance
- [✓] Add helper functions for config access

---

#### 1.3 Logging & Observability

**[NEW]** [src/utils/logger.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/utils/logger.py)

**Tasks:**

- [ ] Create structured logging setup (JSON logs for production)
- [ ] Add correlation ID middleware for request tracing
- [ ] Configure log levels per module
- [ ] Add file rotation handlers
- [ ] Integrate with LangSmith/LangFuse (optional)

---

### Phase 2: Schema & Type Definitions 📋

#### 2.1 Core Schemas

**Source:** `packages/server/type_defs/*` → **Target:** `src/schemas/*`

**[NEW]** [src/schemas/memory.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/memory.py)

**Tasks:**

- [ ] Migrate [memory_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/memory_types.py) → `MemorySchema`, `MemoryResponse`
- [ ] Add strict validation (field constraints, regex patterns)
- [ ] Add JSON schema examples for documentation

**[NEW]** [src/schemas/session.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/session.py)

**Tasks:**

- [ ] Migrate [session_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/session_types.py) → `SessionSchema`, `SessionCreate`, `SessionUpdate`
- [ ] Add session state enums

**[NEW]** [src/schemas/message.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/message.py)

**Tasks:**

- [ ] Migrate [message_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/message_types.py) → `MessageSchema`, `MessageCreate`
- [ ] Add message role enums (user, assistant, system)

**[NEW]** [src/schemas/user_profile.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/user_profile.py)

**Tasks:**

- [ ] Migrate [user_profile.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/user_profile.py) → `UserProfile`, `ProfileUpdate`
- [ ] Add profile topic schemas

**[NEW]** [src/schemas/agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/agent.py)

**Tasks:**

- [ ] Migrate [agent_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/agent_types.py), [classification_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/classification_types.py), [judge_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/judge_types.py), [weaver_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/weaver_types.py)
- [ ] Create unified agent request/response schemas

**[NEW]** [src/schemas/router.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/router.py)

**Tasks:**

- [ ] Migrate [router_types.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/router_types.py) → routing decision schemas

**[NEW]** [src/schemas/response.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/schemas/response.py)

**Tasks:**

- [ ] Migrate [response.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/type_defs/response.py) → standardized API response wrappers
- [ ] Add error response schemas

---

### Phase 3: Storage Layer 💾

#### 3.1 Vector Store Abstraction

**Source:** `packages/server/core/vectorstore/*` → **Target:** `src/storage/vectorstore/*`

**[NEW]** [src/storage/vectorstore/base.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/storage/vectorstore/base.py)

**Tasks:**

- [ ] Migrate [base.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/vectorstore/base.py) → `VectorStoreProtocol` (abstract interface)
- [ ] Define methods: `add`, `search`, `delete`, `update`, `get_stats`
- [ ] Add type hints with generics

**[NEW]** [src/storage/vectorstore/pinecone.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/storage/vectorstore/pinecone.py)

**Tasks:**

- [ ] Migrate [pinecone_store.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/vectorstore/pinecone_store.py) → `PineconeVectorStore`
- [ ] Implement `VectorStoreProtocol`
- [ ] Add connection pooling and retry logic
- [ ] Add batch operations support

**[NEW]** [src/storage/vectorstore/factory.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/storage/vectorstore/factory.py)

**Tasks:**

- [ ] Create factory pattern for Pinecone initialization
- [ ] Add configuration-based namespace selection
- [ ] Add connection health checks
- [ ] Add retry logic and error handling

---

#### 3.2 Database Connections

**Source:** `packages/server/db/*` → **Target:** `src/storage/database/*`

**[NEW]** [src/storage/database/mongodb.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/storage/database/mongodb.py)

**Tasks:**

- [ ] Migrate [connection.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/db/connection.py) → MongoDB connection manager
- [ ] Add connection pooling configuration
- [ ] Add health check endpoint
- [ ] Migrate models from `db/models/*`

**[NEW]** [src/storage/database/neo4j.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/storage/database/neo4j.py)

**Tasks:**

- [ ] Migrate [neo4j_connection.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/db/neo4j_connection.py) → Neo4j connection manager
- [ ] Add session management
- [ ] Add transaction helpers

---

#### 3.3 Embedding Models

**Source:** `packages/server/core/embedding/*` → **Target:** `src/models/embeddings/*`

**[NEW]** [src/models/embeddings/base.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/embeddings/base.py)

**Tasks:**

- [ ] Create `EmbeddingProtocol` abstract interface
- [ ] Define `embed_text`, `embed_batch` methods

**[NEW]** [src/models/embeddings/sentence_transformer.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/embeddings/sentence_transformer.py)

**Tasks:**

- [ ] Migrate [model.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/model.py) → `SentenceTransformerEmbedding`
- [ ] Add model caching and lazy loading
- [ ] Add batch processing with configurable batch size

**[NEW]** [src/models/embeddings/factory.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/embeddings/factory.py)

**Tasks:**

- [ ] Create factory for embedding model selection
- [ ] Support multiple embedding providers (OpenAI, Cohere, etc.)

---

### Phase 4: Core Memory Logic 🧠

#### 4.1 LLM Models

**Source:** [packages/server/core/memory/model.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/model.py) → **Target:** `src/models/llm/*`

**[NEW]** [src/models/llm/base.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/llm/base.py)

**Tasks:**

- [ ] Create `LLMProtocol` abstract interface
- [ ] Define `generate`, `stream`, `batch` methods

**[NEW]** [src/models/llm/google.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/llm/google.py)

**Tasks:**

- [ ] Migrate [model.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/model.py) → `GoogleGenerativeAI`
- [ ] Add retry logic with exponential backoff
- [ ] Add rate limiting
- [ ] Add token counting and cost tracking

**[NEW]** [src/models/llm/anthropic.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/llm/anthropic.py)

**Tasks:**

- [ ] Create `AnthropicLLM` class implementing `LLMProtocol`
- [ ] Add support for Claude models (claude-3-5-sonnet, claude-3-opus, etc.)
- [ ] Add retry logic with exponential backoff
- [ ] Add rate limiting
- [ ] Add token counting and cost tracking

**[NEW]** [src/models/llm/openai.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/llm/openai.py)

**Tasks:**

- [ ] Create `OpenAILLM` class implementing `LLMProtocol`
- [ ] Add support for GPT models (gpt-4, gpt-3.5-turbo, etc.)
- [ ] Add retry logic with exponential backoff
- [ ] Add rate limiting
- [ ] Add token counting and cost tracking

**[NEW]** [src/models/llm/factory.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/models/llm/factory.py)

**Tasks:**

- [ ] Create factory for multi-provider LLM selection (Gemini, Claude, OpenAI)
- [ ] Support fallback chain (e.g., Gemini → Claude → OpenAI)
- [ ] Add provider health checks
- [ ] Add cost optimization logic (route to cheapest available model)

---

#### 4.2 Memory Service

**Source:** [packages/server/core/memory/memory_service.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/memory_service.py) → **Target:** `src/memory/service.py`

**[NEW]** [src/memory/service.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/memory/service.py)

**Tasks:**

- [ ] Migrate [memory_service.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/memory_service.py) → `MemoryService` class
- [ ] Refactor to use dependency injection (vector store, DB, LLM)
- [ ] Add async/await support throughout
- [ ] Add comprehensive error handling
- [ ] Add metrics collection (latency, token usage)

---

#### 4.3 Profile Management

**Source:** `packages/server/core/memory/profile_*.py` → **Target:** `src/memory/profile/*`

**[NEW]** [src/memory/profile/store.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/memory/profile/store.py)

**Tasks:**

- [ ] Migrate [profile_store.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/profile_store.py) → `ProfileStore` class
- [ ] Add CRUD operations for user profiles
- [ ] Add profile versioning

**[NEW]** [src/memory/profile/agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/memory/profile/agent.py)

**Tasks:**

- [ ] Migrate [profile_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/profile_agent.py) → `ProfileAgent` class
- [ ] Decouple from specific LLM implementation

---

#### 4.4 Graph Operations

**Source:** [packages/server/core/memory/graph.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/graph.py), [update_graph.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/update_graph.py) → **Target:** `src/graph/*`

**[NEW]** [src/graph/manager.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/graph/manager.py)

**Tasks:**

- [ ] Migrate [graph.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/graph.py) → `GraphManager` class
- [ ] Add graph query builders
- [ ] Add graph traversal utilities

**[NEW]** [src/graph/updater.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/graph/updater.py)

**Tasks:**

- [ ] Migrate [update_graph.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/update_graph.py) → `GraphUpdater` class
- [ ] Add batch update support
- [ ] Add conflict resolution

---

### Phase 5: Agents & Pipelines 🤖

#### 5.1 Agent Base Classes

**[NEW]** [src/agents/base.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/base.py)

**Tasks:**

- [ ] Create `BaseAgent` abstract class
- [ ] Define common interface: `run`, `arun`, `stream`
- [ ] Add prompt management
- [ ] Add output parsing

---

#### 5.2 Specific Agents

**Source:** `packages/server/core/memory/*_agent.py` → **Target:** `src/agents/*`

**[NEW]** [src/agents/classification.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/classification.py)

**Tasks:**

- [ ] Migrate [classification_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/classification_agent.py) → `ClassificationAgent`
- [ ] Inherit from `BaseAgent`

**[NEW]** [src/agents/judge.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/judge.py)

**Tasks:**

- [ ] Migrate [judge_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/judge_agent.py) → `JudgeAgent`
- [ ] Add configurable scoring thresholds

**[NEW]** [src/agents/weaver.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/weaver.py)

**Tasks:**

- [ ] Migrate [weaver_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/weaver_agent.py) → `WeaverAgent`

**[NEW]** [src/agents/summary.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/summary.py)

**Tasks:**

- [ ] Migrate [summary_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/summary_agent.py) → `SummaryAgent`

**[NEW]** [src/agents/synthesizer.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/synthesizer.py)

**Tasks:**

- [ ] Migrate [synthesizer_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/synthesizer_agent.py) → `SynthesizerAgent`

**[NEW]** [src/agents/verification.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/verification.py)

**Tasks:**

- [ ] Migrate [verification_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/verification_agent.py) → `VerificationAgent`

**[NEW]** [src/agents/code.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/agents/code.py)

**Tasks:**

- [ ] Migrate [code_agent.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/code_agent.py) → `CodeAgent`

---

#### 5.3 Pipelines

**Source:** `packages/server/core/memory/*_pipeline.py` → **Target:** `src/pipelines/*`

**[NEW]** [src/pipelines/retrieval.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/pipelines/retrieval.py)

**Tasks:**

- [ ] Migrate [retrieval_pipeline.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/retrieval_pipeline.py) → `RetrievalPipeline`
- [ ] Add pipeline stages as composable functions
- [ ] Add pipeline metrics and tracing

**[NEW]** [src/pipelines/update.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/pipelines/update.py)

**Tasks:**

- [ ] Migrate [pipeline_connector.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/core/memory/pipeline_connector.py) → `UpdatePipeline`
- [ ] Add idempotency checks

---

### Phase 6: Prompts 📝

**Source:** `packages/server/prompts/*` → **Target:** `src/prompts/*`

**Strategy:** Migrate all prompt files with versioning support.

**Tasks:**

- [ ] Migrate [classification_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/classification_prompts.py) + [classification_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/classification_examples.py)
- [ ] Migrate [judge_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/judge_prompts.py)
- [ ] Migrate [weaver_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/weaver_prompts.py)
- [ ] Migrate [summary_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/summary_prompts.py) + [summary_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/summary_examples.py)
- [ ] Migrate [synthesis_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/synthesis_prompts.py) + [synthesizer_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/synthesizer_examples.py)
- [ ] Migrate [synthesizer_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/synthesizer_prompts.py) + [synthesizer_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/synthesizer_examples.py)
- [ ] Migrate [verification_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/verification_prompts.py) + [verification_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/verification_examples.py)
- [ ] Migrate [code_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/code_prompts.py) + [code_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/code_examples.py)
- [ ] Migrate [temporal_prompts.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/temporal_prompts.py) + [temporal_examples.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/temporal_examples.py)
- [ ] Migrate [extract_profile.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/extract_profile.py)
- [ ] Migrate [user_profile_topics.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/user_profile_topics.py)
- [ ] Migrate [agent_keywords.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/agent_keywords.py)
- [ ] Migrate [utils.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/prompts/utils.py) → `src/prompts/utils.py`
- [ ] Add prompt versioning system (v1, v2, etc.)
- [ ] Add prompt template validation

---

### Phase 7: API Layer 🌐

#### 7.1 API Routes

**Source:** `packages/server/api/v1/*` → **Target:** `src/api/v1/*`

**[NEW]** [src/api/v1/memory.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/api/v1/memory.py)

**Tasks:**

- [ ] Migrate memory endpoints (add, search, update, delete)
- [ ] Add request validation with Pydantic
- [ ] Add response models
- [ ] Add rate limiting decorators
- [ ] Add authentication middleware (if needed)

**[NEW]** [src/api/v1/sessions.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/api/v1/sessions.py)

**Tasks:**

- [ ] Migrate session endpoints (create, get, list, delete)

**[NEW]** [src/api/v1/profiles.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/api/v1/profiles.py)

**Tasks:**

- [ ] Migrate profile endpoints (get, update)

**[NEW]** [src/api/v1/health.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/api/v1/health.py)

**Tasks:**

- [ ] Create health check endpoints
- [ ] Add dependency health checks (DB, vector store, LLM)

---

#### 7.2 Main Application

**Source:** [packages/server/main.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/packages/server/main.py) → **Target:** `src/main.py`

**[NEW]** [src/main.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/main.py)

**Tasks:**

- [ ] Create FastAPI app instance
- [ ] Add CORS middleware
- [ ] Add logging middleware
- [ ] Add exception handlers
- [ ] Register all routers
- [ ] Add startup/shutdown events (DB connections)
- [ ] Add OpenAPI documentation customization

---

### Phase 8: Utilities 🛠️

**Source:** `packages/server/utils/*`, `packages/server/controllers/*` → **Target:** `src/utils/*`

**[NEW]** [src/utils/helpers.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/utils/helpers.py)

**Tasks:**

- [ ] Migrate common utility functions
- [ ] Add text processing utilities
- [ ] Add date/time utilities

**[NEW]** [src/utils/validators.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/src/utils/validators.py)

**Tasks:**

- [ ] Create custom Pydantic validators
- [ ] Add input sanitization functions

---

### Phase 9: Testing 🧪

**Source:** `packages/server/tests/*` → **Target:** `tests/*`

#### 9.1 Test Infrastructure

**[NEW]** [tests/conftest.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/tests/conftest.py)

**Tasks:**

- [ ] Create pytest fixtures for DB connections
- [ ] Create fixtures for mock LLM responses
- [ ] Create fixtures for test data
- [ ] Add test configuration

---

#### 9.2 Unit Tests

**Tasks:**

- [ ] Migrate and update all existing tests from `packages/server/tests/*`
- [ ] Add tests for new modular components
- [ ] Add tests for `src/storage/vectorstore/*`
- [ ] Add tests for `src/storage/database/*`
- [ ] Add tests for `src/models/*`
- [ ] Add tests for `src/memory/*`
- [ ] Add tests for `src/agents/*`
- [ ] Add tests for `src/pipelines/*`
- [ ] Add tests for `src/api/*`
- [ ] Achieve >80% code coverage

---

#### 9.3 Integration Tests

**[NEW]** [tests/integration/test_memory_flow.py](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/tests/integration/test_memory_flow.py)

**Tasks:**

- [ ] Create end-to-end memory add → search flow test
- [ ] Test with real vector store (test mode)
- [ ] Test with real LLM (mock or test API key)

---

### Phase 10: Benchmarks 📊

**Source:** `packages/server/benchmark/*` → **Target:** `benchmarks/*`

**Tasks:**

- [ ] Migrate all benchmark scripts
- [ ] Update import paths
- [ ] Add new benchmarks for modular components
- [ ] Create benchmark reporting dashboard

---

### Phase 11: Documentation 📚

**[NEW]** [README.md](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/README.md)

**Tasks:**

- [ ] Write project overview
- [ ] Add installation instructions
- [ ] Add quick start guide
- [ ] Add architecture diagram
- [ ] Add API documentation link
- [ ] Add contributing guidelines link

**[NEW]** [CONTRIBUTING.md](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/CONTRIBUTING.md)

**Tasks:**

- [ ] Write contribution guidelines
- [ ] Add code style guide
- [ ] Add PR template
- [ ] Add issue templates

**[NEW]** [docs/architecture.md](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/docs/architecture.md)

**Tasks:**

- [ ] Document system architecture
- [ ] Add component diagrams
- [ ] Document data flow

**[NEW]** [docs/api.md](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/docs/api.md)

**Tasks:**

- [ ] Document all API endpoints
- [ ] Add request/response examples
- [ ] Add authentication guide

---

### Phase 12: DevOps & CI/CD 🚀

**[NEW]** [.github/workflows/test.yml](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/.github/workflows/test.yml)

**Tasks:**

- [ ] Create GitHub Actions workflow for testing
- [ ] Add matrix testing (Python 3.10, 3.11, 3.12)
- [ ] Add code coverage reporting

**[NEW]** [.github/workflows/lint.yml](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/.github/workflows/lint.yml)

**Tasks:**

- [ ] Create linting workflow (ruff, black, mypy)
- [ ] Add pre-commit hooks

**[NEW]** [docker/Dockerfile](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/docker/Dockerfile)

**Tasks:**

- [ ] Create production Dockerfile
- [ ] Add multi-stage build
- [ ] Optimize image size

**[NEW]** [docker-compose.yml](file:///Users/vedantmahajan/Desktop/Xortex/Xmem/docker-compose.yml)

**Tasks:**

- [ ] Create docker-compose for local development
- [ ] Add services: API, MongoDB, Neo4j
- [ ] Add environment variable configuration
- [ ] Add volume mounts for persistence

---

## Verification Plan

### Automated Tests

```bash
# Install dependencies
pip install -e ".[dev]"

# Run linting
ruff check src tests
black --check src tests
mypy src

# Run tests
pytest tests/ -v --cov=src --cov-report=html

# Run benchmarks
python benchmarks/xmem_add.py
python benchmarks/xmem_search.py
```

### Manual Verification

1. **Environment Setup**: Verify [.env](file:///Users/vedantmahajan/Desktop/Xortex/Xmem_v0/.env) configuration loads correctly
2. **Database Connections**: Test MongoDB and Neo4j connections
3. **Vector Store**: Test Pinecone operations (add, search, delete)
4. **LLM Providers**: Test all three providers (Gemini, Claude, OpenAI) and fallback logic
5. **API Endpoints**: Test all endpoints with Postman/curl
6. **End-to-End Flow**: Add memory → Search → Retrieve → Update
7. **Performance**: Compare benchmark results with `Xmem_v0`

### Migration Checklist

- [ ] All files migrated from `Xmem_v0`
- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] Documentation complete
- [ ] Docker setup working
- [ ] CI/CD pipelines passing
- [ ] Performance benchmarks meet/exceed v0
- [ ] Security audit complete (no hardcoded secrets)
- [ ] License file added
- [ ] CHANGELOG.md created
