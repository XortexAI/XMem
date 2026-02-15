# Xmem

**Universal Unified Memory System for AI Agents**

> **Status**: Early Development / Work in Progress

Xmem is a framework designed to provide AI agents with a comprehensive memory system that integrates vector, graph, and document databases. It aims to solve the problem of fragmented memory by offering a unified interface for storing and retrieving user profiles, events, and code knowledge.

## Features

### 🧠 Intelligent Agent Routing
At the core of Xmem is the **Classifier Agent**, which intelligently routes user queries to specialized memory intents:
- **`code`**: For software engineering and technical tasks.
- **`profile`**: For permanent user facts (identity, preferences, traits).
- **`event`**: For time-based events and memories.

### 🤖 Multi-LLM Support with Fallback
Xmem features a robust **LLM Registry** that supports multiple providers:
- **Google Gemini**
- **Anthropic Claude**
- **OpenAI GPT**

It includes a smart fallback mechanism: if one provider fails, it automatically tries the next one in your configured priority list (`settings.fallback_order`).

### ⚙️ Centralized Configuration
All settings are managed via `pydantic-settings`, allowing for easy configuration through environment variables or a `.env` file. It validates required keys (like Pinecone API key) at startup to prevent runtime errors.

### 🏗️ Unified Memory Vision (Planned)
Future releases will fully implement the unified memory architecture:
- **Vector Store (Pinecone)**: For semantic search and long-term memory.
- **Graph Database (Neo4j)**: For relationship mapping and knowledge graphs.
- **Document Store (MongoDB)**: For structured data and logs.

## Prerequisites

- Python 3.11 or higher
- [Pinecone](https://www.pinecone.io/) account (for vector storage)
- [Neo4j](https://neo4j.com/) instance (for graph storage)
- [MongoDB](https://www.mongodb.com/) instance (for document storage)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/xmem.git
    cd xmem
    ```

2.  **Install dependencies**:
    ```bash
    pip install -e .
    # Or for development with testing tools:
    pip install -e .[dev]
    ```

## Configuration

Create a `.env` file in the root directory. You can copy the structure from `src/config/settings.py`.

**Required Environment Variables:**

```ini
# Vector Database
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=xmem-index  # Optional, default: xmem-index

# Graph Database
NEO4J_URI=bolt://localhost:7687 # Optional, default provided
NEO4J_PASSWORD=your_neo4j_password

# LLM Providers (At least one is required)
GEMINI_API_KEY=your_gemini_key
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

**Optional Settings:**

```ini
# Database Connections
MONGODB_URI=mongodb://localhost:27017

# Model Selection
GEMINI_MODEL=gemini-2.5-flash-lite
CLAUDE_MODEL=claude-3-5-sonnet
OPENAI_MODEL=gpt-4.1-mini
TEMPERATURE=0.4
```

## Usage

### Using the LLM Registry

You can get an initialized Chat Model instance using `get_model`. It will use the first available provider from your fallback order.

```python
from src.models import get_model

# Get the default model (tries Gemini -> Claude -> OpenAI)
model = get_model()
response = model.invoke("Hello, who are you?")
print(response.content)

# Force a specific provider
gemini = get_model(provider="gemini")
```

### Using the Classifier Agent

The Classifier Agent analyzes user input and determines the intent.

```python
import asyncio
from src.models import get_model
from src.agents.classifier import ClassifierAgent

async def main():
    model = get_model()
    classifier = ClassifierAgent(model)

    state = {"user_query": "My birthday is on March 15th"}
    result = await classifier.arun(state)

    print(result.classifications)
    # Output: [{'source': 'event', 'query': 'My birthday is on March 15th'}]

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

- **`src/agents`**: Agent implementations (Classifier, Base).
- **`src/config`**: Configuration and settings management.
- **`src/models`**: LLM provider integrations and registry.
- **`src/prompts`**: System prompts and prompt management.
- **`src/utils`**: Utility functions (text parsing, etc.).
- **`src/graph`**: (Planned) Neo4j graph database client.
- **`src/memory`**: (Planned) Memory service logic.
- **`src/pipelines`**: (Planned) Data ingestion and retrieval pipelines.
- **`src/api`**: (Planned) FastAPI application and routes.

## Development

To run the tests, use `pytest`. Make sure you have the required environment variables set (or dummy values for unit tests).

```bash
# Run tests
PINECONE_API_KEY=dummy NEO4J_PASSWORD=dummy GEMINI_API_KEY=dummy pytest
```
