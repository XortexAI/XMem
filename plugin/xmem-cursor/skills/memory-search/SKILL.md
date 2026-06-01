---
name: memory-search
description: Search XMem persistent memory for relevant information from past coding sessions. Use when the user asks about previous work, past bugs, architectural decisions, or anything that may have been worked on before.
---

1. Call `xmem_search` with a focused query based on what the user is asking.
2. If results are found, surface the relevant memories with enough context to be useful.
3. If no results are found, say that no prior XMem memory matched this topic.
4. For broad questions, try more specific project, file, or feature terms.
