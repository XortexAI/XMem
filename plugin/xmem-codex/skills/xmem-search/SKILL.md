---
name: xmem-search
description: Search XMem for prior Codex sessions, project decisions, implementation notes, and saved coding context.
allowed-tools: Bash(node:*)
---

# XMem Search

Use this skill when the user asks to recall prior work, implementation details, project decisions, or saved coding memory.

Run:

```bash
node plugin/xmem-codex/scripts/search-memory.cjs "USER_QUERY_HERE"
```

Present the returned memories clearly, including the most relevant details and any file paths mentioned. If the results are thin, try a more specific query before giving up.
