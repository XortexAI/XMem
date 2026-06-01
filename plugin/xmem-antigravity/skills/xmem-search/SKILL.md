---
name: xmem-search
description: Search XMem for prior Antigravity sessions, project decisions, implementation notes, and remembered coding context.
allowed-tools: Bash(node:*)
---

# XMem Search

Use this skill when the user asks to recall previous work, earlier implementation details, project decisions, or saved coding memory.

Run:

```bash
node "${ANTIGRAVITY_PLUGIN_ROOT}/scripts/search-memory.cjs" "USER_QUERY_HERE"
```

Summarize the returned memories clearly. If nothing useful is returned, ask a sharper follow-up question or try a more specific query.
