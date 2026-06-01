---
description: Index the current codebase into XMem for future Antigravity context
allowed-tools: ["Read", "Glob", "Grep", "Bash"]
---

# Index Codebase Into XMem

Explore the repository and save a concise architecture summary into XMem.

1. Read `README.md`, package manifests, config files, and entry points.
2. Identify the stack, runtime commands, major modules, API routes, data stores, and conventions.
3. Skip dependency folders, generated output, lock files, virtual environments, and secrets.
4. Save the final summary:

```bash
node "${ANTIGRAVITY_PLUGIN_ROOT}/scripts/save-project-memory.cjs" "SUMMARY_HERE"
```

Include important files and decisions, but do not save secrets.
