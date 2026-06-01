---
name: xmem-status
description: Check whether the XMem Codex plugin has the environment variables needed to search and save memory.
allowed-tools: Bash(node:*)
---

# XMem Status

Use this skill when XMem memory commands fail or the user asks whether the plugin is configured.

Run:

```bash
node plugin/xmem-codex/scripts/status.cjs
```

Do not print API key values. If `XMEM_API_KEY` is missing, ask the user to set it in their shell or secret manager.
