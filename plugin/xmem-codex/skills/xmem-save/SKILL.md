---
name: xmem-save
description: Save important project knowledge, decisions, conventions, bug fixes, or implementation context into XMem.
allowed-tools: Bash(node:*)
---

# XMem Save

Use this skill when the user asks to remember or save durable project knowledge for future Codex sessions.

Format the memory:

```text
[SAVE:<date>]
Project:
Decision or fact:
Relevant files:
Why it matters:
[/SAVE]
```

Then run:

```bash
node plugin/xmem-codex/scripts/save-memory.cjs "FORMATTED_CONTENT"
```

Never save API keys, tokens, passwords, private customer data, or other secrets.
