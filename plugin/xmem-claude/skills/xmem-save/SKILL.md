---
name: xmem-save
description: Save important project knowledge, decisions, conventions, bug fixes, or implementation context into XMem.
allowed-tools: Bash(node:*)
---

# XMem Save

Use this skill when the user asks to remember or save something for future Claude Code sessions.

Format the memory with useful context:

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
node "${CLAUDE_PLUGIN_ROOT}/scripts/save-project-memory.cjs" "FORMATTED_CONTENT"
```

Never include API keys, tokens, passwords, or other secrets in the saved content.
