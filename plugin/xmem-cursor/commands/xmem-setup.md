---
description: Configure XMem memory for Cursor
---

# XMem Setup

Set the API key in your shell or secret manager before starting Cursor:

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

Then verify:

```bash
node "${CURSOR_PLUGIN_ROOT}/scripts/status.cjs"
```

Do not commit API keys to project files.
