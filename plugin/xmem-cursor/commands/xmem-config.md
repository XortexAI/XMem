---
description: Show XMem Cursor configuration
---

# XMem Config

The plugin reads:

- `XMEM_API_KEY` or `XMEM_CURSOR_API_KEY`
- `XMEM_API_URL` or `XMEM_CURSOR_API_URL` (defaults to `https://api.xmem.in`)
- `XMEM_USER_ID` or `XMEM_CURSOR_USER_ID` (falls back to your OS username)

Run:

```bash
node "${CURSOR_PLUGIN_ROOT}/scripts/status.cjs"
```

The status command never prints the API key value.
