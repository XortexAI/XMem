---
description: Check whether XMem Antigravity memory is configured
allowed-tools: ["Bash"]
---

# XMem Session

Check plugin configuration without printing secrets:

```bash
node -e "console.log(process.env.XMEM_API_KEY ? 'XMEM_API_KEY is set' : 'XMEM_API_KEY is not set'); console.log('XMEM_API_URL=' + (process.env.XMEM_API_URL || 'https://api.xmem.in'))"
```

If the key is missing, set `XMEM_API_KEY` before starting Antigravity.
