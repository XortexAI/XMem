---
description: Show XMem Claude plugin configuration options
allowed-tools: ["Read", "Write", "Bash"]
---

# XMem Claude Configuration

The plugin reads credentials from environment variables first:

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

Optional project config lives at `.claude/.xmem-claude/config.json`:

```json
{
  "apiUrl": "https://api.xmem.in",
  "userId": "your-user-id"
}
```

Avoid storing API keys in project config. Prefer environment variables or your shell secret manager.
