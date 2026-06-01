# XMem Codex Plugin

Codex plugin for searching and saving XMem memory from the main XMem repository.

This is inspired by `codex-supermemory`, but intentionally kept repo-local and dependency-free for now.

## Configuration

Use environment variables. Do not commit secrets.

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

`XMEM_API_URL` defaults to `https://api.xmem.in`. `XMEM_USER_ID` falls back to the local OS username.

## Skills

- `xmem-search` - search prior XMem memories
- `xmem-save` - save durable project knowledge
- `xmem-status` - verify environment configuration without printing secrets

## Scripts

```bash
node plugin/xmem-codex/scripts/search-memory.cjs "query"
node plugin/xmem-codex/scripts/save-memory.cjs "memory to save"
node plugin/xmem-codex/scripts/status.cjs
```

The scripts redact obvious API key patterns before saving content.
