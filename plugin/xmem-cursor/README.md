# XMem Cursor Plugin

Cursor plugin for persistent coding memory through XMem.

This mirrors the strong structure of `cursor-supermemory`:

- `.cursor-plugin/plugin.json` for plugin metadata
- `.mcp.json` and `.cursor/mcp.json` for MCP registration
- `rules/` with proactive memory guidance
- `skills/` for search, save, and setup
- `commands/` for setup/config/logout
- `hooks/` and small Node scripts for session lifecycle

## Configuration

Use environment variables. Do not commit secrets.

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

`XMEM_API_URL` defaults to `https://api.xmem.in`. `XMEM_USER_ID` falls back to the local OS username.

## MCP Tools

- `xmem_status` - show safe config status
- `xmem_search` - search prior memory
- `xmem_add` - save project knowledge

## Local Checks

```bash
npm run check
node scripts/status.cjs
```
