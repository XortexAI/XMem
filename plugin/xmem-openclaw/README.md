# XMem OpenClaw Plugin

Long-term memory for OpenClaw powered by XMem.

This mirrors `openclaw-supermemory`'s shape: a root `openclaw.plugin.json`, OpenClaw runtime entrypoint, tools, hooks, slash commands, config parsing, and memory runtime integration.

## Setup

Use environment variables or plugin config. Do not commit secrets.

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

## Tools

- `xmem_search` - search long-term XMem memory
- `xmem_store` - save important information to XMem
- `xmem_status` - show safe connection status without printing secrets

## Slash Commands

- `/remember <text>` - manually save something to XMem
- `/recall <query>` - search XMem memories
- `/xmem-status` - show safe plugin status

## Configuration

```json
{
  "plugins": {
    "entries": {
      "xmem-openclaw": {
        "enabled": true,
        "config": {
          "apiKey": "${XMEM_API_KEY}",
          "apiUrl": "https://api.xmem.in",
          "autoRecall": true,
          "autoCapture": true,
          "maxRecallResults": 8,
          "debug": false
        }
      }
    }
  }
}
```

Prefer environment variables or a secret manager for API keys.
