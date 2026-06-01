# XMem Claude Plugin

Claude Code plugin for persistent memory through XMem.

This package is inspired by `claude-supermemory`, but talks directly to XMem:

- loads relevant project memory on `SessionStart`
- stores a redacted transcript tail on `Stop`
- provides `xmem-search` and `xmem-save` skills
- includes commands for indexing, config, session checks, and logout

## Install

From Claude Code, install this plugin from the local folder once it is published or linked by the plugin marketplace flow for this repo.

## Configuration

Use environment variables. Do not commit secrets.

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

`XMEM_API_URL` defaults to `https://api.xmem.in`. `XMEM_USER_ID` falls back to the local OS username; production API keys scope requests to the authenticated key owner.

Optional project config can live at `.claude/.xmem-claude/config.json`:

```json
{
  "apiUrl": "https://api.xmem.in",
  "userId": "your-user-id"
}
```

Prefer environment variables for API keys.

## Commands

- `/xmem-claude:index` - explore the current repo and save a project summary
- `/xmem-claude:project-config` - show configuration options
- `/xmem-claude:session` - check whether memory is configured
- `/xmem-claude:logout` - remove project-local config

## Skills

- `xmem-search` - search prior XMem memories
- `xmem-save` - save durable project knowledge
