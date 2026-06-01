# XMem Antigravity Plugin

Antigravity plugin for persistent memory through XMem.

This plugin mirrors the structure of `xmem-claude`, adapted for Antigravity's
agent lifecycle:

- loads relevant project memory on `SessionStart`
- stores a redacted transcript tail on `Stop`
- provides `xmem-search` and `xmem-save` skills
- includes commands for indexing, config, session checks, and logout

## Install

From Antigravity, install this plugin from the local folder once it is published
or linked by the plugin marketplace flow for this repo.

## Configuration

Use environment variables. Do not commit secrets.

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_API_URL="https://api.xmem.in"
export XMEM_USER_ID="your-user-id"
```

`XMEM_API_URL` defaults to `https://api.xmem.in`. `XMEM_USER_ID` falls back to the local OS username; production API keys scope requests to the authenticated key owner.

Optional project config can live at `.antigravity/.xmem-antigravity/config.json`:

```json
{
  "apiUrl": "https://api.xmem.in",
  "userId": "your-user-id"
}
```

Prefer environment variables for API keys.

## Commands

- `/xmem-antigravity:index` - explore the current repo and save a project summary
- `/xmem-antigravity:project-config` - show configuration options
- `/xmem-antigravity:session` - check whether memory is configured
- `/xmem-antigravity:logout` - remove project-local config

## Skills

- `xmem-search` - search prior XMem memories
- `xmem-save` - save durable project knowledge
