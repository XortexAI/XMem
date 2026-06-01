# opencode-xmem

OpenCode plugin for persistent memory using [XMem](https://xmem.in).

Your agent remembers what you tell it — across sessions, across projects.

## Installation

### For Humans

```bash
bunx opencode-xmem@latest install
```

This registers the plugin and opens browser login to connect your XMem account.

**Or let your agent do it** — paste this into OpenCode:

```
Install opencode-xmem by following https://raw.githubusercontent.com/xmem-ai/opencode-xmem/main/README.md
```

### For LLM Agents

<details>
<summary>Instructions for LLM Agents (click to expand)</summary>

#### Step 1: Run the installer

```bash
bunx opencode-xmem@latest install --no-tui
```

This will:

- Register the plugin in `~/.config/opencode/opencode.jsonc`
- Create the `/xmem-init`, `/xmem-login`, and `/xmem-logout` commands

#### Step 2: Verify the config

```bash
cat ~/.config/opencode/opencode.jsonc
```

Should contain:

```json
{
  "plugin": ["opencode-xmem@latest"]
}
```

#### Step 3: Authenticate

```bash
bunx opencode-xmem@latest login
```

Or set credentials manually:

```bash
export XMEM_API_KEY="xmem_..."
export XMEM_USERNAME="your_username"
export XMEM_API_URL="https://api.xmem.in"
```

#### Step 4: Restart OpenCode

Restart OpenCode to load the plugin.

</details>

## Commands

| Command | Description |
|---------|-------------|
| `install` | Register plugin and create slash commands |
| `login` | Browser auth via xmem.in/auth/connect |
| `logout` | Clear ~/.xmem-opencode/credentials.json |

## Agent Tool: `xmem`

The plugin exposes an `xmem` tool to the coding agent:

| Mode | Description |
|------|-------------|
| `add` | Store a memory (`content`, optional `scope`) |
| `search` | Search raw memory records (`query`, optional `scope`) |
| `recall` | Get synthesized answer from memories (`query`, optional `scope`) |
| `code` | Query indexed codebase (`query`, `orgId`, `repo`) |
| `help` | Show usage guide |

**Scopes:**

- `user` — cross-project preferences
- `project` — project-specific knowledge (default)

## Configuration

Optional config at `~/.config/opencode/xmem.jsonc`:

```jsonc
{
  "maxMemories": 5,
  "maxProjectMemories": 10,
  "autoRecallEveryPrompt": true,
  "compactionThreshold": 0.80,
  "defaultOrgId": "your-org",
  "defaultRepo": "your-repo"
}
```

## Features

- **Auto-recall** — Injects relevant memories on the first message of each session
- **Keyword detection** — Nudges the agent when you say "remember this"
- **Preemptive compaction** — Saves session summaries to XMem before context overflow
- **Code memory** — Query indexed repos via the `code` tool mode

## Links

- [XMem](https://xmem.in)
- [Docs](https://xmem.in/docs#opencode)
- [Connect manually](https://xmem.in/auth/connect?client=opencode)

## License

Apache-2.0
