# xmem-hermes

Hermes Agent MCP connector for XMem persistent memory.

This package connects Hermes to XMem through the XMem MCP server. It writes local connector configuration and agent-facing memory instructions, while keeping XMem credentials in environment variables.

## Install

```bash
npx xmem-hermes@latest install
```

For local development:

```bash
node src/cli.js install --config-root ./tmp/hermes
```

## Authentication

Set credentials in your shell, OS secret store, or the client environment before launching Hermes:

```bash
export XMEM_API_URL="https://api.xmem.in"
export XMEM_API_KEY="xmem_..."
export XMEM_USERNAME="your-xmem-username"
```

The installer writes `${XMEM_API_KEY}` as a placeholder instead of copying secret values into config files.

## Commands

| Command | Description |
| --- | --- |
| `install` | Write `.hermes/config.yaml` and `HERMES.md`. |
| `doctor` | Check whether generated connector files exist. |
| `smoke-test` | Verify XMem API access via environment variables without printing secrets. |

## Smoke Test

```bash
XMEM_API_KEY="xmem_..." XMEM_USERNAME="connector-test" npm run smoke
```

The smoke test calls XMem search with a low-risk read query. It never logs the API key.

## Notes

- Requires the XMem MCP server to be available as `uvx xmem-mcp`.
- Uses stdio transport by default for local agent clients.
- You can override the API URL with `--api-url` during install.

## License

Apache-2.0
