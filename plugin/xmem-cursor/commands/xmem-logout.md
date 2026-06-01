---
description: Disconnect XMem from the current Cursor shell
---

# XMem Logout

This plugin does not store API keys by default. It reads them from environment variables.

Unset the key in your shell to disconnect:

```bash
unset XMEM_API_KEY
unset XMEM_CURSOR_API_KEY
```

On PowerShell:

```powershell
Remove-Item Env:XMEM_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:XMEM_CURSOR_API_KEY -ErrorAction SilentlyContinue
```
