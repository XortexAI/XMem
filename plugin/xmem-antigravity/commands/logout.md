---
description: Remove local XMem Antigravity plugin project config
allowed-tools: ["Bash"]
---

# XMem Logout

This plugin does not store credentials by default. It reads `XMEM_API_KEY` from the environment.

To remove project-local config:

```bash
node -e "const fs=require('fs'); const p='.antigravity/.xmem-antigravity/config.json'; if(fs.existsSync(p)){fs.rmSync(p); console.log('Removed '+p)}else{console.log('No project config found')}"
```

Also unset `XMEM_API_KEY` in your shell if you want to disconnect this terminal session.
