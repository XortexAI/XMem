#!/usr/bin/env node
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import * as readline from "node:readline";
import { stripJsoncComments } from "./services/jsonc.js";
import { startAuthFlow, clearCredentials, loadCredentials } from "./services/auth.js";
import { writeInstallDefaults, CONFIG_FILE } from "./config.js";

const OPENCODE_CONFIG_DIR = join(homedir(), ".config", "opencode");
const OPENCODE_COMMAND_DIR = join(OPENCODE_CONFIG_DIR, "command");
const OH_MY_OPENCODE_CONFIG = join(OPENCODE_CONFIG_DIR, "oh-my-opencode.json");
const PLUGIN_NAME = "opencode-xmem@latest";

const XMEM_INIT_COMMAND = `---
description: Initialize XMem with comprehensive codebase knowledge
---

# Initializing XMem

You are initializing persistent memory for this codebase using XMem.

## What to Remember

### 1. Procedures (Rules & Workflows)
- Build, test, lint commands
- Branching and commit conventions

### 2. Preferences (Style & Conventions)
- Coding style preferences
- Framework and library choices

### 3. Architecture & Context
- Key directories and data flow
- Known issues and solutions

## Memory Scopes

**Project-scoped** (\`scope: "project"\`):
- Build/test/lint commands, architecture, team conventions

**User-scoped** (\`scope: "user"\`):
- Personal coding preferences across all projects

## Saving Memories

Use the \`xmem\` tool for each distinct insight:

\`\`\`
xmem(mode: "add", content: "...", scope: "project")
\`\`\`

## Your Task

1. Check existing memories: \`xmem(mode: "recall", query: "project context", scope: "project")\`
2. Research the codebase thoroughly
3. Save memories incrementally as you discover insights
4. Summarize what was learned
`;

const XMEM_LOGIN_COMMAND = `---
description: Authenticate with XMem via browser
---

# XMem Login

Run this command to authenticate the user with XMem:

\`\`\`bash
bunx opencode-xmem@latest login
\`\`\`

This will:
1. Start a local server on port 19878
2. Open the browser to XMem's authentication page
3. After the user logs in, save credentials to ~/.xmem-opencode/credentials.json

Wait for the command to complete, then inform the user whether authentication succeeded or failed.
`;

const XMEM_LOGOUT_COMMAND = `---
description: Log out from XMem and clear credentials
---

# XMem Logout

Run this command to log out and clear XMem credentials:

\`\`\`bash
bunx opencode-xmem@latest logout
\`\`\`

This will remove the saved credentials from ~/.xmem-opencode/credentials.json.
`;

function createReadline(): readline.Interface {
  return readline.createInterface({ input: process.stdin, output: process.stdout });
}

async function confirm(rl: readline.Interface, question: string): Promise<boolean> {
  return new Promise((resolve) => {
    rl.question(`${question} (y/n) `, (answer) => {
      resolve(answer.toLowerCase() === "y" || answer.toLowerCase() === "yes");
    });
  });
}

function findOpencodeConfig(): string | null {
  const candidates = [
    join(OPENCODE_CONFIG_DIR, "opencode.jsonc"),
    join(OPENCODE_CONFIG_DIR, "opencode.json"),
  ];
  for (const path of candidates) {
    if (existsSync(path)) return path;
  }
  return null;
}

function addPluginToConfig(configPath: string): boolean {
  try {
    const content = readFileSync(configPath, "utf-8");
    if (content.includes("opencode-xmem")) {
      console.log("✓ Plugin already registered in config");
      return true;
    }

    const jsonContent = stripJsoncComments(content);
    let config: Record<string, unknown>;
    try {
      config = JSON.parse(jsonContent);
    } catch {
      console.error("✗ Failed to parse config file");
      return false;
    }

    const plugins = (config.plugin as string[]) || [];
    plugins.push(PLUGIN_NAME);
    config.plugin = plugins;

    if (configPath.endsWith(".jsonc")) {
      if (content.includes('"plugin"')) {
        const newContent = content.replace(
          /("plugin"\s*:\s*\[)([^\]]*?)(\])/,
          (_match, start, middle, end) => {
            const trimmed = middle.trim();
            if (trimmed === "") {
              return `${start}\n    "${PLUGIN_NAME}"\n  ${end}`;
            }
            return `${start}${middle.trimEnd()},\n    "${PLUGIN_NAME}"\n  ${end}`;
          }
        );
        writeFileSync(configPath, newContent);
      } else {
        const newContent = content.replace(/^(\s*\{)/, `$1\n  "plugin": ["${PLUGIN_NAME}"],`);
        writeFileSync(configPath, newContent);
      }
    } else {
      writeFileSync(configPath, JSON.stringify(config, null, 2));
    }

    console.log(`✓ Added plugin to ${configPath}`);
    return true;
  } catch (err) {
    console.error("✗ Failed to update config:", err);
    return false;
  }
}

function createNewConfig(): boolean {
  const configPath = join(OPENCODE_CONFIG_DIR, "opencode.jsonc");
  mkdirSync(OPENCODE_CONFIG_DIR, { recursive: true });
  writeFileSync(configPath, `{\n  "plugin": ["${PLUGIN_NAME}"]\n}\n`);
  console.log(`✓ Created ${configPath}`);
  return true;
}

function createCommands(): boolean {
  mkdirSync(OPENCODE_COMMAND_DIR, { recursive: true });
  writeFileSync(join(OPENCODE_COMMAND_DIR, "xmem-init.md"), XMEM_INIT_COMMAND);
  writeFileSync(join(OPENCODE_COMMAND_DIR, "xmem-login.md"), XMEM_LOGIN_COMMAND);
  writeFileSync(join(OPENCODE_COMMAND_DIR, "xmem-logout.md"), XMEM_LOGOUT_COMMAND);
  console.log("✓ Created /xmem-init, /xmem-login, and /xmem-logout commands");
  return true;
}

function isOhMyOpencodeInstalled(): boolean {
  const configPath = findOpencodeConfig();
  if (!configPath) return false;
  try {
    return readFileSync(configPath, "utf-8").includes("oh-my-opencode");
  } catch {
    return false;
  }
}

function isAutoCompactAlreadyDisabled(): boolean {
  if (!existsSync(OH_MY_OPENCODE_CONFIG)) return false;
  try {
    const config = JSON.parse(readFileSync(OH_MY_OPENCODE_CONFIG, "utf-8"));
    const disabledHooks = config.disabled_hooks as string[] | undefined;
    return disabledHooks?.includes("anthropic-context-window-limit-recovery") ?? false;
  } catch {
    return false;
  }
}

function disableAutoCompactHook(): boolean {
  try {
    let config: Record<string, unknown> = {};
    if (existsSync(OH_MY_OPENCODE_CONFIG)) {
      config = JSON.parse(readFileSync(OH_MY_OPENCODE_CONFIG, "utf-8"));
    }
    const disabledHooks = (config.disabled_hooks as string[]) || [];
    if (!disabledHooks.includes("anthropic-context-window-limit-recovery")) {
      disabledHooks.push("anthropic-context-window-limit-recovery");
    }
    config.disabled_hooks = disabledHooks;
    writeFileSync(OH_MY_OPENCODE_CONFIG, JSON.stringify(config, null, 2));
    console.log("✓ Disabled anthropic-context-window-limit-recovery hook in oh-my-opencode.json");
    return true;
  } catch (err) {
    console.error("✗ Failed to update oh-my-opencode.json:", err);
    return false;
  }
}

interface InstallOptions {
  tui: boolean;
  disableAutoCompact: boolean;
}

async function install(options: InstallOptions): Promise<number> {
  console.log("\n🧠 opencode-xmem installer\n");

  writeInstallDefaults(existsSync(CONFIG_FILE));

  const rl = options.tui ? createReadline() : null;

  console.log("Step 1: Register plugin in OpenCode config");
  const configPath = findOpencodeConfig();

  if (configPath) {
    if (options.tui) {
      const shouldModify = await confirm(rl!, `Add plugin to ${configPath}?`);
      if (shouldModify) addPluginToConfig(configPath);
      else console.log("Skipped.");
    } else {
      addPluginToConfig(configPath);
    }
  } else {
    if (options.tui) {
      const shouldCreate = await confirm(rl!, "No OpenCode config found. Create one?");
      if (shouldCreate) createNewConfig();
      else console.log("Skipped.");
    } else {
      createNewConfig();
    }
  }

  console.log("\nStep 2: Create /xmem-init, /xmem-login, and /xmem-logout commands");
  if (options.tui) {
    const shouldCreate = await confirm(rl!, "Add xmem commands?");
    if (shouldCreate) createCommands();
    else console.log("Skipped.");
  } else {
    createCommands();
  }

  if (isOhMyOpencodeInstalled()) {
    console.log("\nStep 3: Configure Oh My OpenCode");
    if (isAutoCompactAlreadyDisabled()) {
      console.log("✓ anthropic-context-window-limit-recovery hook already disabled");
    } else if (options.tui) {
      const shouldDisable = await confirm(
        rl!,
        "Disable anthropic-context-window-limit-recovery hook to let XMem handle context?"
      );
      if (shouldDisable) disableAutoCompactHook();
      else console.log("Skipped.");
    } else if (options.disableAutoCompact) {
      disableAutoCompactHook();
    } else {
      console.log("Skipped. Use --disable-context-recovery to disable the hook in non-interactive mode.");
    }
  }

  if (rl) rl.close();

  console.log("\n" + "─".repeat(50));
  console.log("\n🔑 Final step: Authenticate with XMem\n");

  if (options.tui) {
    return login();
  }

  console.log("Run this command to authenticate:");
  console.log("  bunx opencode-xmem@latest login");
  console.log("\nOr set credentials manually:");
  console.log('  export XMEM_API_KEY="xmem_..."');
  console.log('  export XMEM_USERNAME="your_username"');
  console.log('  export XMEM_API_URL="https://api.xmem.in"');
  console.log("\n" + "─".repeat(50));
  console.log("\n✓ Setup complete! Restart OpenCode to activate.\n");
  return 0;
}

async function login(): Promise<number> {
  const existing = loadCredentials();
  if (existing) {
    console.log("Already authenticated. Use 'logout' first to re-authenticate.");
    return 0;
  }

  const result = await startAuthFlow();

  if (result.success) {
    console.log("\n✓ Successfully authenticated with XMem!");
    console.log("Restart OpenCode to activate.\n");
    return 0;
  } else {
    console.error(`\n✗ Authentication failed: ${result.error}`);
    return 1;
  }
}

function logout(): number {
  if (clearCredentials()) {
    console.log("✓ Logged out. Credentials cleared.");
    return 0;
  } else {
    console.log("No credentials found.");
    return 0;
  }
}

function printHelp(): void {
  console.log(`
opencode-xmem - Persistent memory for OpenCode agents

Commands:
  install    Install and configure the plugin
    --no-tui                     Non-interactive mode (for LLM agents)
    --disable-context-recovery   Disable Oh My OpenCode's context hook
  login      Authenticate with XMem (opens browser)
  logout     Clear stored credentials

Examples:
  bunx opencode-xmem@latest install
  bunx opencode-xmem@latest login
  bunx opencode-xmem@latest logout
`);
}

const args = process.argv.slice(2);

if (args.length === 0 || args[0] === "help" || args[0] === "--help" || args[0] === "-h") {
  printHelp();
  process.exit(0);
}

if (args[0] === "install") {
  const noTui = args.includes("--no-tui");
  const disableAutoCompact = args.includes("--disable-context-recovery");
  install({ tui: !noTui, disableAutoCompact }).then((code) => process.exit(code));
} else if (args[0] === "login") {
  login().then((code) => process.exit(code));
} else if (args[0] === "logout") {
  process.exit(logout());
} else {
  console.error(`Unknown command: ${args[0]}`);
  printHelp();
  process.exit(1);
}
