#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const DEFAULT_REPO = "https://github.com/XortexAI/XMem.git";
const DEFAULT_BRANCH = "main";

function usage(exitCode = 0) {
  console.log(`Create a local XMem workspace

Usage:
  npx create-xmem@latest
  npx create-xmem@latest my-xmem

Options:
  --repo <url>       XMem git repository URL
  --branch <name>    XMem branch to use
  --help             Show this message

After creation:
  cd xmem
  npm run dev
`);
  process.exit(exitCode);
}

function parseArgs(argv) {
  const options = {
    target: "xmem",
    repo: process.env.XMEM_REPO || DEFAULT_REPO,
    branch: process.env.XMEM_BRANCH || DEFAULT_BRANCH,
  };
  let targetSet = false;

  function readOptionValue(index, name) {
    const value = argv[index + 1];
    if (!value || value.startsWith("-")) {
      console.error(`[create-xmem] ${name} requires a value.`);
      usage(1);
    }
    return value;
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--help" || arg === "-h") {
      usage(0);
    }

    if (arg === "--repo") {
      options.repo = readOptionValue(index, arg);
      index += 1;
      continue;
    }

    if (arg === "--branch") {
      options.branch = readOptionValue(index, arg);
      index += 1;
      continue;
    }

    if (arg.startsWith("-")) {
      console.error(`[create-xmem] Unknown option: ${arg}`);
      usage(1);
    }

    if (!targetSet) {
      options.target = arg;
      targetSet = true;
      continue;
    }

    console.error(`[create-xmem] Unexpected extra argument: ${arg}`);
    usage(1);
  }

  if (!options.repo || !options.branch) {
    console.error("[create-xmem] --repo and --branch require values.");
    usage(1);
  }

  return options;
}

function runGit(args, cwd) {
  const result = spawnSync("git", args, {
    cwd,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    console.error(`[create-xmem] Git is required: ${result.error.message}`);
    console.error("[create-xmem] Install Git, reopen your terminal, and run the command again.");
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function assertCleanTarget(targetPath) {
  if (!fs.existsSync(targetPath)) {
    return;
  }

  const entries = fs.readdirSync(targetPath);
  if (entries.length > 0) {
    console.error(`[create-xmem] Target folder is not empty: ${targetPath}`);
    console.error("[create-xmem] Choose a new folder name or empty the existing folder.");
    process.exit(1);
  }
}

function removeGitMetadata(targetPath) {
  fs.rmSync(path.join(targetPath, ".git"), {
    recursive: true,
    force: true,
  });
}

const options = parseArgs(process.argv.slice(2));
const targetPath = path.resolve(process.cwd(), options.target);

assertCleanTarget(targetPath);

console.log(`[create-xmem] Creating XMem workspace in ${targetPath}`);
runGit(["clone", "--depth", "1", "--branch", options.branch, options.repo, targetPath], process.cwd());
removeGitMetadata(targetPath);

console.log("");
console.log("[create-xmem] Created local XMem workspace.");
console.log("");
console.log("Next:");
console.log(`  cd ${path.relative(process.cwd(), targetPath) || "."}`);
console.log("  npm run dev");
console.log("");
console.log("Chrome extension after setup:");
console.log("  Load unpacked: repos/xmem-extension/dist");
