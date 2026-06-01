#!/usr/bin/env node

const { spawnSync } = require("node:child_process");

function runNpm(args) {
  return spawnSync("npm", args, {
    encoding: "utf8",
    shell: process.platform === "win32",
  });
}

function printOutput(result) {
  if (result.error) {
    console.error(result.error.message);
  }
  if ((result.stdout || "").trim()) {
    console.log(result.stdout.trim());
  }
  if ((result.stderr || "").trim()) {
    console.error(result.stderr.trim());
  }
}

const whoami = runNpm(["whoami"]);
if (whoami.status !== 0) {
  console.error("[xmem] npm is not logged in. Run: npm login");
  printOutput(whoami);
  process.exit(1);
}

console.log(`[xmem] npm user: ${whoami.stdout.trim()}`);

const profile = runNpm(["profile", "get", "--json"]);
if (profile.status !== 0) {
  const combinedOutput = `${profile.stdout}\n${profile.stderr}`;
  if (combinedOutput.includes("E403")) {
    console.log("[xmem] npm profile is not readable with this token.");
    console.log("[xmem] That is okay for granular publish tokens; continuing package checks.");
  } else {
    console.error("[xmem] Could not read npm profile.");
    printOutput(profile);
    process.exit(1);
  }
} else {
  const profileJson = JSON.parse(profile.stdout);
  console.log(`[xmem] npm email verified: ${profileJson.email_verified}`);
  console.log(`[xmem] npm 2FA enabled: ${profileJson.tfa}`);
  if (!profileJson.tfa) {
    console.log("[xmem] Enable npm 2FA or use a granular publish token before publishing.");
  }
}

const packageView = runNpm(["view", "create-xmem", "name", "version", "--json"]);
if (packageView.status === 0) {
  console.log("[xmem] create-xmem already exists on npm:");
  printOutput(packageView);
  process.exit(0);
}

const combinedOutput = `${packageView.stdout}\n${packageView.stderr}`;
if (combinedOutput.includes("E404")) {
  console.log("[xmem] create-xmem is available on npm.");
  process.exit(0);
}

console.error("[xmem] Could not check create-xmem package availability.");
printOutput(packageView);
process.exit(packageView.status || 1);
