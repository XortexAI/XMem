import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import assert from "node:assert/strict";

test("installer writes config without copying secret values", () => {
  const root = mkdtempSync(join(tmpdir(), "xmem-hermes-"));
  const secret = "test_secret_should_not_be_written";
  try {
    const result = spawnSync(process.execPath, ["src/cli.js", "install", "--config-root", root], {
      cwd: process.cwd(),
      env: { ...process.env, XMEM_API_KEY: secret },
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);

    const doctor = spawnSync(process.execPath, ["src/cli.js", "doctor", "--config-root", root], {
      cwd: process.cwd(),
      env: { ...process.env, XMEM_API_KEY: secret },
      encoding: "utf8",
    });
    assert.equal(doctor.status, 0, doctor.stderr);

    for (const file of [".hermes/config.yaml", "HERMES.md"]) {
      const content = readFileSync(join(root, file), "utf8");
      assert.ok(!content.includes(secret), file + " leaked secret");
      assert.ok(
        content.includes("XMem") || content.includes("xmem") || content.includes("XMEM"),
        file + " should mention XMem",
      );
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("installer refuses to overwrite existing files unless forced", () => {
  const root = mkdtempSync(join(tmpdir(), "xmem-hermes-"));
  try {
    const first = spawnSync(process.execPath, ["src/cli.js", "install", "--config-root", root], {
      cwd: process.cwd(),
      encoding: "utf8",
    });
    assert.equal(first.status, 0, first.stderr);

    const second = spawnSync(process.execPath, ["src/cli.js", "install", "--config-root", root], {
      cwd: process.cwd(),
      encoding: "utf8",
    });
    assert.notEqual(second.status, 0);
    assert.match(second.stderr, /Refusing to overwrite/);

    const forced = spawnSync(process.execPath, ["src/cli.js", "install", "--config-root", root, "--force"], {
      cwd: process.cwd(),
      encoding: "utf8",
    });
    assert.equal(forced.status, 0, forced.stderr);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("installer reports missing option values", () => {
  const result = spawnSync(process.execPath, ["src/cli.js", "install", "--api-url"], {
    cwd: process.cwd(),
    encoding: "utf8",
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /--api-url requires a value/);
});
