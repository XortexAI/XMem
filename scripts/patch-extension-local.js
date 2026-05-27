#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

function argValue(names, fallback = "") {
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 1) {
    if (names.includes(args[index])) {
      return args[index + 1] || fallback;
    }
  }
  return fallback;
}

const root = path.resolve(__dirname, "..");
const extensionDir = path.resolve(
  argValue(["--extension-dir", "-ExtensionDir"], path.join(root, "repos", "xmem-extension")),
);

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let index = 0; index < 8; index += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    return Math.hypot(px - ax, py - ay);
  }

  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared));
  const x = ax + t * dx;
  const y = ay + t * dy;
  return Math.hypot(px - x, py - y);
}

function writeHollowXIcon(size, outPath) {
  const margin = Math.max(3, Math.round(size * 0.22));
  const outerRadius = Math.max(2, Math.round(size * 0.11));
  const innerRadius = Math.max(1, Math.round(size * 0.052));
  const rows = [];

  for (let y = 0; y < size; y += 1) {
    const row = Buffer.alloc(1 + size * 4);
    row[0] = 0;
    for (let x = 0; x < size; x += 1) {
      const px = x + 0.5;
      const py = y + 0.5;
      const d1 = distanceToSegment(px, py, margin, margin, size - margin, size - margin);
      const d2 = distanceToSegment(px, py, size - margin, margin, margin, size - margin);
      const distance = Math.min(d1, d2);
      const color = distance <= outerRadius && distance > innerRadius ? 0 : 255;
      const offset = 1 + x * 4;
      row[offset] = color;
      row[offset + 1] = color;
      row[offset + 2] = color;
      row[offset + 3] = 255;
    }
    rows.push(row);
  }

  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8;
  header[9] = 6;
  header[10] = 0;
  header[11] = 0;
  header[12] = 0;

  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", zlib.deflateSync(Buffer.concat(rows))),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, png);
}

function patchFile(relativePath, patcher) {
  const file = path.join(extensionDir, relativePath);
  if (!fs.existsSync(file)) {
    return;
  }
  const previous = fs.readFileSync(file, "utf8");
  const next = patcher(previous);
  if (next !== previous) {
    fs.writeFileSync(file, next);
  }
}

function normalizeSource(source) {
  return source
    .replaceAll("https://api.xmem.in", "http://localhost:8000")
    .replaceAll(
      "new XMemClient(API_BASE_URL, config.apiKey, config.userId)",
      "new XMemClient(API_BASE_URL, config.apiKey)",
    )
    .replaceAll(".replace(/[^\\\\w.\\\\-@]+/g, '_')", ".replace(/[^A-Za-z0-9_.@-]+/g, '_')");
}

function ensureApiNormalizeUserId(source) {
  if (!source.includes("function normalizeUserId")) {
    source = source.replace(
      /(const API_BASE_URL = 'http:\/\/localhost:8000';\r?\n)/,
      `$1
function normalizeUserId(userId: string): string {
  const normalized = (userId || '')
    .trim()
    .replace(/[^A-Za-z0-9_.@-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return normalized || 'xmem-local-user';
}
`,
    );
  }

  return source.replaceAll(
    "userId: data.xmem_user_id || '',",
    "userId: normalizeUserId(data.xmem_user_id || ''),",
  );
}

function ensureBackgroundNormalizeUserId(source) {
  if (!source.includes("function normalizeUserId")) {
    source = source.replace(
      /(interface XMemConfig \{\r?\n  apiKey: string;\r?\n  userId: string;\r?\n\}\r?\n)/,
      `$1
function normalizeUserId(userId: string): string {
  const normalized = (userId || '')
    .trim()
    .replace(/[^A-Za-z0-9_.@-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return normalized || 'xmem-local-user';
}
`,
    );
  }

  return source.replaceAll(
    "userId: data.xmem_user_id || '',",
    "userId: normalizeUserId(data.xmem_user_id || ''),",
  );
}

function patchValidateCredentials(source) {
  const replacement = `export async function validateCredentials(apiKey: string, username: string): Promise<boolean> {
  const url = \`\${API_BASE_URL}/auth/verify-key\`;
  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': \`Bearer \${apiKey}\`
      }
    });

    if (!response.ok) {
      console.log('[XMem] Validation failed: HTTP', response.status);
      return false;
    }

    const data = await response.json();
    console.log('[XMem] Validated user data:', data);

    // Local dev static keys do not always map to a real username. If the local
    // API accepted the key, allow any non-empty local user id from the popup.
    if (API_BASE_URL.includes('localhost') || API_BASE_URL.includes('127.0.0.1')) {
      return Boolean(username && username.trim());
    }

    return Boolean(data.username && data.username.toLowerCase() === username.toLowerCase());
  } catch (err) {
    console.error('[XMem] Credential validation network error:', err);
    return false;
  }
}

//`;

  return source.replace(/export async function validateCredentials[\s\S]*?\r?\n}\r?\n\r?\n\/\//, replacement);
}

const apiFile = path.join(extensionDir, "src", "api.ts");
if (!fs.existsSync(apiFile)) {
  throw new Error(`Could not find extension API file at ${apiFile}`);
}

const iconDir = path.join(extensionDir, "icons");
writeHollowXIcon(16, path.join(iconDir, "icon16.png"));
writeHollowXIcon(48, path.join(iconDir, "icon48.png"));
writeHollowXIcon(128, path.join(iconDir, "icon128.png"));
writeHollowXIcon(128, path.join(iconDir, "logo.png"));

for (const file of ["src/api.ts", "src/background.ts", "src/content.ts"]) {
  patchFile(file, normalizeSource);
}

patchFile("src/api.ts", (source) => patchValidateCredentials(ensureApiNormalizeUserId(source)));
patchFile("src/background.ts", ensureBackgroundNormalizeUserId);

console.log("[xmem] Patched extension API for http://localhost:8000");
