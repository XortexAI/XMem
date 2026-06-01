import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { execFile } from "node:child_process";
import { join } from "node:path";
import { homedir } from "node:os";

const CREDENTIALS_DIR = join(homedir(), ".xmem-opencode");
const CREDENTIALS_FILE = join(CREDENTIALS_DIR, "credentials.json");
const AUTH_PORT = 19878;
const AUTH_BASE_URL = process.env.XMEM_AUTH_URL || "https://xmem.in/auth/connect";
const CLIENT_NAME = "opencode";

export interface Credentials {
  apiKey: string;
  apiUrl: string;
  username: string;
  createdAt: string;
}

export function loadCredentials(): Credentials | null {
  if (!existsSync(CREDENTIALS_FILE)) return null;
  try {
    const content = readFileSync(CREDENTIALS_FILE, "utf-8");
    return JSON.parse(content) as Credentials;
  } catch {
    return null;
  }
}

export function saveCredentials(credentials: Omit<Credentials, "createdAt">): void {
  mkdirSync(CREDENTIALS_DIR, { recursive: true, mode: 0o700 });
  const stored: Credentials = {
    ...credentials,
    createdAt: new Date().toISOString(),
  };
  writeFileSync(CREDENTIALS_FILE, JSON.stringify(stored, null, 2), { mode: 0o600 });
}

export function clearCredentials(): boolean {
  if (!existsSync(CREDENTIALS_FILE)) return false;
  rmSync(CREDENTIALS_FILE);
  return true;
}

function openBrowser(url: string): void {
	const platform = process.platform;

	const command = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
	const args = platform === "win32" ? ["/c", "start", "", url] : [url];
	execFile(command, args, (err) => {
		if (err) console.error("Failed to open browser:", err.message);
	});
}

export interface AuthResult {
  success: boolean;
  credentials?: Credentials;
  error?: string;
}

export function startAuthFlow(timeoutMs = 120000): Promise<AuthResult> {
  return new Promise((resolve) => {
    let resolved = false;

    const server = createServer((req: IncomingMessage, res: ServerResponse) => {
      if (resolved) return;

      const url = new URL(req.url || "/", `http://localhost:${AUTH_PORT}`);

      if (url.pathname === "/callback") {
        const apiKey = url.searchParams.get("apikey");
        const username = url.searchParams.get("username");
        const apiUrl = url.searchParams.get("apiurl") || "https://api.xmem.in";

        if (apiKey && username) {
          const credentials = { apiKey, apiUrl, username, createdAt: new Date().toISOString() };
          saveCredentials({ apiKey, apiUrl, username });
          res.writeHead(200, { "Content-Type": "text/html" });
          res.end(`
            <!DOCTYPE html>
            <html>
            <head><title>Success</title></head>
            <body style="font-family: system-ui; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #0a0a0a; color: #fafafa;">
              <div style="text-align: center;">
                <h1 style="color: #22c55e;">✓ Connected!</h1>
                <p>XMem is now connected to OpenCode. You can close this window and return to your terminal.</p>
              </div>
            </body>
            </html>
          `);
          resolved = true;
          server.close();
          resolve({ success: true, credentials });
        } else {
          res.writeHead(400, { "Content-Type": "text/html" });
          res.end(`
            <!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body style="font-family: system-ui; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #0a0a0a; color: #fafafa;">
              <div style="text-align: center;">
                <h1 style="color: #ef4444;">✗ Connection Failed</h1>
                <p>Missing API key or username. Please try again.</p>
              </div>
            </body>
            </html>
          `);
          resolved = true;
          server.close();
          resolve({ success: false, error: "Missing API key or username" });
        }
      } else {
        res.writeHead(404);
        res.end("Not Found");
      }
    });

    server.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "EADDRINUSE") {
        resolve({ success: false, error: `Port ${AUTH_PORT} is already in use` });
      } else {
        resolve({ success: false, error: err.message });
      }
    });

    server.listen(AUTH_PORT, () => {
      const callbackUrl = `http://localhost:${AUTH_PORT}/callback`;
      const authUrl = `${AUTH_BASE_URL}?callback=${encodeURIComponent(callbackUrl)}&client=${CLIENT_NAME}`;

      console.log("Opening browser for XMem authentication...");
      console.log(`If it doesn't open, visit: ${authUrl}`);
      openBrowser(authUrl);
    });

    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        server.close();
        resolve({ success: false, error: "Authentication timed out" });
      }
    }, timeoutMs);
  });
}
