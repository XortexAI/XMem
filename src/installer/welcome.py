"""
Xmem Clone Telemetry — runs on `pip install -e .`

Collects basic machine info and sends it to the Xmem AWS backend.
The backend validates the auth hash and writes the details into
the 'cloners' collection in MongoDB.

No credentials are stored here — the backend uses its own env vars.
Uses only Python stdlib (no pip dependencies needed).
"""

import hashlib
import json
import os
import platform
import getpass
import socket
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

API_URL = "https://api.xmem.in/telemetry/clone"

# Shared secret — backend verifies against the same hash
AUTH_TOKEN_HASH = hashlib.sha256(b"xmem-clone-telemetry-key").hexdigest()


def collect_user_details() -> dict:
    """Gather non-sensitive machine info."""
    return {
        "username": getpass.getuser(),
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "os": platform.system(),
        "osVersion": platform.version(),
        "arch": platform.machine(),
        "pythonVersion": platform.python_version(),
        "homeDir": os.path.expanduser("~"),
        "cwd": os.getcwd(),
        "clonedAt": datetime.now(timezone.utc).isoformat(),
    }


def report_clone() -> None:
    """Send clone telemetry to the Xmem backend."""
    details = collect_user_details()

    print("\n✨ Welcome to XMem! Registering installation...\n")

    payload = json.dumps(details).encode("utf-8")

    req = Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Clone-Auth": AUTH_TOKEN_HASH,
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                print("✅ Installation registered successfully.\n")
            else:
                print(f"⚠️  Registration returned {resp.status}.")
                print("   (Non-critical — installation will continue.)\n")
    except HTTPError as e:
        print(f"⚠️  Registration returned HTTP {e.code}: {e.reason}")
        print("   (Non-critical — installation will continue.)\n")
    except URLError as e:
        print(f"⚠️  Could not reach telemetry server: {e.reason}")
        print("   (Non-critical — installation will continue.)\n")
    except Exception as e:
        print(f"⚠️  Telemetry error: {e}")
        print("   (Non-critical — installation will continue.)\n")


if __name__ == "__main__":
    report_clone()
