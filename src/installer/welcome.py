"""
Xmem Clone Telemetry — runs on `pip install -e .`

Collects basic machine info and sends it to the Xmem AWS backend.
The backend validates the auth hash and writes the details into
the 'cloners' collection in MongoDB.

No credentials are stored here — the backend uses its own env vars.
Uses only Python stdlib (no pip dependencies needed).
"""

import ctypes
import hashlib
import json
import os
import platform
import getpass
import socket
import ssl
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

API_URL = "https://api.xmem.in/telemetry/clone"

AUTH_TOKEN_HASH = hashlib.sha256(b"xmem-clone-telemetry-key").hexdigest()

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30), "IST")


def get_full_name() -> str:
    """Try to get the user's full/display name from the OS."""
    
    # Windows: use GetUserNameExW via ctypes
    if sys.platform == "win32":
        try:
            GetUserNameExW = ctypes.windll.secur32.GetUserNameExW
            NameDisplay = 3  # EXTENDED_NAME_FORMAT.NameDisplay
            
            size = ctypes.pointer(ctypes.c_ulong(0))
            GetUserNameExW(NameDisplay, None, size)
            
            name_buffer = ctypes.create_unicode_buffer(size.contents.value)
            GetUserNameExW(NameDisplay, name_buffer, size)
            
            full_name = name_buffer.value.strip()
            if full_name:
                return full_name
        except Exception:
            pass
        
        # Fallback: try net user command
        try:
            username = getpass.getuser()
            result = subprocess.run(
                ["net", "user", username],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip().startswith("Full Name"):
                        full_name = line.split(None, 2)[-1].strip()
                        if full_name:
                            return full_name
        except Exception:
            pass
    
    # macOS/Linux: try pwd module or /etc/passwd
    else:
        try:
            import pwd
            pw_entry = pwd.getpwuid(os.getuid())
            # GECOS field often contains full name (before first comma)
            gecos = pw_entry.pw_gecos
            if gecos:
                full_name = gecos.split(",")[0].strip()
                if full_name:
                    return full_name
        except Exception:
            pass
        
        # Fallback for macOS: use id -F
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["id", "-F"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    full_name = result.stdout.strip()
                    if full_name:
                        return full_name
            except Exception:
                pass
    
    return ""


def collect_user_details() -> dict:
    """Gather non-sensitive machine info."""
    details = {}
    
    try:
        details["username"] = getpass.getuser()
    except Exception:
        details["username"] = "unknown"
    
    # Try to get full/display name
    full_name = get_full_name()
    if full_name:
        details["fullName"] = full_name
    
    try:
        details["hostname"] = socket.gethostname()
    except Exception:
        details["hostname"] = "unknown"
    
    details["platform"] = sys.platform
    details["os"] = platform.system()
    details["osVersion"] = platform.version()
    details["arch"] = platform.machine()
    details["pythonVersion"] = platform.python_version()
    details["homeDir"] = os.path.expanduser("~")
    details["cwd"] = os.getcwd()
    details["clonedAt"] = datetime.now(IST).isoformat()
    
    return details


def safe_print(msg: str) -> None:
    """Print with fallback for Windows encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback: replace problematic chars with ASCII equivalents
        ascii_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(ascii_msg)


def report_clone() -> None:
    """Send clone telemetry to the Xmem backend."""
    safe_print("\n[*] Welcome to XMem!\n")
    
    # Collect details
    safe_print("[xmem] Collecting system info...")
    details = collect_user_details()
    user_info = details['username']
    if details.get('fullName'):
        user_info = f"{details['fullName']} ({details['username']})"
    safe_print(f"[xmem] User: {user_info}@{details['hostname']}")
    safe_print(f"[xmem] System: {details['os']} {details['arch']} (Python {details['pythonVersion']})")
    
    # Prepare request
    safe_print("[xmem] Sending telemetry...")
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
            status = resp.status
            if status in (200, 201):
                safe_print("[xmem] [OK] Registered successfully!\n")
            else:
                safe_print(f"[xmem] [FAIL] Unexpected response: {status}\n")
                
    except HTTPError as e:
        safe_print(f"[xmem] [FAIL] HTTP Error: {e.code} {e.reason}")
        try:
            error_body = e.read().decode("utf-8", errors="replace")
            safe_print(f"[xmem] Response: {error_body[:200]}")
        except Exception:
            pass
            
    except URLError as e:
        safe_print(f"[xmem] [FAIL] Connection Error: {e.reason}")
        
    except ssl.SSLError as e:
        safe_print(f"[xmem] [FAIL] SSL Error: {e}")
        safe_print("[xmem] On Mac, run: /Applications/Python*/Install\\ Certificates.command")
        
    except socket.timeout:
        safe_print("[xmem] [FAIL] Request timed out (15s)")
        
    except Exception as e:
        safe_print(f"[xmem] [FAIL] Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    report_clone()