from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os
import sys


def run_welcome_script():
    """Run scripts/welcome.py before dependencies install.

    Uses pure Python stdlib — no Node.js, no pip packages needed.
    Sends user details via HTTP to the Xmem AWS backend.
    """
    installer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "installer")
    welcome_file = os.path.join(installer_dir, "welcome.py")

    if not os.path.exists(welcome_file):
        print("⚠️  src/installer/welcome.py not found — skipping telemetry.")
        return

    try:
        print("\n" + "=" * 50)
        subprocess.run(
            [sys.executable, welcome_file],
            check=True,
            timeout=30,
        )
        print("=" * 50 + "\n")
    except subprocess.TimeoutExpired:
        print("⚠️  Telemetry timed out — continuing installation.")
    except Exception as e:
        print(f"⚠️  Could not run welcome script: {e}")
        print("   (Non-critical — installation will continue.)\n")


class PostDevelopCommand(develop):
    """Post-installation for development mode (pip install -e .)."""

    def run(self):
        run_welcome_script()
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for standard mode (pip install .)."""

    def run(self):
        run_welcome_script()
        install.run(self)


setup(
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)
