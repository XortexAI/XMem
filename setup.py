from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os
import sys

def run_welcome_script():
    """Run src/installer/welcome.py.
    
    This is called at the top level to ensure it runs during the 
    metadata generation phase of `pip install`.
    """
    # Prevent running multiple times in the same session (pip often calls setup.py twice)
    if os.environ.get('XMEM_INSTALL_RUNNING') == '1':
        return
    os.environ['XMEM_INSTALL_RUNNING'] = '1'

    installer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "installer")
    welcome_file = os.path.join(installer_dir, "welcome.py")

    if not os.path.exists(welcome_file):
        return

    try:
        # Use stdout directly to bypass some pip output capturing
        subprocess.run(
            [sys.executable, welcome_file],
            check=False,
            timeout=20,
        )
    except Exception:
        pass

# Run it immediately when setup.py is loaded
run_welcome_script()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)

class PostInstallCommand(install):
    def run(self):
        install.run(self)

setup(
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)
