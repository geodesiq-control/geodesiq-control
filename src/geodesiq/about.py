# About geodesiq-control

import importlib
import inspect
import os
import platform
import sys
from pathlib import Path

from ._meta import __author__, __version__

__all__ = ["about"]


def _module_version(module_name: str) -> str:
    """Return module __version__ when available, otherwise 'None'."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "None"
    return str(getattr(module, "__version__", "unknown"))


def _installation_path() -> Path:
    """Resolve the package installation path for reporting."""
    source_file = inspect.getsourcefile(_installation_path)
    if source_file is None:
        return Path(__file__).resolve().parent
    return Path(source_file).resolve().parent


def about() -> None:
    """Print information on geodesiq, key dependencies, and runtime environment."""
    print()
    print("geodesiq: geometric optimal control")
    print("===================================")
    print(f"Authors:            {__author__}")
    print(f"geodesiq Version:   {__version__}")
    print(f"Python Version:     {platform.python_version()} ({sys.implementation.name})")
    print(f"Number of CPUs:     {os.cpu_count()}")
    print(f"Platform Info:      {platform.system()} ({platform.release()}, {platform.machine()})")
    # print(f"Installation path:  {_installation_path()}")

    print("\nCore Dependencies:")
    print(f"Numpy Version:      {_module_version('numpy')}")
    print(f"Scipy Version:      {_module_version('scipy')}")
    print(f"QuTiP Version:      {_module_version('qutip')}")
    print(f"Matplotlib Version: {_module_version('matplotlib')}")

    print()
    print("=" * 50)
    print("Please cite geodesiq in your publication.")
    print("Reference paper: ")
    print("=" * 50)


if __name__ == "__main__":
    about()
