"""Report geodesiq version, dependency versions and runtime environment information."""

import importlib
import os
import platform
import sys

from ._meta import __author__, __version__

__all__ = ["about"]

_REPOSITORY_URL = "https://github.com/geodesiq-control/geodesiq-control"


def _module_version(module_name: str) -> str:
    """Return module __version__ when available, otherwise 'None'."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "None"
    return str(getattr(module, "__version__", "unknown"))


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

    print("\nCore Dependencies:")
    print(f"Numpy Version:      {_module_version('numpy')}")
    print(f"Scipy Version:      {_module_version('scipy')}")
    print(f"QuTiP Version:      {_module_version('qutip')}")
    print(f"Matplotlib Version: {_module_version('matplotlib')}")

    print()
    print("=" * 50)
    print("Please cite geodesiq in your publication.")
    print(f"See: {_REPOSITORY_URL}#citing-geodesiq")
    print("=" * 50)


if __name__ == "__main__":
    about()
