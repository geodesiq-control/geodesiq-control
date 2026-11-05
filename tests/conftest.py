"""Shared pytest configuration."""

import matplotlib
import pytest

# Non-interactive backend: plotting tests must never open windows or need a display.
matplotlib.use("Agg")


def pytest_configure(config):
    """
    Turn unexpected package warnings into errors; expected ones are asserted with ``pytest.warns`` (P-03).

    The filter names the class, because ``warnings.warn(message, category)`` is matched against the message before
    ``GeodesiQWarning`` adds its ``[geodesiq]`` prefix. It is added here and not under ``filterwarnings`` in
    ``pyproject.toml``: resolving the class imports geodesiq, which there would happen before coverage starts.
    """
    config.addinivalue_line("filterwarnings", "error::geodesiq.warnings.GeodesiQWarning")


@pytest.fixture(autouse=True)
def _no_plt_show(monkeypatch):
    """Make ``plt.show()`` a no-op: the plot methods call it by default, and Agg warns that it cannot show."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
