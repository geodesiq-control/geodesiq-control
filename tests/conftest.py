"""Shared pytest configuration."""

import matplotlib

# Non-interactive backend: plotting tests must never open windows or need a display.
matplotlib.use("Agg")
