from ._meta import PACKAGE_NAME


class GeodesiQWarning(Warning):
    """Base warning for all package warnings."""

    _prefix = f"[{PACKAGE_NAME}]"

    def __init__(self, message: str = ""):
        super().__init__(f"{self._prefix} {message}" if message else self._prefix)


# ---- Warnings ----
class NumericalStabilityWarning(GeodesiQWarning):
    """Potentially unstable numerical regime detected."""


class PerformanceWarning(GeodesiQWarning):
    """Likely expensive configuration (e.g., very large num_steps)."""


class ExperimentalFeatureWarning(GeodesiQWarning):
    """Feature is experimental and API/behavior may change."""
