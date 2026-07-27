from ._meta import PACKAGE_NAME


class GeodesiQError(Exception):
    """Base exception for all package errors."""

    _prefix = f"[{PACKAGE_NAME}]"

    def __init__(self, message: str = ""):
        super().__init__(f"{self._prefix} {message}" if message else self._prefix)


# ---- Error families ----
class ValidationError(GeodesiQError):
    """Invalid user-provided input values or shapes."""


class ConfigurationError(GeodesiQError):
    """Invalid or incomplete object configuration/state."""


class ComputationError(GeodesiQError):
    """Numerical failures during eigensolve/metric/ODE workflows."""


class SolverError(ComputationError):
    """ODE/eigen solver failed or returned unusable output."""


class IOErrorGeodesiQ(GeodesiQError):
    """Package-level IO/serialization/export failures."""


# ---- Focused leaf errors  ----
class MissingControlParameterError(ConfigurationError):
    """Required control parameter(s) not set."""


class ImmutableConfigurationError(ConfigurationError):
    """Attempt to modify write-once settings (e.g., H_func)."""


class InvalidControlParameterError(ValidationError):
    """Control parameter type/range/index is invalid."""


class MetricComputationError(ComputationError):
    """Metric tensor computation failed (NaN/Inf/singularity)."""


class MissingArgsError(ValidationError):
    """Required argument(s) not provided."""
