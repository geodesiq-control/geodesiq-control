"""Public package API for `geodesiq`."""

from ._meta import __version__
from .about import about
from .controlmodel import ControlModel
from .decompose_hamiltonian import decompose_hamiltonian
from .dynamics import Dynamics
from .exceptions import (
    ComputationError,
    ConfigurationError,
    GeodesiQError,
    ImmutableConfigurationError,
    InvalidControlParameterError,
    IOErrorGeodesiQ,
    MetricComputationError,
    MissingArgsError,
    MissingControlParameterError,
    SolverError,
    ValidationError,
)
from .pulses import PulseControl
from .warnings import ExperimentalFeatureWarning, GeodesiQWarning, NumericalStabilityWarning, PerformanceWarning

__all__ = [
    "ControlModel",
    "PulseControl",
    "Dynamics",
    "about",
    "GeodesiQError",
    "ValidationError",
    "ConfigurationError",
    "ComputationError",
    "SolverError",
    "IOErrorGeodesiQ",
    "MissingControlParameterError",
    "ImmutableConfigurationError",
    "InvalidControlParameterError",
    "MetricComputationError",
    "MissingArgsError",
    "GeodesiQWarning",
    "NumericalStabilityWarning",
    "PerformanceWarning",
    "ExperimentalFeatureWarning",
    "decompose_hamiltonian",
    "__version__",
]
