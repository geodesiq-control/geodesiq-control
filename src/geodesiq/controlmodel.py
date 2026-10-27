import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.differentiate import jacobian
from scipy.integrate import romb, solve_ivp
from scipy.interpolate import PchipInterpolator

from ._utils import (
    DiabaticList,
    Flags,
    build_dia_list,
    crossing_map,
    degenerate_clusters,
    follow_state,
    is_hermitian,
    limit_blas_threads,
    rotate_degenerate_subspaces,
    stack_samples,
    traceless,
    uniform_grid_derivative,
    validate_state_index,
    values_equal,
)
from .exceptions import (
    ConfigurationError,
    ImmutableConfigurationError,
    InvalidControlParameterError,
    MetricComputationError,
    MissingControlParameterError,
    SolverError,
    ValidationError,
)
from .pulses import PulseControl
from .warnings import NumericalStabilityWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


@dataclass(frozen=True)
class _EigensystemParameters:
    control_name: str | None
    pulse_initial: float
    pulse_final: float
    num_steps: int


@dataclass(frozen=True)
class _ControlParameters(_EigensystemParameters):
    initial_state: int
    final_state: int
    final_state_explicit: bool
    alpha: float
    beta: float
    dia_alpha: float | None
    dia_beta: float | None


def _figure_and_axes(fig: "Figure | SubFigure | None", ax: "Axes | None") -> "tuple[Figure | SubFigure, Axes]":
    """
    Return the figure and axis to plot on, creating whichever is missing.

    A given ``ax`` takes precedence and its parent figure is returned. Any other type of ``fig`` or ``ax`` raises
    ``ValidationError``, instead of a silent ``None`` result (e.g. for a figure passed positionally as ``ax``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

    if ax is not None:
        if not isinstance(ax, Axes):
            raise ValidationError(f"ax must be a matplotlib Axes or None, got {type(ax).__name__}.")
        fig = ax.figure
    elif fig is None:
        fig, ax = plt.subplots()
    elif isinstance(fig, (Figure, SubFigure)):
        ax = fig.add_subplot(111)
    else:
        raise ValidationError(f"fig must be a matplotlib Figure, SubFigure or None, got {type(fig).__name__}.")
    assert fig is not None
    return fig, ax


class ControlModel:
    """
    A class to represent a parameter-dependent ControlModel and solve the optimization problem for finding the optimal
    control pulse. The ControlModel is defined as a function of a control parameter (e.g., lambda) and can also depend
    on other parameters. The class provides methods to set the parameters, compute the metric tensor, solve the ODE
    for the control pulse, and synthesize the control pulse based on the solution of the optimization problem.
    """

    # All tolerances are relative, so the results do not depend on the units of the Hamiltonian.
    # An energy gap below _GAP_RTOL * (spectral bandwidth) is treated as a degeneracy.
    _GAP_RTOL = 1e-12
    # Two states are not coupled (and may cross truly) if their matrix element of dH/dx is at most _CROSSING_RTOL times
    # the largest one at that grid point.
    _CROSSING_RTOL = 1e-8
    # Metric values below _METRIC_RTOL * max(metric) are treated as (numerical) zeros.
    _METRIC_RTOL = 1e-20
    # Default number of samples of the normalized pulse when solve_problem() is first called without one.
    _DEFAULT_PULSE_ACCURACY = 1000
    # Numerical dH/dx: finite differences of order _FD_ORDER on the already sampled control grid. Points where the
    # difference with the order (_FD_ORDER - 2) estimate exceeds _FD_RTOL * max|dH/dx| are recomputed with the
    # adaptive scipy.differentiate.jacobian, as are all points of grids with fewer than _FD_MIN_POINTS samples.
    # Both tolerances are floored at _FD_ROUNDOFF_FACTOR * eps * max|x| / step, the relative rounding error of the
    # control steps, which dominates when the control values carry a large offset (e.g. x = 1e6 + [0, 1]). Only the
    # traceless part of H is differentiated, and the absolute tolerances are also floored at
    # _FD_ROUNDOFF_FACTOR * eps * max|H| / step, the rounding error that a large identity offset (e.g. H = 1e8 I + ...)
    # leaves in the samples, which no smaller step can resolve.
    _FD_ORDER = 8
    _FD_RTOL = 1e-8
    _FD_MIN_POINTS = 5
    _FD_ROUNDOFF_FACTOR = 100.0
    # The matrix elements |<m|dH/dx|n>| are computed in chunks of grid points whose complex (chunk, dim, dim)
    # temporaries take about this many bytes, so the products do not need further full-size stacks.
    _MATRIX_ELEMENT_CHUNK_BYTES = 2**22
    # A custom ODE solver's pulse x(tau) may deviate from the exact geodesic by at most _SOLVER_TAU_TOL in normalized
    # time tau, i.e. |tau(x(tau_i)) - tau_i| <= _SOLVER_TAU_TOL at every sample
    # (solve_ivp at its default rtol=1e-3 stays below about 5e-3).
    _SOLVER_TAU_TOL = 1e-2

    def __init__(
        self,
        H_func: Callable[..., ArrayLike] | None = None,
        partial_H_func: Callable[..., ArrayLike] | None = None,
        *,
        H_d: ArrayLike | None = None,
        H_c: ArrayLike | None = None,
        _flags_verbose: bool = False,
    ) -> None:
        """
        Initialize the ControlModel from either a Hamiltonian function or constant drift and control matrices.

        Parameters
        ----------
        H_func : Callable[..., ArrayLike] | None
            A function that takes the control parameter and other parameters as input and returns the Hamiltonian
            matrix as a numpy array (or any array-like, converted with ``np.asarray``). The function should be defined
            such that it can accept the control parameter as a keyword argument, e.g.,
            H_func(lambda=..., param1=..., param2=..., ...). Provide either ``H_func`` or the pair ``H_d`` and ``H_c``.
        partial_H_func : Callable[..., ArrayLike] | None
            An optional function that takes the control parameter and other parameters as input and returns the partial
            derivative of the Hamiltonian with respect to the control parameter as a numpy array.
            If this function is not provided, the class will compute the numerical partial derivative.
        H_d, H_c : ArrayLike | None
            Constant drift and control Hamiltonians defining the affine model
            ``H(control) = H_d + control * H_c``. Both matrices must be finite, Hermitian, square, and have the same
            shape. In this mode the derivative is exactly ``H_c`` and ``set_parameters`` is unavailable.
        _flags_verbose : bool
            Private debugging switch that prints every change of the cached-stage flags. Not part of the public API.
        """
        affine_requested = H_d is not None or H_c is not None

        if affine_requested:
            if H_d is None or H_c is None:
                raise ValidationError("Both H_d and H_c must be provided for an affine Hamiltonian.")
            if H_func is not None or partial_H_func is not None:
                raise ValidationError("Provide either H_func/partial_H_func or H_d/H_c, not both.")

            drift = self._validate_constant_hamiltonian(H_d, "H_d")
            control = self._validate_constant_hamiltonian(H_c, "H_c")
            if control.shape != drift.shape:
                raise ValidationError("H_c must have the same shape as H_d.")

            dtype = np.result_type(drift.dtype, control.dtype)
            self._H_d: np.ndarray | None = np.asarray(drift, dtype=dtype)
            self._H_c: np.ndarray | None = np.asarray(control, dtype=dtype)
            self._H_func: Callable[..., ArrayLike] | None = None
            self._partial_H_func: Callable[..., ArrayLike] | None = None
            self._affine_hamiltonian = True
            self._hamiltonian_dimension: int | None = drift.shape[0]
        else:
            if H_func is None or not callable(H_func):
                raise ValidationError("H_func must be callable when H_d and H_c are not provided.")
            if partial_H_func is not None and not callable(partial_H_func):
                raise ValidationError("partial_H_func must be callable when provided.")

            self._H_func = H_func
            self._partial_H_func = partial_H_func
            self._H_d = None
            self._H_c = None
            self._affine_hamiltonian = False
            self._hamiltonian_dimension = None

        # Initialize flags needed to track the state of the computations
        self._flags = Flags(_verbose=_flags_verbose)
        self._flags.add("eigenproblem_solved")
        self._flags.add("dia_list_computed", parent="eigenproblem_solved")
        self._flags.add("metric_computed", parents=["eigenproblem_solved", "dia_list_computed"])
        self._flags.add("ode_solved", parent="metric_computed")

        self._flag_numerical_partial_H = not self._affine_hamiltonian and self._partial_H_func is None

        # Initialize parameters and control settings
        self._parameters: dict[str, Any] = {}
        self._control_name: str | None = None
        self._pulse_initial: float | None = None
        self._pulse_final: float | None = None
        self._initial_state: int | None = None
        self._final_state: int | None = None
        # Until final_state is given explicitly, it follows initial_state (adiabatic transfer).
        self._final_state_explicit = False
        self._alpha: float | None = None
        self._beta: float | None = None
        self._dia_alpha: float | None = None
        self._dia_beta: float | None = None
        self._num_steps: int | None = None

        # Initialize energy gaps and matrix elements to None (to be computed in self.solve_problem())
        self._centered_energies: np.ndarray | None = None
        self._energy_offset: np.ndarray | None = None
        self._matrix_elements: np.ndarray | None = None
        # Energy index changes of the states at true level crossings (see _utils.crossing_map).
        self._state_map: np.ndarray | None = None
        # Energy index of the occupied state along the grid, and the index it ends in.
        self._occupied_states: np.ndarray | None = None
        self._resolved_final_state: int | None = None

        # Initialize metric tensor and normalization factor to None (to be computed in self.solve_problem())
        self._dia_list: DiabaticList | None = None
        self._metric_tensor: np.ndarray | None = None
        self._a_tilde: float | None = None

        # Initialize pulse parameters
        self._tau: np.ndarray | None = None
        self._control_pulse: np.ndarray | None = None
        self._control_sol: np.ndarray | None = None

        # Numerical integration configuration (user-overridable in solve_problem).
        self._solver = solve_ivp
        self._solver_kwargs: dict[str, Any] = {}
        self._metric_integrator = romb
        self._metric_integrator_kwargs: dict[str, Any] = {}
        self._previous_pulse_accuracy: int | None = None  # To track changes in pulse accuracy for ODE solving

    @property
    def hamiltonian_dimension(self) -> int | None:
        """Dimension of the Hilbert space, or None before the Hamiltonian has been evaluated."""
        return self._hamiltonian_dimension

    def _call_hamiltonian(self, *args: Any, **kwargs: Any) -> np.ndarray:
        H_func = self.H_func
        if H_func is None:
            raise ImmutableConfigurationError(
                "Direct H_func evaluation is unavailable when H_d and H_c define the Hamiltonian."
            )
        return self._validate_hamiltonian(np.asarray(H_func(*args, **{**self._parameters, **kwargs})))

    def _validate_hamiltonian(self, matrix: np.ndarray) -> np.ndarray:
        """Check a matrix returned by H_func, and record the Hamiltonian dimension on the first call."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValidationError(f"H_func must return a 2D square matrix, got shape {matrix.shape}.")

        dimension = matrix.shape[0]

        if self._hamiltonian_dimension is None:
            self._hamiltonian_dimension = dimension
        elif dimension != self._hamiltonian_dimension:
            raise ValidationError(
                "The Hamiltonian dimension cannot change after ControlModel "
                f"initialization: expected {self._hamiltonian_dimension}, "
                f"got {dimension}."
            )

        if not np.all(np.isfinite(matrix)):  # Non-finite values
            raise ValidationError("H_func must return a matrix with finite values.")

        if not is_hermitian(matrix):
            raise ValidationError("H_func must return a Hermitian matrix.")

        return matrix

    def _call_partial_hamiltonian(self, *args: Any, **kwargs: Any) -> np.ndarray:
        partial_H_func = self.partial_H_func
        if partial_H_func is None:
            raise MissingControlParameterError("partial_H_func is not configured.")
        matrix = np.asarray(partial_H_func(*args, **{**self._parameters, **kwargs}))

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValidationError(f"partial_H_func must return a 2D square matrix, got shape {matrix.shape}.")

        if not np.all(np.isfinite(matrix)):  # Non-finite values
            raise ValidationError("partial_H_func must return a matrix with finite values.")

        if not is_hermitian(matrix):
            raise ValidationError("partial_H_func must return a Hermitian matrix.")

        if self._hamiltonian_dimension is None:
            # Validated call: H_func may return e.g. a nested list, and this also records the dimension.
            dim_temp = self._call_hamiltonian(*args, **kwargs).shape[0]
        else:
            dim_temp = self._hamiltonian_dimension

        if np.shape(matrix)[0] != dim_temp:
            raise ValidationError("partial_H_func must return a matrix with the same shape as H_func.")

        return matrix

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        # Return the ControlModel function if the object is called directly, allowing for easy evaluation of the
        #   ControlModel at specific control values
        # Runtime kwargs override stored defaults for explicit one-off evaluations.
        if self._affine_hamiltonian:
            if len(args) == 1 and not kwargs:
                return self.evaluate_hamiltonian(args[0])

            control_name = self.control_name
            if not args and control_name is not None and set(kwargs) == {control_name}:
                return self.evaluate_hamiltonian(kwargs[control_name])

            raise InvalidControlParameterError(
                "Affine models must be evaluated with one positional control value or with the configured "
                "control_name as the only keyword argument."
            )

        return self._call_hamiltonian(*args, **kwargs)

    def _evaluation_kwargs(self, control_value: float) -> dict[str, Any]:
        control_name = self.control_name
        if control_name is None:
            raise MissingControlParameterError("control_name must be set before evaluating the Hamiltonian.")

        if not isinstance(control_value, (int, float, np.integer, np.floating)) or isinstance(control_value, bool):
            raise InvalidControlParameterError("Control value must be a finite real number.")
        if not np.isfinite(control_value):
            raise InvalidControlParameterError("Control value must be finite.")

        return {**self._parameters, control_name: float(control_value)}

    def evaluate_hamiltonian(self, control_value: float) -> np.ndarray:
        """Evaluate and validate the Hamiltonian at one control value.

        Parameters
        ----------
        control_value : float
            Finite real value of the control parameter. Callable models receive it
            under ``control_name``, together with the stored fixed parameters.

        Returns
        -------
        np.ndarray
            Hamiltonian matrix of shape ``(dimension, dimension)``. For affine models
            this is ``H_d + control_value * H_c``.

        Raises
        ------
        InvalidControlParameterError
            If ``control_value`` is not a finite real number.
        MissingControlParameterError
            If a callable model has no configured ``control_name``.
        ValidationError
            If a callable returns a non-square, non-finite or non-Hermitian matrix,
            or changes the Hamiltonian dimension after it has been established.
        """

        control_value = self._check_control_value(control_value)

        if self._affine_hamiltonian:
            assert self._H_d is not None
            assert self._H_c is not None
            return self._H_d + control_value * self._H_c

        return self._call_hamiltonian(**self._evaluation_kwargs(control_value))

    def _frozen_hamiltonian(self) -> Callable[[float], np.ndarray]:
        """
        Evaluator of the current Hamiltonian that later changes to the model do not affect (used by ``Dynamics``).

        It takes copies of ``H_d`` and ``H_c``, or of the fixed parameters and ``control_name`` of a callable model,
        and validates like ``evaluate_hamiltonian``. ``H_func`` itself cannot be replaced; state that it keeps
        internally (e.g. in a closure) is not copied.
        """
        if self._affine_hamiltonian:
            if self._H_d is None or self._H_c is None:
                raise ConfigurationError("Affine ControlModel is missing its constant drift or control Hamiltonian.")
            H_d, H_c = self._H_d.copy(), self._H_c.copy()

            def evaluate_affine(control_value: float) -> np.ndarray:
                return H_d + self._check_control_value(control_value) * H_c

            return evaluate_affine

        H_func = self._H_func
        if H_func is None:
            raise MissingControlParameterError("H_func is not configured.")
        kwargs = self._evaluation_kwargs(0.0)  # Checks that control_name is set.
        control_name = self.control_name
        assert control_name is not None
        kwargs = deepcopy(kwargs)

        def evaluate_callable(control_value: float) -> np.ndarray:
            kwargs[control_name] = self._check_control_value(control_value)
            return self._validate_hamiltonian(np.asarray(H_func(**kwargs)))

        return evaluate_callable

    @staticmethod
    def _check_control_value(control_value: float) -> float:
        if not isinstance(control_value, (int, float, np.integer, np.floating)) or isinstance(control_value, bool):
            raise InvalidControlParameterError("Control value must be a finite real number.")
        control_value = float(control_value)
        if not np.isfinite(control_value):
            raise InvalidControlParameterError("Control value must be finite.")
        return control_value

    # Setters and getters are defined for the critical attributes of the ControlModel class, so we have control over how
    # the user modifies them. Every control setter delegates to set_control(), which validates the full configuration
    # and updates the flags in a single place. Assigning None keeps the previous value.
    @property
    def H_func(self) -> Callable[..., ArrayLike] | None:
        """Hamiltonian function, or None for affine models defined by H_d and H_c."""
        return self._H_func

    @H_func.setter
    def H_func(self, func: Callable[..., ArrayLike]) -> None:
        if self._affine_hamiltonian:
            raise ImmutableConfigurationError("H_func cannot be set when the Hamiltonian is defined by H_d and H_c.")
        if self._H_func is None:
            if not callable(func):
                raise ValidationError("H_func must be callable.")
            self._H_func = func
        else:
            raise ImmutableConfigurationError(
                "H_func is already set and cannot be changed. If you want to change it,"
                " please create a new instance of the ControlModel class."
            )

    @property
    def partial_H_func(self) -> Callable[..., ArrayLike] | None:
        """Analytical derivative of the Hamiltonian, or None when the numerical derivative is used."""
        return self._partial_H_func

    @partial_H_func.setter
    def partial_H_func(self, func: Callable[..., ArrayLike]) -> None:
        if self._affine_hamiltonian:
            raise ImmutableConfigurationError(
                "partial_H_func cannot be set when the Hamiltonian is defined by H_d and H_c."
            )
        if self._partial_H_func is None:
            if not callable(func):
                raise ValidationError("partial_H_func must be callable.")
            self._partial_H_func = func
            self._flag_numerical_partial_H = False  # Update the numerical partial flag
            self._flags["eigenproblem_solved"] = False  # Reset the eigenproblem solved flag
        else:
            raise ImmutableConfigurationError(
                "partial_H_func is already set and cannot be changed. If you want to change it,"
                " please create a new instance of the ControlModel class."
            )

    @property
    def H_d(self) -> np.ndarray | None:
        """Return a copy of the constant drift Hamiltonian, if configured."""
        return None if self._H_d is None else self._H_d.copy()

    @property
    def H_c(self) -> np.ndarray | None:
        """Return a copy of the constant control Hamiltonian, if configured."""
        return None if self._H_c is None else self._H_c.copy()

    @property
    def affine_hamiltonian(self) -> bool:
        """Whether the model uses ``H(control) = H_d + control * H_c``."""
        return self._affine_hamiltonian

    @property
    def control_name(self) -> str | None:
        """Name of the control parameter in the Hamiltonian function."""
        return self._control_name

    @control_name.setter
    def control_name(self, value: str | None) -> None:
        self.set_control(control_name=value)

    @property
    def pulse_initial(self) -> float | None:
        """Value of the control parameter at the start of the pulse."""
        return self._pulse_initial

    @pulse_initial.setter
    def pulse_initial(self, value: float | None) -> None:
        self.set_control(pulse_initial=value)

    @property
    def pulse_final(self) -> float | None:
        """Value of the control parameter at the end of the pulse."""
        return self._pulse_final

    @pulse_final.setter
    def pulse_final(self, value: float | None) -> None:
        self.set_control(pulse_final=value)

    @property
    def initial_state(self) -> int | None:
        """Index (in increasing energy, at ``pulse_initial``) of the initial eigenstate."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: int | None) -> None:
        self.set_control(initial_state=value)

    @property
    def final_state(self) -> int | None:
        """
        Index (in increasing energy, at ``pulse_final``) of the target eigenstate.

        As long as it has not been set explicitly, it is the index in which the state followed adiabatically from
        ``initial_state`` ends. Before ``solve_problem()``, and without true level crossings, this is
        ``initial_state``.
        """
        resolved = self._resolved_final_state
        if not self._final_state_explicit and self._flags["dia_list_computed"] and resolved is not None:
            return resolved
        return self._final_state

    @final_state.setter
    def final_state(self, value: int | None) -> None:
        self.set_control(final_state=value)

    @property
    def alpha(self) -> float | None:
        """Exponent of the energy gaps in the adiabatic contribution to the metric."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | None) -> None:
        self.set_control(alpha=value)

    @property
    def beta(self) -> float | None:
        """Exponent of the matrix elements in the adiabatic contribution to the metric."""
        return self._beta

    @beta.setter
    def beta(self, value: float | None) -> None:
        self.set_control(beta=value)

    @property
    def dia_alpha(self) -> float | None:
        """Exponent of the energy gaps in the diabatic contribution to the metric."""
        return self._dia_alpha

    @dia_alpha.setter
    def dia_alpha(self, value: float | None) -> None:
        self.set_control(dia_alpha=value)

    @property
    def dia_beta(self) -> float | None:
        """Exponent of the matrix elements in the diabatic contribution to the metric."""
        return self._dia_beta

    @dia_beta.setter
    def dia_beta(self, value: float | None) -> None:
        self.set_control(dia_beta=value)

    @property
    def num_steps(self) -> int | None:
        """Number of points of the control-parameter grid."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int | None) -> None:
        self.set_control(num_steps=value)

    @property
    def control_sol(self) -> NDArray[np.float64]:
        """Return the optimal control pulse on the normalized time grid ``tau``, solving the problem if needed."""
        if not self._flags["ode_solved"]:
            self.solve_problem()  # Attempt to solve the ODE if not already solved (same accuracy as last time)
        if self._control_sol is None:
            raise SolverError("The control solution is unavailable after solving.")
        return self._control_sol.copy()

    @property
    def tau(self) -> NDArray[np.float64]:
        """Return the normalized time grid (tau = t / t_f) of the control solution, solving the problem if needed."""
        if not self._flags["ode_solved"]:
            self.solve_problem()
        if self._tau is None:
            raise SolverError("The normalized time grid is unavailable after solving.")
        return self._tau.copy()

    @property
    def eigenenergies(self) -> NDArray[np.float64]:
        """Return the real (float64) eigenenergies, also for complex H, solving the eigenproblem if needed."""
        if not self._flags["eigenproblem_solved"]:
            self._check_eigensystem_parameters()
            self._solve_eigenproblem()
        if self._centered_energies is None or self._energy_offset is None:
            raise SolverError("Eigenenergies are unavailable after eigensystem solution.")
        return self._centered_energies.copy() + self._energy_offset[:, None]

    @property
    def control_pulse(self) -> NDArray[np.float64]:
        """Return the control-parameter grid, solving the eigenproblem if needed."""
        if not self._flags["eigenproblem_solved"]:
            self._check_eigensystem_parameters()
            self._solve_eigenproblem()
        if self._control_pulse is None:
            raise SolverError("Control grid is unavailable after eigensystem solution.")
        return self._control_pulse.copy()

    def set_parameters(self, **parameters: Any) -> None:
        """
        Set the parameters for the ControlModel. This method allows you to specify any parameters that are needed to
        compute the ControlModel and its partial derivative. The parameters should be provided as keyword arguments,
        e.g., set_parameters(param1=value1, param2=value2, ...), and they will be stored. If a single parameter is
        updated, others will not be affected.

        The values are stored as deep copies, so changing a passed object in place (e.g. an array) does not affect the
        model until it is passed to ``set_parameters`` again.

        Parameters
        ----------
        **parameters : Any
            Fixed keyword arguments passed to ``H_func`` and ``partial_H_func``.
            Supplied names replace their stored values; other parameters are retained.
            A change invalidates the cached eigensystem and all dependent results.

        Raises
        ------
        ImmutableConfigurationError
            If the model was constructed with constant ``H_d`` and ``H_c`` matrices.
        InvalidControlParameterError
            If a parameter name is the configured ``control_name``.
        """

        if self._affine_hamiltonian:
            raise ImmutableConfigurationError(
                "set_parameters() is unavailable when constant H_d and H_c matrices are supplied. "
                "Construct a new ControlModel to change parameters absorbed into these matrices."
            )

        if self.control_name in parameters:
            raise InvalidControlParameterError(
                f"{self.control_name!r} is the control variable and cannot also be supplied as a fixed parameter."
            )

        # Copy on store: an aliased array changed in place would otherwise compare equal to itself here and leave the
        # cached eigensystem and pulse stale.
        new_params = {**self._parameters, **deepcopy(parameters)}

        if values_equal(new_params, self._parameters):
            return

        self._parameters = new_params
        self._flags["eigenproblem_solved"] = False

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return a read-only copy of the Hamiltonian parameters."""
        return MappingProxyType(deepcopy(self._parameters))

    def set_control(
        self,
        control_name: str | None = None,
        pulse_initial: float | None = None,
        pulse_final: float | None = None,
        initial_state: int | None = None,
        final_state: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        dia_alpha: float | None = None,
        dia_beta: float | None = None,
        num_steps: int | None = None,
    ) -> None:
        """
        Set the control parameters for the optimization problem. This method allows you to specify the control
        parameters such as the name of the control parameter, the initial and final values of the control pulse, ....
        Control parameters can be set individually, and if any parameter is not provided, it will retain its previous
        value, so you can update only the parameters you want without affecting the others.

        Parameters
        ----------
        control_name : str | None
            The name of the control parameter (e.g., "lambda"). This is used to identify the control parameter in the
             Hamiltonian function.
        pulse_initial : float | None
            The initial value of the control pulse. This is the value of the control parameter at the beginning of the
             pulse.
        pulse_final : float | None
            The final value of the control pulse. This is the value of the control parameter at the end of the pulse.
        initial_state : int | None
            The index of the initial state in the energy spectrum at ``pulse_initial``. This is used to compute the
            metric tensor and the ODE for the control pulse. The occupied state is followed along the sweep by
            eigenvector overlap, so at a true (symmetry-protected) level crossing it keeps its identity and changes its
            energy index.
        final_state : int | None
            The index of the final state in the energy spectrum at ``pulse_final``. This is used to compute the metric
            tensor and the ODE for the control pulse. As long as it has never been provided, the transfer is adiabatic
            and ``final_state`` is the index in which the state followed from ``initial_state`` ends (also when
            ``initial_state`` changes later); once provided, it keeps its value. If it differs from that index, the
            occupied state is followed along the sweep: each anticrossing with the neighbouring state towards
            ``final_state`` is passed diabatically, until ``final_state`` is reached. All other anticrossings are passed
            adiabatically, and only transitions from the occupied state contribute to the metric tensor.
        alpha : float | None
            The exponent alpha used in the metric tensor computation. This parameter controls the weighting of the
             energy gaps in the metric tensor.
        beta : float | None
            The exponent beta used in the metric tensor computation. This parameter controls the weighting of the
            matrix elements in the metric tensor.
        dia_alpha : float | None
            The exponent alpha used in the diabatic passage contribution to the metric tensor. This parameter controls
             the weighting of the energy gaps in the diabatic passage contribution to the metric tensor.
        dia_beta : float | None
            The exponent beta used in the diabatic passage contribution to the metric tensor. This parameter controls
             the weighting of the matrix elements in the diabatic passage contribution to the metric tensor.
        num_steps : int | None
            The number of steps to use in the discretization of the control pulse. This determines the resolution of
            the control pulse and the accuracy of the numerical solution. If omitted and no previous value exists,
            a default of ``2**10 + 1`` is used.
        """

        # Build and validate the complete candidate configuration first.
        # ToDo: With affine Hamiltonian, we could allow the user to set the control_name to None
        candidate_name = self._control_name if control_name is None else self._validate_control_name(control_name)

        candidate_pulse_initial = (
            self._pulse_initial if pulse_initial is None else self._validate_pulse_value(pulse_initial, "pulse_initial")
        )

        candidate_pulse_final = (
            self._pulse_final if pulse_final is None else self._validate_pulse_value(pulse_final, "pulse_final")
        )

        candidate_initial_state = (
            self._initial_state if initial_state is None else self._validate_state_index(initial_state, "Initial")
        )

        candidate_final_state = (
            self._final_state if final_state is None else self._validate_state_index(final_state, "Final")
        )
        candidate_final_state_explicit = self._final_state_explicit or final_state is not None

        # A final state that was never given explicitly follows the initial state, so changing initial_state alone
        # keeps an adiabatic problem adiabatic instead of silently turning it into a diabatic transfer.
        if not candidate_final_state_explicit:
            candidate_final_state = candidate_initial_state

        candidate_alpha = self._alpha if alpha is None else self._validate_exponent(alpha, "Alpha")

        candidate_beta = self._beta if beta is None else self._validate_exponent(beta, "Beta")

        candidate_dia_alpha = (
            self._dia_alpha if dia_alpha is None else self._validate_exponent(dia_alpha, "Diabatic alpha")
        )

        candidate_dia_beta = self._dia_beta if dia_beta is None else self._validate_exponent(dia_beta, "Diabatic beta")

        if num_steps is None:
            candidate_num_steps = self._num_steps if self._num_steps is not None else 2**10 + 1
        else:
            candidate_num_steps = self._validate_num_steps(num_steps)

        # Cross-parameter validation.
        if candidate_name is not None and candidate_name in self._parameters:
            raise InvalidControlParameterError(
                f"Control name {candidate_name!r} collides with a stored Hamiltonian parameter."
            )

        if (
            candidate_pulse_initial is not None
            and candidate_pulse_final is not None
            and candidate_pulse_initial == candidate_pulse_final
        ):
            raise InvalidControlParameterError("pulse_initial and pulse_final values must be different.")

        # Determine what must be invalidated before committing.
        eigensystem_changed = (
            candidate_name != self._control_name
            or candidate_pulse_initial != self._pulse_initial
            or candidate_pulse_final != self._pulse_final
            or candidate_num_steps != self._num_steps
        )

        # Making final_state explicit turns an adiabatic transfer into a diabatic one if it differs from the index in
        # which the followed state ends (self.final_state), which a true level crossing can change.
        became_explicit = candidate_final_state_explicit and not self._final_state_explicit
        states_changed = candidate_initial_state != self._initial_state or candidate_final_state != (
            self.final_state if became_explicit else self._final_state
        )

        metric_changed = (
            states_changed
            or candidate_alpha != self._alpha
            or candidate_beta != self._beta
            or candidate_dia_alpha != self._dia_alpha
            or candidate_dia_beta != self._dia_beta
        )

        # Everything is valid. Commit atomically.
        self._control_name = candidate_name
        self._pulse_initial = candidate_pulse_initial
        self._pulse_final = candidate_pulse_final
        self._initial_state = candidate_initial_state
        self._final_state = candidate_final_state
        self._final_state_explicit = candidate_final_state_explicit
        self._alpha = candidate_alpha
        self._beta = candidate_beta
        self._dia_alpha = candidate_dia_alpha
        self._dia_beta = candidate_dia_beta
        self._num_steps = candidate_num_steps

        # Invalidate caches only after the successful commit.
        if eigensystem_changed:
            self._flags["eigenproblem_solved"] = False

        if states_changed:
            self._flags["dia_list_computed"] = False
            self._dia_list = None

        if metric_changed:
            self._flags["metric_computed"] = False

    def solve_problem(
        self,
        pulse_accuracy: int | None = None,
        solver: Callable[..., Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        metric_integrator: Callable[..., Any] | None = None,
        metric_integrator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Solve the optimization problem to find the optimal control pulse. This method computes the metric tensor based
        on the energies and matrix elements of the ControlModel, and then solves the ODE for the control pulse using the
        computed metric tensor. Note that this method does not synthesize the pulse itself, for that you need to call
        the synthesize_pulse() method after solving the problem.

        Parameters
        ----------
        pulse_accuracy : int | None
            The number of points to use in the numerical solution of the ODE for the control pulse. Higher values will
             yield a more accurate solution but will also increase the computational cost. If omitted, the value of
             the previous solve is reused (1000 on the first solve).
        solver : Callable[..., Any] | None
            Callable used to integrate the ODE. Defaults to ``scipy.integrate.solve_ivp``.
            Expected signature is solver(fun, t_span, y0, t_eval=..., **kwargs), and the return
            value must expose ``t`` and ``y`` (or be a ``(t, y)`` tuple). A custom solver integrates the forward
            equation dx/dtau, normalized so that its exact solution ends at ``pulse_final``. Its result must follow the
            exact normalized time tau(x) to within 1e-2 at every sample, otherwise a ``SolverError`` is raised; both
            endpoints are then set exactly. The forward equation is badly conditioned for large exponents, where the
            default solver (which integrates the inverse equation dtau/dx) is preferable.
        solver_kwargs : dict[str, Any] | None
            Additional keyword arguments forwarded to ``solver``. Stored as a deep copy and reused by later calls until
            replaced; values may be arrays. For the default ``solve_ivp``, the keys ``dense_output``, ``events`` and
            ``t_eval`` are reserved, and a key that the chosen ``method`` does not accept raises ``ValidationError``.
            With the default solver, a non-empty dictionary replaces the exact antiderivative by Runge-Kutta
            integration of the inverse equation dtau/dx (defaults ``method="RK45"``, ``atol=1e-12``, ``rtol=1e-10``).
            Its independent variable is the control x, not tau: step options such as ``max_step``, ``first_step`` and
            ``min_step`` are in units of the control, and ``atol``/``rtol`` apply to tau(x). For example, with
            x from -500 to 500, ``max_step=1e-2`` takes at least 1e5 steps.
        metric_integrator : Callable[..., Any] | None
            Callable used to compute ``a_tilde`` from ``sqrt(metric_tensor)``. Defaults to
            ``scipy.integrate.romb``.
        metric_integrator_kwargs : dict[str, Any] | None
            Additional keyword arguments forwarded to ``metric_integrator``. Stored as a deep copy and reused by later
            calls until replaced; values may be arrays (e.g. ``x`` for ``scipy.integrate.trapezoid``).

        Notes
        -----
        ``solver``, ``solver_kwargs``, ``metric_integrator``, ``metric_integrator_kwargs`` and ``pulse_accuracy`` are
        stored on the model. An argument left as None keeps the stored value, so a choice passed once is reused by
        every later ``solve_problem()`` call and by the automatic re-solves of ``control_sol``, ``tau`` and
        ``synthesize_pulse()`` (and therefore of ``Dynamics(duration=..., model=...)``). To return to the defaults,
        pass them explicitly: ``solve_problem(solver=scipy.integrate.solve_ivp, solver_kwargs={},
        metric_integrator=scipy.integrate.romb, metric_integrator_kwargs={})``. Changing any of them invalidates only
        the dependent cached results (the metric for the integrator, the pulse for the solver).

        ``metric_integrator`` computes the positive scale ``a_tilde`` from the sampled metric. Both solver paths
        renormalize the PCHIP interpolant to span ``tau`` in [0, 1], so this scale cancels algebraically from their
        normalized equations. In particular, custom solvers use the integral of that interpolant for their
        normalization, avoiding an endpoint mismatch from using a different quadrature on the raw metric samples.
        """
        if pulse_accuracy is None:
            pulse_accuracy = (
                self._DEFAULT_PULSE_ACCURACY if self._previous_pulse_accuracy is None else self._previous_pulse_accuracy
            )
        if not isinstance(pulse_accuracy, (int, np.integer)) or isinstance(pulse_accuracy, bool):
            raise InvalidControlParameterError("pulse_accuracy must be an integer >= 3.")
        pulse_accuracy = int(pulse_accuracy)
        if pulse_accuracy < 3:
            raise InvalidControlParameterError("pulse_accuracy must be an integer >= 3.")
        config = self._check_control_parameters(metric_integrator)

        self._configure_integration(
            solver=solver,
            solver_kwargs=solver_kwargs,
            metric_integrator=metric_integrator,
            metric_integrator_kwargs=metric_integrator_kwargs,
        )

        self._solve_eigenproblem(config)
        self._compute_metric_tensor(config)

        self._solve_ode(pulse_accuracy)

    def _configure_integration(
        self,
        solver: Callable[..., Any] | None,
        solver_kwargs: dict[str, Any] | None,
        metric_integrator: Callable[..., Any] | None,
        metric_integrator_kwargs: dict[str, Any] | None,
    ) -> None:
        if solver_kwargs is not None and not isinstance(solver_kwargs, dict):
            raise ValidationError("solver_kwargs must be a dictionary.")
        if metric_integrator_kwargs is not None and not isinstance(metric_integrator_kwargs, dict):
            raise ValidationError("metric_integrator_kwargs must be a dictionary.")
        selected_solver = self._solver if solver is None else solver
        selected_metric_integrator = self._metric_integrator if metric_integrator is None else metric_integrator
        selected_solver_kwargs = self._solver_kwargs if solver_kwargs is None else dict(solver_kwargs)
        selected_metric_kwargs = (
            self._metric_integrator_kwargs if metric_integrator_kwargs is None else dict(metric_integrator_kwargs)
        )
        if not callable(selected_solver):
            raise ValidationError("solver must be a callable integration function.")

        if not callable(selected_metric_integrator):
            raise ValidationError("metric_integrator must be a callable integration function.")

        # values_equal handles array values (a plain != raises for them), and the stored deep copies keep an array
        # changed in place and passed again from comparing equal to itself.
        integrator_changed = selected_metric_integrator is not self._metric_integrator
        kwargs_changed = not values_equal(selected_metric_kwargs, self._metric_integrator_kwargs)

        if integrator_changed or kwargs_changed:
            self._metric_integrator = selected_metric_integrator
            self._metric_integrator_kwargs = deepcopy(selected_metric_kwargs)
            self._flags["metric_computed"] = False
        if selected_solver is not self._solver or not values_equal(selected_solver_kwargs, self._solver_kwargs):
            self._solver = selected_solver
            self._solver_kwargs = deepcopy(selected_solver_kwargs)
            self._flags["ode_solved"] = False

    def _solve_dia_list(self, config: _ControlParameters) -> None:
        """
        Compute the occupied state along the grid and the diabatic-passage list for a validated control configuration.

        The state is first followed from ``initial_state`` through the true level crossings alone. The transfer is
        adiabatic if ``final_state`` was not given or is the index in which that state ends; then no list is needed:
        ``_dia_list`` stays ``None``, and the stage is marked done so that the metric, which depends on it, can be
        marked as computed.
        """
        if self._flags["dia_list_computed"]:
            return

        if self._centered_energies is None or self._state_map is None:
            raise SolverError("Eigenenergies are unavailable before computing the occupied state.")

        adiabatic_path = follow_state(self._state_map, config.initial_state)
        if not config.final_state_explicit or config.final_state == adiabatic_path[-1]:
            self._dia_list = None
            self._occupied_states = adiabatic_path
        else:
            if config.dia_alpha is None or config.dia_beta is None:
                exponents = (("dia_alpha", config.dia_alpha), ("dia_beta", config.dia_beta))
                missing_params = [name for name, value in exponents if value is None]
                raise MissingControlParameterError(
                    f"Missing control parameters: {', '.join(missing_params)}. They are required for a diabatic "
                    f"transfer (final_state={config.final_state}, while the state followed adiabatically from "
                    f"initial_state={config.initial_state} ends in state {adiabatic_path[-1]}). Please set them using "
                    f"set_control({', '.join(f'{name}=<...>' for name in missing_params)})."
                )
            self._dia_list = build_dia_list(
                self._centered_energies,
                initial_state=config.initial_state,
                final_state=config.final_state,
                state_map=self._state_map,
            )
            self._occupied_states = self._dia_list.occupied
        self._resolved_final_state = int(self._occupied_states[-1])
        self._flags["dia_list_computed"] = True

    def _solve_eigenproblem(self, config: _EigensystemParameters | None = None) -> None:
        """Solve the Hamiltonian eigenproblem over a validated control grid."""
        if self._flags["eigenproblem_solved"]:
            return

        if config is None:
            config = self._check_eigensystem_parameters()

        self._control_pulse = np.linspace(
            config.pulse_initial,
            config.pulse_final,
            num=config.num_steps,
            dtype=float,
        )
        # The (n_points, dim, dim) stacks dominate the memory for large dim. Each one is released as soon as it is no
        # longer needed, and the Hamiltonian stack is centered in place.
        if self._affine_hamiltonian:
            assert self._H_d is not None
            assert self._H_c is not None
            hamiltonian = np.empty(
                (self._control_pulse.size, *self._H_d.shape), dtype=np.result_type(self._H_d, self._H_c, float)
            )
            np.multiply(self._control_pulse[:, None, None], self._H_c[None, :, :], out=hamiltonian)
            hamiltonian += self._H_d
        else:
            hamiltonian = stack_samples(self.evaluate_hamiltonian, self._control_pulse)
        hamiltonian = hamiltonian.astype(np.result_type(hamiltonian.dtype, float), copy=False)
        dimension = hamiltonian.shape[1]

        # State indices are required for a full solve, but not for plotting/eigenenergy access.
        if isinstance(config, _ControlParameters):
            for label, index in (("initial_state", config.initial_state), ("final_state", config.final_state)):
                if index >= dimension:
                    raise InvalidControlParameterError(
                        f"{label}={index} is out of range for a {dimension}-dimensional Hamiltonian."
                    )

        numerical_partial_H = self._flag_numerical_partial_H and not self._affine_hamiltonian
        # Rounding error of the samples for the numerical derivative, set by their magnitude before centering.
        sample_scale = float(np.max(np.abs(hamiltonian))) if numerical_partial_H else 0.0
        # The trace of a Hermitian matrix is real; keeping it complex would make the public eigenenergies complex.
        energy_offset = np.trace(hamiltonian, axis1=1, axis2=2).real / dimension
        diagonal = np.arange(dimension)
        hamiltonian[:, diagonal, diagonal] -= energy_offset[:, None]

        # The finite differences need the samples, so they run before the eigenvectors are allocated.
        partial_H = None
        if numerical_partial_H:
            partial_H = self._compute_numerical_partial_H(hamiltonian, sample_scale=sample_scale)

        try:
            with limit_blas_threads(dimension):
                energies, eigenvectors = np.linalg.eigh(hamiltonian)
        except np.linalg.LinAlgError as exc:
            raise SolverError("Hamiltonian eigendecomposition failed.") from exc
        del hamiltonian

        energies = np.asarray(energies, dtype=float)
        if not np.all(np.isfinite(energies)):
            raise SolverError("Hamiltonian eigendecomposition produced non-finite eigenenergies.")
        self._centered_energies = energies
        self._energy_offset = energy_offset

        if self._affine_hamiltonian:
            assert self._H_c is not None
            partial_H = self._H_c
        elif partial_H is None:
            partial_H = stack_samples(
                lambda value: self._call_partial_hamiltonian(**self._evaluation_kwargs(value)), self._control_pulse
            )

        # States are followed through true level crossings by eigenvector overlap. At a crossing on a grid point the
        # crossing states are degenerate, and their eigenvectors are first rotated into the limits from either side.
        with limit_blas_threads(dimension):
            rotate_degenerate_subspaces(eigenvectors, partial_H, degenerate_clusters(energies, self._GAP_RTOL))
        self._matrix_elements = self._derivative_matrix_elements(eigenvectors, partial_H)
        chunk = max(1, self._MATRIX_ELEMENT_CHUNK_BYTES // (16 * dimension * dimension))
        with limit_blas_threads(dimension):
            self._state_map = crossing_map(eigenvectors, self._matrix_elements, self._CROSSING_RTOL, chunk)
        self._flags["eigenproblem_solved"] = True

    def _derivative_matrix_elements(self, eigenvectors: np.ndarray, partial_H: np.ndarray) -> np.ndarray:
        """
        |<m|dH/dx|n>| between the eigenvectors at every grid point, with shape (n_points, dim, dim).

        ``partial_H`` is one matrix (affine model) or one per grid point. The products are formed in chunks of grid
        points, so their temporaries stay small compared with the (n_points, dim, dim) inputs.
        """
        n_points, dimension = eigenvectors.shape[:2]
        chunk = max(1, self._MATRIX_ELEMENT_CHUNK_BYTES // (16 * dimension * dimension))
        matrix_elements = np.empty((n_points, dimension, dimension), dtype=float)
        with limit_blas_threads(dimension):
            for start in range(0, n_points, chunk):
                block = slice(start, min(start + chunk, n_points))
                vectors = eigenvectors[block]
                derivative = partial_H if partial_H.ndim == 2 else partial_H[block]
                np.abs(vectors.conj().transpose(0, 2, 1) @ derivative @ vectors, out=matrix_elements[block])
        if not np.all(np.isfinite(matrix_elements)):
            raise SolverError("Hamiltonian derivative matrix elements contain non-finite values.")
        return matrix_elements

    def _metric_ratio(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        alpha: float | np.ndarray,
        beta: float | np.ndarray,
        transition: tuple[int | np.ndarray, int | np.ndarray],
        points: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Metric contribution of one transition, on all grid points or on the grid indices ``points``.

        The exponents are scalars or arrays with one value per point. Several transitions can be evaluated at once by
        passing one entry per (point, transition) pair, with ``points`` holding the grid index of each entry and the two
        states of ``transition`` given as arrays.

        A vanishing gap raises ``MetricComputationError``, except at a true level crossing on a grid point: there the
        two states are degenerate at that point only, dH/dx does not couple them, and the contribution (zero on either
        side of the crossing for ``beta > 0``) is set to zero.
        """
        crossings = np.array([], dtype=int)
        positive_alpha = np.asarray(alpha) > 0
        if np.any(positive_alpha):
            if self._centered_energies is None:
                raise MetricComputationError("Eigenenergies are unavailable for degeneracy detection.")

            bandwidth = np.ptp(self._centered_energies, axis=1)
            if points is not None:
                bandwidth = bandwidth[points]

            tolerance = self._GAP_RTOL * bandwidth

            singular = (denominator <= tolerance) & positive_alpha

            indices = np.flatnonzero(singular)
            grid_indices = indices if points is None else points[indices]
            states = [np.broadcast_to(state, denominator.shape)[indices] for state in transition]
            entry_beta = np.broadcast_to(beta, denominator.shape)[indices]
            is_crossing = self._is_true_crossing(grid_indices, states[0], states[1], numerator[indices], entry_beta)
            crossings = indices[is_crossing]

            if not np.all(is_crossing):
                control_pulse = self._control_pulse

                if control_pulse is None:
                    raise MetricComputationError("Control grid is unavailable for degeneracy detection.")

                indices, grid_indices = indices[~is_crossing], grid_indices[~is_crossing]
                first = int(np.flatnonzero(~is_crossing)[0])

                details = ", ".join(
                    f"x={control_pulse[g]:.6g}, gap={denominator[i]:.3e}, tol={tolerance[i]:.3e}"
                    for i, g in zip(indices[:5], grid_indices[:5], strict=True)
                )

                raise MetricComputationError(
                    "Degenerate or near-degenerate energy gap encountered for transition "
                    f"({int(states[0][first])}, {int(states[1][first])}): {details}"
                )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            ratio = np.asarray(numerator**beta / denominator**alpha, dtype=float)
        ratio[crossings] = 0.0

        return ratio

    def _is_true_crossing(
        self, points: np.ndarray, source: np.ndarray, target: np.ndarray, numerator: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """
        Whether each degenerate pair of states ``(source, target)`` at grid index ``points`` is a true level crossing.

        The pair is degenerate at that grid point only, not at its neighbours, and dH/dx does not couple it
        (``numerator`` at most ``_CROSSING_RTOL`` times the largest matrix element at that point). Its metric
        contribution vanishes on both sides of the crossing only for ``beta > 0``.
        """
        assert self._centered_energies is not None and self._matrix_elements is not None
        energies = self._centered_energies
        n_points = energies.shape[0]
        scale = np.max(self._matrix_elements[points], axis=(1, 2))
        result = (numerator <= self._CROSSING_RTOL * scale) & (beta > 0)
        for offset in (-1, 1):
            neighbour = np.clip(points + offset, 0, n_points - 1)
            inside = neighbour != points
            gap = np.abs(energies[neighbour, target] - energies[neighbour, source])
            open_gap = gap > self._GAP_RTOL * np.ptp(energies[neighbour], axis=1)
            result &= open_gap | ~inside
        return result

    def _compute_metric_tensor(self, config: _ControlParameters) -> None:
        """
        Compute the metric tensor G_tensor based on the energies and matrix elements of the ControlModel. If the
        eigenproblem has not been solved yet, solve it first to obtain the energies and matrix elements.
        """
        if self._flags["metric_computed"]:
            return

        # Also for an adiabatic transfer, where it follows the occupied state and leaves the list empty.
        self._solve_dia_list(config)
        if self._dia_list is None:
            self._compute_G_adiabatic(config)
        else:
            self._compute_G_diabatic(config)

        if self._metric_tensor is None or self._control_pulse is None:
            raise MetricComputationError("Metric tensor was not produced.")
        metric = np.asarray(self._metric_tensor, dtype=float)
        if metric.ndim != 1 or metric.shape != self._control_pulse.shape:
            raise MetricComputationError(
                f"Metric tensor must have shape {self._control_pulse.shape}; received {metric.shape}."
            )
        if not np.all(np.isfinite(metric)):
            raise MetricComputationError("Metric tensor contains NaN or infinite values.")

        tolerance = self._METRIC_RTOL * float(np.max(np.abs(metric)))
        if np.any(metric < -tolerance):
            raise MetricComputationError("Metric tensor contains negative values.")
        metric = np.maximum(metric, 0.0)

        if not np.any(metric > tolerance):
            raise MetricComputationError("Metric tensor is always zero or numerically singular")

        if np.any(metric <= tolerance):
            locations = self._control_pulse[metric <= tolerance]
            sample = ", ".join(f"{value:.6g}" for value in locations[:3])
            warnings.warn(
                "Metric tensor is zero or numerically singular"
                + (f" near control value(s) {sample}." if sample else "."),
                NumericalStabilityWarning,
                stacklevel=2,
            )

        dx = float(np.abs(self._control_pulse[1] - self._control_pulse[0]))
        metric_values = np.sqrt(metric)

        self._check_metric_integrator_samples(metric_values.size, self._metric_integrator)

        try:
            result = self._metric_integrator(metric_values, dx=dx, **self._metric_integrator_kwargs)
        except TypeError as exc:
            raise ValidationError(
                "Invalid metric_integrator call: ensure it accepts arguments like (values, dx=..., **kwargs)."
            ) from exc
        try:
            self._a_tilde = float(result)
        except (TypeError, ValueError) as exc:
            raise MetricComputationError("Metric integrator must return one finite real scalar.") from exc
        if not np.isfinite(self._a_tilde) or self._a_tilde <= 0:
            raise MetricComputationError("Metric normalization must be finite and strictly positive.")
        self._flags["metric_computed"] = True

    def _compute_G_diabatic(self, config: _ControlParameters) -> None:
        if self._centered_energies is None or self._matrix_elements is None or self._dia_list is None:
            raise SolverError("Diabatic metric prerequisites are unavailable.")
        if config.dia_alpha is None or config.dia_beta is None:
            raise MissingControlParameterError("Diabatic metric exponents are required.")

        num = self._centered_energies.shape[0]

        # Only transitions from the occupied state contribute. They are evaluated together, one entry per (point,
        # target state) pair; the adiabatic fraction blends the adiabatic and diabatic exponents.
        fraction = self._dia_list.adiabatic_fraction
        points, targets = np.nonzero(fraction >= 0)
        sources = self._dia_list.occupied[points]
        adiabatic_fraction = fraction[points, targets]
        denominator = np.abs(self._centered_energies[points, targets] - self._centered_energies[points, sources])
        ratio = self._metric_ratio(
            self._matrix_elements[points, sources, targets],
            denominator,
            alpha=adiabatic_fraction * config.alpha + (1.0 - adiabatic_fraction) * config.dia_alpha,
            beta=adiabatic_fraction * config.beta + (1.0 - adiabatic_fraction) * config.dia_beta,
            transition=(sources, targets),
            points=points,
        )
        self._metric_tensor = np.bincount(points, weights=ratio, minlength=num)

    def _compute_G_adiabatic(self, config: _ControlParameters) -> None:
        """
        Compute the adiabatic contribution to the metric tensor.

        Only transitions from the occupied state contribute. It is followed through true level crossings, so its
        energy index may change along the grid.
        """
        if self._centered_energies is None or self._matrix_elements is None or self._occupied_states is None:
            raise SolverError("Adiabatic metric prerequisites are unavailable.")

        num, dim = self._centered_energies.shape
        occupied = self._occupied_states
        metric = np.zeros(num, dtype=float)
        points = np.arange(num)
        for offset in range(1, dim):
            # The states other than the occupied one, one per point and offset, in no particular order.
            state = (occupied + offset) % dim
            denominator = np.abs(self._centered_energies[points, state] - self._centered_energies[points, occupied])
            metric += self._metric_ratio(
                self._matrix_elements[points, occupied, state],
                denominator,
                alpha=config.alpha,
                beta=config.beta,
                transition=(occupied, state),
            )
        self._metric_tensor = metric

    def _compute_numerical_partial_H(
        self, samples: np.ndarray | None = None, sample_scale: float | None = None
    ) -> np.ndarray:
        """
        Evaluate the derivative of the traceless part of H, d(H - tr(H) I / N)/dx, at every point of the control grid.

        The trace part only adds a multiple of the identity, which does not change the off-diagonal matrix elements
        between eigenstates that the metric uses; leaving it out keeps a large identity offset in H from dominating
        the error estimate.

        The derivative is first computed with finite differences of the Hamiltonian samples on the grid, which needs
        no extra Hamiltonian evaluations. Its error is estimated by comparison with a lower-order stencil, and points
        where the estimate is too large are recomputed with the adaptive ``scipy.differentiate.jacobian``. The
        tolerance is floored at the rounding error of the samples, which a smaller step cannot resolve. If the
        adaptive derivative does not converge at such a point, the finite-difference value is kept and a
        ``NumericalStabilityWarning`` is issued.

        Parameters
        ----------
        samples : np.ndarray | None
            Hamiltonian evaluated on the control grid, with shape (n_points, *H_shape). Evaluated if not given.
        sample_scale : float | None
            If given, ``samples`` are already traceless (they are used as they are, without a copy) and
            ``sample_scale`` is max|H| of the samples before the trace was removed, which sets their rounding error.

        Returns
        -------
        dH_dx
            Complex derivative of the traceless part of H, with shape (n_points, *H_shape).
        """
        if self._control_pulse is None:
            raise SolverError("x_grid is unavailable before the eigenproblem grid is initialized.")

        x_grid = np.asarray(self._control_pulse, dtype=float)

        if x_grid.ndim != 1:
            raise ValidationError("x_grid must be one-dimensional.")

        if x_grid.size < 2:
            raise ValidationError("x_grid must contain at least two points.")

        if not np.all(np.isfinite(x_grid)):
            raise ValidationError("x_grid must contain only finite values.")

        if np.unique(x_grid).size < 2:
            raise ValidationError("x_grid must contain at least two distinct points.")

        if samples is None:
            samples = np.stack([self.evaluate_hamiltonian(float(x)) for x in x_grid])
        samples = np.asarray(samples, dtype=np.complex128)
        if samples.shape[0] != x_grid.size:
            raise SolverError("Hamiltonian samples do not match the control grid.")

        if sample_scale is None:
            # Rounding error of the samples, set by their full magnitude before the trace is removed.
            sample_scale = float(np.max(np.abs(samples)))
            samples = traceless(samples)

        n_points = x_grid.size
        # The grid is a linspace, so its nominal step follows from the endpoints. The individual steps differ from it
        # by rounding, which is large relative to the step when the control values carry a large offset.
        step = float(x_grid[-1] - x_grid[0]) / (n_points - 1)
        roundoff = self._FD_ROUNDOFF_FACTOR * np.finfo(float).eps * float(np.max(np.abs(x_grid))) / abs(step)
        uniform = np.allclose(np.diff(x_grid), step, rtol=max(1e-9, roundoff), atol=0.0)
        sample_noise = self._FD_ROUNDOFF_FACTOR * np.finfo(float).eps * sample_scale / abs(step)

        if uniform and n_points >= self._FD_MIN_POINTS:
            stencil = min(self._FD_ORDER + 1, n_points)
            derivative = uniform_grid_derivative(samples, step, stencil)
            estimate = uniform_grid_derivative(samples, step, stencil - 2)
            # In place, so the difference does not need another (n_points, *H_shape) array.
            estimate -= derivative
            error = np.max(np.abs(estimate), axis=tuple(range(1, samples.ndim)))
            del estimate
            tolerance = max(max(self._FD_RTOL, roundoff) * float(np.max(np.abs(derivative))), sample_noise)
            inaccurate = error > tolerance
            fallback = derivative
        else:
            derivative = np.zeros_like(samples)
            inaccurate = np.ones(n_points, dtype=bool)
            fallback = None

        if np.any(inaccurate):
            derivative[inaccurate] = self._jacobian_partial_H(
                x_grid,
                inaccurate,
                samples,
                fallback=None if fallback is None else fallback[inaccurate],
                sample_scale=sample_scale,
            )

        return derivative

    def _jacobian_partial_H(
        self,
        x_grid: np.ndarray,
        selected: np.ndarray,
        samples: np.ndarray,
        fallback: np.ndarray | None = None,
        sample_scale: float = 0.0,
    ) -> np.ndarray:
        """
        Adaptive derivative of the traceless part of H at the ``selected`` points of ``x_grid``, computed with
        ``scipy.differentiate.jacobian``.

        H_func is assumed not to be vectorized: it accepts one scalar x and returns a real- or complex-valued
        Hamiltonian. ``samples`` are the traceless Hamiltonian samples on the grid, and ``sample_scale`` is max|H| of
        the original samples, which sets their rounding error. Steps never leave the domain of the grid (one-sided
        differences at its boundaries). Points where ``jacobian`` does not converge take the corresponding value of
        ``fallback`` (shape (n_selected, *H_shape)) with a ``NumericalStabilityWarning``; without a fallback, a
        ``SolverError`` is raised.
        """
        order = self._FD_ORDER
        unique_x = np.unique(x_grid)
        initial_step = float(np.min(np.diff(unique_x)))

        H_reference = samples[0]
        H_end = samples[-1]
        H_shape = H_reference.shape
        n_elements = H_reference.size

        # Scale the absolute tolerance with the typical derivative size |H| / |x range|, so that convergence does not
        # depend on the units of the Hamiltonian or of the control parameter.
        derivative_scale = max(float(np.max(np.abs(H_reference))), float(np.max(np.abs(H_end)))) / float(
            unique_x[-1] - unique_x[0]
        )
        # The relative accuracy is limited by the rounding of x + step when x carries a large offset.
        roundoff = self._FD_ROUNDOFF_FACTOR * np.finfo(float).eps * float(np.max(np.abs(unique_x))) / initial_step
        # The rounding error of the Hamiltonian values (e.g. from a large identity offset) cannot be resolved by
        # smaller steps either.
        sample_noise = self._FD_ROUNDOFF_FACTOR * np.finfo(float).eps * sample_scale / initial_step
        tolerances = {
            "atol": max(1e-10 * derivative_scale if derivative_scale > 0 else 1e-10, sample_noise),
            "rtol": max(1e-8, roundoff),
        }

        def evaluate_and_pack(
            x_column: np.ndarray,
        ) -> np.ndarray:
            """
            Evaluate the non-vectorized Hamiltonian at one scalar x.

            x_column has shape (1,), since x is the only independent
            variable.
            """
            x = float(x_column[0])

            H = np.asarray(
                self.evaluate_hamiltonian(float(x)),
                dtype=np.complex128,
            )

            if H.shape != H_shape:
                raise ValidationError(f"H_func returned inconsistent shapes: expected {H_shape}, got {H.shape}.")

            H_flat = traceless(H).ravel()

            # scipy.differentiate.jacobian expects real outputs.
            return np.concatenate(
                (
                    H_flat.real,
                    H_flat.imag,
                )
            )

        def packed_hamiltonian(
            x_array: np.ndarray,
        ) -> np.ndarray:
            """
            Adapt the scalar H_func to SciPy's vectorized interface.

            x_array has shape (1, ...), where the trailing axes contain
            evaluation points introduced by SciPy.
            """
            return np.apply_along_axis(
                evaluate_and_pack,
                axis=0,
                arr=x_array,
            )

        # Use one-sided differences at the two domain boundaries.
        x_min = np.min(x_grid)
        x_max = np.max(x_grid)
        points = x_grid[selected]

        step_direction = np.zeros_like(points, dtype=int)
        step_direction[points - x_min < initial_step] = 1
        step_direction[x_max - points < initial_step] = -1

        result = jacobian(
            packed_hamiltonian,
            points[np.newaxis, :],
            order=order,
            initial_step=initial_step,
            step_direction=step_direction[np.newaxis, :],
            tolerances=tolerances,
        )

        success = np.asarray(result.success, dtype=bool)
        failed_points = np.zeros(points.size, dtype=bool)
        if not np.all(success):
            status = np.asarray(result.status)
            error = np.asarray(result.error, dtype=float)

            failed_points = np.any(
                ~success,
                axis=tuple(range(success.ndim - 1)),
            )

            failed_indices = np.flatnonzero(failed_points)

            details = []
            for index in failed_indices[:5]:
                point_failed = ~success[..., index]
                statuses = np.unique(status[..., index][point_failed])

                point_errors = error[..., index][point_failed]
                finite_errors = point_errors[np.isfinite(point_errors)]
                max_error = float(np.max(finite_errors)) if finite_errors.size else np.nan

                details.append(f"x={points[index]:.6g}: status={statuses.tolist()}, max_error={max_error:.3e}")

            message = (
                "Numerical differentiation failed to converge at "
                f"{failed_indices.size} control point(s): " + "; ".join(details)
            )
            if fallback is None:
                raise SolverError(message)
            warnings.warn(
                message + ". The finite-difference derivative on the control grid is used there instead.",
                NumericalStabilityWarning,
                stacklevel=3,
            )

        # result.df shape:
        # (2 * n_elements, 1, n_points)
        derivative = np.asarray(result.df[:, 0, :])

        dH_flat = derivative[:n_elements] + 1j * derivative[n_elements:]

        # Convert from (*H_shape, n_points) to
        # (n_points, *H_shape).
        dH_dx = np.moveaxis(dH_flat.reshape(H_shape + (points.size,)), -1, 0)

        if fallback is not None:
            dH_dx[failed_points] = fallback[failed_points]

        if not np.all(np.isfinite(dH_dx)):
            raise SolverError("Numerical Hamiltonian derivative contains non-finite values.")

        return dH_dx

    def _solve_ode(self, pulse_accuracy: int) -> None:
        """Solve the normalized geodesic ODE and validate the solver result."""
        if self._previous_pulse_accuracy != pulse_accuracy:
            self._flags["ode_solved"] = False
        if self._flags["ode_solved"]:
            return
        if self._control_pulse is None or self._metric_tensor is None or self._a_tilde is None:
            raise SolverError("ODE cannot be solved before the control pulse and metric tensor are available.")

        # dtau/dx = sqrt(G) / a_tilde stays finite where the metric vanishes (the pulse passes such a point infinitely
        # fast), unlike the speed dx/dtau = a_tilde / sqrt(G).
        inverse_speed = np.sqrt(self._metric_tensor) / self._a_tilde
        if not np.all(np.isfinite(inverse_speed)) or np.any(inverse_speed < 0):
            raise MetricComputationError("ODE rate dtau/dx = sqrt(G) / a_tilde is non-finite or negative.")

        order = np.argsort(self._control_pulse)
        # Both solver paths use the PCHIP interpolant of dtau/dx, which keeps the non-negative samples non-negative.
        inverse_interpolation = PchipInterpolator(
            self._control_pulse[order],
            inverse_speed[order],
            extrapolate=False,
        )
        antiderivative = inverse_interpolation.antiderivative()

        direction = float(np.sign(self._control_pulse[-1] - self._control_pulse[0]))
        lower = float(np.min(self._control_pulse))
        upper = float(np.max(self._control_pulse))
        x_start = float(self._control_pulse[0])
        x_end = float(self._control_pulse[-1])

        tau = np.linspace(0.0, 1.0, pulse_accuracy)
        kwargs = dict(self._solver_kwargs)
        if self._solver is solve_ivp:
            reserved = {"dense_output", "events", "t_eval"}.intersection(kwargs)
            if reserved:
                names = ", ".join(sorted(reserved))
                raise ValidationError(f"solver_kwargs must not override internally managed option(s): {names}.")
            # The ODE dx/dtau = a_tilde / sqrt(G(x)) is autonomous; solve its inverse dtau/dx = sqrt(G(x)) / a_tilde and
            # invert tau(x) afterward. The speed can span many orders of magnitude (e.g. large alpha/beta), which makes
            # the forward problem require steps below floating-point spacing, while the inverse is well conditioned.
            x_grid = np.linspace(x_start, x_end, max(self._control_pulse.size, 8 * pulse_accuracy))

            if kwargs:
                # Errors in tau are amplified by dx/dtau when inverting, hence the tight default tolerances.
                kwargs = {"method": "RK45", "atol": 1e-12, "rtol": 1e-10, **kwargs}

                def inverse_model(x: float, _: np.ndarray) -> np.ndarray:
                    clipped = np.clip(x, lower, upper)
                    return np.atleast_1d(direction * inverse_interpolation(clipped)).astype(float)

                try:
                    with warnings.catch_warnings():
                        # solve_ivp only warns about options the chosen method does not accept, so a typo such as
                        # {"rtl": 1e-8} would silently fall back to the default tolerances.
                        warnings.filterwarnings(
                            "error", message="The following arguments have no effect", category=UserWarning
                        )
                        sol = solve_ivp(inverse_model, [x_start, x_end], [0.0], dense_output=True, **kwargs)
                except UserWarning as exc:
                    raise ValidationError(f"Invalid solve_ivp keyword arguments: {exc}") from exc
                except TypeError as exc:
                    raise ValidationError("Invalid solve_ivp keyword arguments.") from exc
                if not sol.success:
                    raise SolverError(str(sol.message))
                if sol.sol is None:
                    raise SolverError("ODE solution did not reach the requested final control value.")
                tau_grid = np.asarray(sol.sol(x_grid), dtype=float)[0]
            else:
                # Without solver options, integrate the interpolant exactly through its piecewise-polynomial
                # antiderivative, which is deterministic and free of step-size control.
                tau_grid = direction * (antiderivative(x_grid) - antiderivative(x_start))

            endpoint_time = float(tau_grid[-1])
            if not np.isfinite(endpoint_time) or endpoint_time <= 0:
                raise SolverError("ODE endpoint time must be finite and strictly positive.")
            tau_grid = np.maximum.accumulate(tau_grid / endpoint_time)
            # Keep only strictly increasing nodes; flat stretches occur where the metric is numerically negligible.
            keep = np.concatenate(([True], np.diff(tau_grid) > 0))
            # In a trailing flat stretch keep its last node (x_end) instead of its first, so that x(1) = x_end. The
            # pulse still jumps there: a vanishing metric means the geodesic passes that stretch in zero time.
            keep[tau_grid == tau_grid[-1]] = False
            keep[-1] = True
            if np.count_nonzero(keep) < 2:
                raise SolverError("ODE solution did not reach the requested final control value.")
            t = tau
            y = PchipInterpolator(tau_grid[keep], x_grid[keep])(np.clip(tau, tau_grid[keep][0], tau_grid[keep][-1]))
        else:
            # The forward equation divides by dtau/dx, so it is undefined where the metric vanishes.
            zeros = inverse_speed == 0
            if np.any(zeros):
                sample = ", ".join(f"{value:.6g}" for value in self._control_pulse[zeros][:3])
                raise MetricComputationError(
                    f"The metric tensor is zero at control value(s) {sample}, where the forward equation "
                    "dx/dtau = a_tilde / sqrt(G) of a custom solver is infinite. Use the default solver "
                    "(scipy.integrate.solve_ivp), which integrates dtau/dx instead."
                )
            # Normalize the forward ODE with the exact integral of the interpolant it uses, so that x(1) = x_end.
            # (a_tilde comes from the raw samples and differs from it, which made the pulse overshoot or fall short.)
            length = direction * float(antiderivative(x_end) - antiderivative(x_start))
            if not np.isfinite(length) or length <= 0:
                raise SolverError("ODE normalization must be finite and strictly positive.")

            def model(_: float, y: np.ndarray) -> np.ndarray:
                y = np.asarray(y, dtype=float)
                speed = length / inverse_interpolation(np.clip(y, lower, upper))
                # Stop at x_end, so that an overshooting step cannot run away beyond the control grid.
                return np.where(direction * (x_end - y) > 0, direction * speed, 0.0)

            try:
                sol = self._solver(model, [0.0, 1.0], [x_start], t_eval=tau, **kwargs)
            except TypeError as exc:
                raise ValidationError(
                    "Invalid solver call: ensure solver accepts (fun, t_span, y0, t_eval=..., **kwargs)."
                ) from exc
            if hasattr(sol, "success") and not sol.success:
                raise SolverError(str(getattr(sol, "message", "ODE solver failed.")))
            if isinstance(sol, tuple):
                if len(sol) != 2:
                    raise SolverError("Solver tuple output must be a (t, y) pair.")
                t_arr, y_arr = sol
            elif hasattr(sol, "t") and hasattr(sol, "y"):
                t_arr, y_arr = sol.t, sol.y
            else:
                raise SolverError("Solver output must expose 't' and 'y', or return a (t, y) tuple.")
            t = np.asarray(t_arr, dtype=float)
            y = np.asarray(y_arr, dtype=float)
        if t.ndim != 1 or t.size != pulse_accuracy:
            raise SolverError(f"Solver time grid must be one-dimensional with {pulse_accuracy} samples.")
        if y.ndim == 2:
            if y.shape == (1, pulse_accuracy):
                control = y[0]
            elif y.shape == (pulse_accuracy, 1):
                control = y[:, 0]
            else:
                raise SolverError("Solver state array must have shape (1, n) or (n, 1).")
        elif y.ndim == 1 and y.size == pulse_accuracy:
            control = y
        else:
            raise SolverError("Solver state array has an invalid shape.")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(control)):
            raise SolverError("Solver output contains NaN or infinite values.")
        if np.any(np.diff(t) <= 0) or not np.isclose(t[0], 0.0) or not np.isclose(t[-1], 1.0):
            raise SolverError("Solver time grid must be strictly increasing and span [0, 1].")
        if self._solver is not solve_ivp:
            # Validate the whole trajectory against the exact normalized time tau(x) of the interpolant: an endpoint
            # check alone would accept a pulse that reaches x_end early and then stays there.
            tau_exact = direction * (antiderivative(np.clip(control, lower, upper)) - antiderivative(x_start)) / length
            deviation = float(np.max(np.abs(tau_exact - t)))
            if not np.isfinite(deviation) or deviation > self._SOLVER_TAU_TOL:
                raise SolverError(
                    f"Custom solver result deviates from the geodesic by {deviation:.3g} in normalized time "
                    f"(tolerance {self._SOLVER_TAU_TOL:g}). Tighten its tolerances (e.g. rtol) or use the default "
                    "solver."
                )
        control[0] = x_start
        control[-1] = x_end
        self._tau = t.copy()
        self._control_sol = control.copy()
        self._previous_pulse_accuracy = pulse_accuracy
        self._flags["ode_solved"] = True

    def _check_eigensystem_parameters(self) -> _EigensystemParameters:
        """Validate and return the configuration required by the eigensystem calculation."""
        control_name = self.control_name
        pulse_initial = self.pulse_initial
        pulse_final = self.pulse_final
        num_steps = self.num_steps

        missing_params = [
            name
            for name, value in (
                ("pulse_initial", pulse_initial),
                ("pulse_final", pulse_final),
                ("num_steps", num_steps),
            )
            if value is None
        ]
        if not self._affine_hamiltonian and control_name is None:
            missing_params.insert(0, "control_name")

        if missing_params:
            raise MissingControlParameterError(
                f"Missing control parameters for eigensystem: {', '.join(missing_params)}. "
                "Please set them using set_control(...)."
            )

        assert pulse_initial is not None
        assert pulse_final is not None
        assert num_steps is not None

        return _EigensystemParameters(
            control_name=control_name,
            pulse_initial=pulse_initial,
            pulse_final=pulse_final,
            num_steps=num_steps,
        )

    def plot_eigenvalues(
        self,
        fig: "Figure | SubFigure | None" = None,
        ax: "Axes | None" = None,
        legend: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        **plot_kwargs: Any,
    ) -> "tuple[Figure | SubFigure, Axes]":
        """
        Plot ControlModel eigenvalues as a function of the control parameter.

        Parameters
        ----------
        fig : Figure | SubFigure | None
            Existing figure or subfigure. If ``ax`` is absent, a subplot is added to it; if neither is supplied,
            a new figure and axes are created.
        ax : Axes | None
            Existing axes to draw on. Its parent figure or subfigure is returned when supplied.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : dict | None
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : str | None
            Label for the x-axis. Defaults to ``control_name`` when not provided.
        ylabel : str | None
            Label for the y-axis. Defaults to ``"Energy"`` when not provided.
        title : str | None
            Plot title. Defaults to ``"Hamiltonian eigenenergies"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot`` for each energy branch.

        Returns
        -------
        tuple[Figure | SubFigure, Axes]
            ``(fig, ax)`` with the generated plot.

        Raises
        ------
        ValidationError
            If the supplied axes or, when no axes are supplied, figure has an unsupported type.
        """
        config = self._check_eigensystem_parameters()
        control_name = config.control_name or "control"

        # Read the grid only after (re-)solving: a stale grid would not match the new eigenenergies.
        energies = self.eigenenergies
        control_grid = self._control_pulse
        assert control_grid is not None

        fig, ax = _figure_and_axes(fig, ax)

        for level in range(energies.shape[1]):
            ax.plot(control_grid, energies[:, level], label=f"E{level}", **plot_kwargs)

        ax.set_xlabel(control_name if xlabel is None else xlabel)
        ax.set_ylabel("Energy" if ylabel is None else ylabel)
        ax.set_title("Hamiltonian eigenenergies" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax

    def plot_metric_tensor(
        self,
        fig: "Figure | SubFigure | None" = None,
        ax: "Axes | None" = None,
        legend: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        **plot_kwargs: Any,
    ) -> "tuple[Figure | SubFigure, Axes]":
        """
        Plot the metric tensor (G tensor) as a function of the control parameter.

        If the metric tensor is not available yet, it is computed from the current
        ControlModel/control configuration.

        Parameters
        ----------
        fig : Figure | SubFigure | None
            Existing figure or subfigure. If ``ax`` is absent, a subplot is added to it; if neither is supplied,
            a new figure and axes are created.
        ax : Axes | None
            Existing axes to draw on. Its parent figure or subfigure is returned when supplied.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : dict | None
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : str | None
            Label for the x-axis. Defaults to ``control_name`` when not provided.
        ylabel : str | None
            Label for the y-axis. Defaults to ``"G tensor"`` when not provided.
        title : str | None
            Plot title. Defaults to ``"G tensor"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot``.

        Returns
        -------
        tuple[Figure | SubFigure, Axes]
            ``(fig, ax)`` with the generated plot.

        Raises
        ------
        ValidationError
            If the supplied axes or, when no axes are supplied, figure has an unsupported type.
        """
        config = self._check_control_parameters()
        self._solve_eigenproblem(config)
        self._compute_metric_tensor(config)
        control_name = config.control_name or "control"

        if self._control_pulse is None:
            raise ValidationError("Control pulse can not be None.")

        if self._metric_tensor is None:
            raise ValidationError("Metric tensor can not be None.")

        fig, ax = _figure_and_axes(fig, ax)
        ax.plot(self._control_pulse, self._metric_tensor, label="G", **plot_kwargs)

        ax.set_xlabel(control_name if xlabel is None else xlabel)
        ax.set_ylabel("G tensor" if ylabel is None else ylabel)
        ax.set_title("G tensor" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax

    def synthesize_pulse(self, duration: float) -> PulseControl:
        """
        Synthesize the control pulse based on the solution of the optimization problem. If the problem has not been
        solved yet, this method will automatically solve it first.

        Parameters
        ----------
        duration : float
            The total duration of the control pulse.

        Returns
        -------
        PulseControl
             An instance of the PulseControl class representing the synthesized control pulse.
        """
        if not self._flags["ode_solved"]:
            self.solve_problem()
        if self._control_sol is None:
            raise SolverError("Cannot synthesize a pulse before the control solution is available.")
        return PulseControl(self._control_sol, duration)

    @staticmethod
    def _check_metric_integrator_samples(n_samples: int, metric_integrator: Callable[..., Any]) -> None:
        """Raise if ``metric_integrator`` is ``scipy.integrate.romb`` and ``n_samples`` is not ``2**k + 1``."""
        if metric_integrator is romb:
            power_minus_one = n_samples - 1
            if power_minus_one <= 0 or (power_minus_one & (power_minus_one - 1)) != 0:
                raise InvalidControlParameterError(
                    f"num_steps={n_samples} is incompatible with romb. Use num_steps=2**k+1, "
                    "or pass a different metric_integrator to solve_problem(...)."
                )

    def _check_control_parameters(self, metric_integrator: Callable[..., Any] | None = None) -> _ControlParameters:
        """
        Validate and return all parameters required for a complete solve.

        ``metric_integrator`` is the integrator the solve will use (the stored one if None); ``num_steps`` is checked
        against it here, before the eigenproblem is solved.
        """
        control_name = self.control_name
        pulse_initial = self.pulse_initial
        pulse_final = self.pulse_final
        initial_state = self.initial_state
        # The stored value, not the one resolved by a previous solve.
        final_state = self._final_state
        alpha = self.alpha
        beta = self.beta
        dia_alpha = self.dia_alpha
        dia_beta = self.dia_beta
        num_steps = self.num_steps

        if (
            pulse_initial is None
            or pulse_final is None
            or num_steps is None
            or initial_state is None
            or final_state is None
            or alpha is None
            or beta is None
            or (not self._affine_hamiltonian and control_name is None)
        ):  # noqa: E501
            missing_params = [
                name
                for name, value in (
                    ("pulse_initial", pulse_initial),
                    ("pulse_final", pulse_final),
                    ("initial_state", initial_state),
                    ("final_state", final_state),
                    ("alpha", alpha),
                    ("beta", beta),
                    ("num_steps", num_steps),
                )
                if value is None
            ]
            if not self._affine_hamiltonian and control_name is None:
                missing_params.insert(0, "control_name")
            raise MissingControlParameterError(
                f"Missing control parameters: {', '.join(missing_params)}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)})."
            )

        assert pulse_initial is not None
        assert pulse_final is not None
        assert initial_state is not None
        assert final_state is not None
        assert alpha is not None
        assert beta is not None
        assert num_steps is not None

        # dia_alpha and dia_beta are checked once the spectrum shows whether the transfer is diabatic: final_state may
        # differ from initial_state also in an adiabatic transfer through a true level crossing.

        self._check_metric_integrator_samples(
            num_steps, self._metric_integrator if metric_integrator is None else metric_integrator
        )

        # The dimension is fixed once known, so H is evaluated only when it is not known yet.
        dimension = self._hamiltonian_dimension
        if dimension is None:
            dimension = self.evaluate_hamiltonian(pulse_initial).shape[0]
        validate_state_index(initial_state, dimension, "initial_state")
        validate_state_index(final_state, dimension, "final_state")

        return _ControlParameters(
            control_name=control_name,
            pulse_initial=pulse_initial,
            pulse_final=pulse_final,
            num_steps=num_steps,
            initial_state=initial_state,
            final_state=final_state,
            final_state_explicit=self._final_state_explicit,
            alpha=alpha,
            beta=beta,
            dia_alpha=dia_alpha,
            dia_beta=dia_beta,
        )

    def _generate_summary(self) -> str:
        """
        Generate a summary string of the current control parameters and settings. This method creates a formatted
        string that provides a clear overview of the control parameters, their values, and any relevant settings for
        the optimization problem. The summary can be used for logging, debugging, or displaying the current state of the
        ControlModel object.
        """
        if self._affine_hamiltonian:
            hamiltonian_description = "✅ affine (H_d + control * H_c)"
            partial_hamiltonian_description = "✅ exact constant H_c"
            hamiltonian_params = "absorbed into H_d and H_c"
        else:
            hamiltonian_description = "✅ function" if self.H_func is not None else "❌ not set"
            partial_hamiltonian_description = (
                "✅ function" if self.partial_H_func is not None else "numerical derivative"
            )
            hamiltonian_params = (
                ", ".join(f"{key}: {value}" for key, value in self._parameters.items())
                if self._parameters
                else "❌ not set"
            )
        alpha_beta = (
            f"({self.alpha if self.alpha is not None else '❌ not set'}, "
            f"{self.beta if self.beta is not None else '❌ not set'})"
        )
        diabatic_alpha_beta = (
            "("
            f"{self.dia_alpha if self.dia_alpha is not None else '❌ not set'}, "
            f"{self.dia_beta if self.dia_beta is not None else '❌ not set'}"
            ")"
        )

        summary_lines = [
            "------------------ ControlModel Control Summary ------------------",
            f"Hamiltonian: {hamiltonian_description}",
            f"Partial Hamiltonian: {partial_hamiltonian_description}",
            f"Hamiltonian parameters: {hamiltonian_params}",
            f"Control name → {self.control_name if self.control_name is not None else '❌ not set'}",
            f"Pulse initial → {self.pulse_initial if self.pulse_initial is not None else '❌ not set'}",
            f"Pulse final → {self.pulse_final if self.pulse_final is not None else '❌ not set'}",
            f"Initial state index → {self.initial_state if self.initial_state is not None else '❌ not set'}",
            f"Final state index → {self.final_state if self.final_state is not None else '❌ not set'}",
            f"(Alpha, Beta) → {alpha_beta}",
            f"(Diabatic Alpha, Diabatic Beta) → {diabatic_alpha_beta}",
            f"Eigenproblem solved → {'✅ yes' if self._flags['eigenproblem_solved'] else '❌ no'}",
            f"Metric computed → {'✅ yes' if self._flags['metric_computed'] else '❌ no'}",
            f"ODE solved → {'✅ yes' if self._flags['ode_solved'] else '❌ no'}",
            "---------------------------------------------------------------",
        ]
        return "\n".join(summary_lines)

    def print_summary(self) -> None:
        """
        Print a summary of the current control parameters and settings. This method provides a clear overview of the
        control parameters, their values, and any relevant settings for the optimization problem.
        """
        print(self._generate_summary())

    def __str__(self) -> str:
        return (
            f"ControlModel(control_name={self.control_name}, pulse_initial={self.pulse_initial}, "
            f"pulse_final={self.pulse_final}, initial_state={self.initial_state}, final_state={self.final_state}, "
            f"alpha={self.alpha}, beta={self.beta}, dia_alpha={self.dia_alpha}, dia_beta={self.dia_beta}, "
            f"num_steps={self.num_steps})"
        )

    def __repr__(self) -> str:
        return self._generate_summary()

    @staticmethod
    def _validate_control_name(name: Any) -> str:
        if not isinstance(name, str) or not name.strip():
            raise InvalidControlParameterError("Control name must be a non-empty string.")
        return name

    @staticmethod
    def _validate_constant_hamiltonian(matrix: Any, name: str) -> np.ndarray:
        """Validate and detach a constant Hamiltonian matrix."""
        array = np.asarray(matrix)

        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValidationError(f"{name} must be a 2D square matrix, got shape {array.shape}.")
        if not np.issubdtype(array.dtype, np.number):
            raise ValidationError(f"{name} must contain numeric values.")
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"{name} must contain only finite values.")
        if not is_hermitian(array):
            raise ValidationError(f"{name} must be Hermitian.")

        return array.copy()

    @staticmethod
    def _validate_pulse_value(value: Any, name: str) -> float:
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError(f"{name} value must be a number.")

        value = float(value)

        if not np.isfinite(value):
            raise InvalidControlParameterError(f"{name} value must be finite.")

        return value

    @staticmethod
    def _validate_state_index(index: Any, name: str) -> int:
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise InvalidControlParameterError(f"{name} state index must be an integer.")

        index = int(index)

        if index < 0:
            raise InvalidControlParameterError(f"{name} state index must be non-negative.")

        return index

    @staticmethod
    def _validate_exponent(value: Any, name: str) -> float:
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError(f"{name} must be a number.")

        value = float(value)

        if not np.isfinite(value):
            raise InvalidControlParameterError(f"{name} must be a finite number.")

        return value

    @staticmethod
    def _validate_num_steps(value: Any) -> int:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise InvalidControlParameterError("Number of steps must be an integer >= 3.")

        value = int(value)

        if value < 3:
            raise InvalidControlParameterError("Number of steps must be an integer >= 3.")

        return value
