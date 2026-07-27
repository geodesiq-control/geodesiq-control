from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import romb, solve_ivp
from scipy.interpolate import PchipInterpolator

from ._utils import Flags, build_diab, values_equal
from .exceptions import (ImmutableConfigurationError, InvalidControlParameterError, MissingControlParameterError,
                         SolverError, ValidationError, MetricComputationError, )
from .pulses import PulseControl


@dataclass(frozen=True)
class _EigensystemParameters:
    control_name: str
    pulse_initial: float
    pulse_final: float
    num_steps: int


@dataclass(frozen=True)
class _ControlParameters(_EigensystemParameters):
    initial_state: int
    final_state: int
    alpha: float
    beta: float
    dia_alpha: float | None
    dia_beta: float | None


class ControlModel:
    """
    A class to represent a parameter-dependent ControlModel and solve the optimization problem for finding the optimal
    control pulse. The ControlModel is defined as a function of a control parameter (e.g., lambda) and can also depend
    on other parameters. The class provides methods to set the parameters, compute the metric tensor, solve the ODE
    for the control pulse, and synthesize the control pulse based on the solution of the optimization problem.
    """

    _SINGULARITY_RTOL = 1e-12

    def __init__(self, H_func: Callable[..., np.ndarray], partial_H_func: Callable[..., np.ndarray] | None = None,
                 _flags_verbose: bool = False, ) -> None:
        """
        Initialize the ControlModel class with the Hamiltonian function and its partial derivative (if provided).

        Parameters
        ----------
        H_func : Callable[..., np.ndarray]
            A function that takes the control parameter and other parameters as input and returns the Hamiltonian
            matrix as a numpy array. The function should be defined such that it can accept the control parameter as a
            keyword argument, e.g., H_func(lambda=..., param1=..., param2=..., ...).
        partial_H_func : Callable[..., np.ndarray] | None
            An optional function that takes the control parameter and other parameters as input and returns the partial
            derivative of the Hamiltonian with respect to the control parameter as a numpy array.
            If this function is not provided, the class will compute the numerical partial derivative.
        """
        if not callable(H_func):
            raise ValidationError("H_func must be callable.")
        if partial_H_func is not None and not callable(partial_H_func):
            raise ValidationError("partial_H_func must be callable when provided.")

        self._H_func = H_func
        self._partial_H_func = partial_H_func

        # Initialize flags needed to track the state of the computations
        self._flags = Flags(_verbose=_flags_verbose)
        self._flags.add("eigenproblem_solved")
        self._flags.add("dia_list_computed")
        self._flags.add("metric_computed", parents=["eigenproblem_solved", "dia_list_computed"])
        self._flags.add("ode_solved", parent="metric_computed")

        if self._partial_H_func is not None:
            self._flag_numerical_partial_H = False
        else:
            self._flag_numerical_partial_H = True

        # Initialize parameters and control settings
        self._parameters: dict[str, Any] = {}
        self._control_name: str | None = None
        self._pulse_initial: float | None = None
        self._pulse_final: float | None = None
        self._initial_state: int | None = None
        self._final_state: int | None = None
        self._alpha: float | None = None
        self._beta: float | None = None
        self._dia_alpha: float | None = None
        self._dia_beta: float | None = None
        self._num_steps: int | None = None

        # Initialize energy gaps and matrix elements to None (to be computed in self.solve_problem())
        self._energies: np.ndarray | None = None
        self._matrix_elements: np.ndarray | None = None

        # Initialize metric tensor and normalization factor to None (to be computed in self.solve_problem())
        self._dia_list: np.ndarray | None = None
        self._metric_tensor: np.ndarray | None = None
        self._a_tilde: float | None = None

        # Initialize pulse parameters
        self._s: np.ndarray | None = None
        self._control_pulse: np.ndarray | None = None
        self._control_sol: np.ndarray | None = None
        self._pulse: PulseControl | None = None

        # Numerical integration configuration (user-overridable in solve_problem).
        self._solver = solve_ivp
        self._solver_kwargs: dict[str, Any] = {}
        self._metric_integrator = romb
        self._metric_integrator_kwargs: dict[str, Any] = {}
        self._previous_pulse_accuracy: int | None = None  # To track changes in pulse accuracy for ODE solving
        self._hamiltonian_dimension: int | None = None

    def _call_hamiltonian(self, *args: Any, **kwargs: Any) -> np.ndarray:
        matrix = np.asarray(self.H_func(*args, **{**self._parameters, **kwargs}))

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValidationError(f"H_func must return a 2D square matrix, got shape {matrix.shape}.")

        dimension = matrix.shape[0]

        if self._hamiltonian_dimension is None:
            self._hamiltonian_dimension = dimension
        elif dimension != self._hamiltonian_dimension:
            raise ValidationError("The Hamiltonian dimension cannot change after ControlModel "
                                  f"initialization: expected {self._hamiltonian_dimension}, "
                                  f"got {dimension}.")

        if not np.allclose(matrix, matrix.T.conj()):  # Hermitian
            raise ValidationError("H_func must return a Hermitian matrix.")

        if not np.all(np.isfinite(matrix)):  # Non-finite values
            raise ValidationError("H_func must return a matrix with finite values.")

        return matrix

    def _call_partial_hamiltonian(self, *args: Any, **kwargs: Any) -> np.ndarray:
        partial_H_func = self.partial_H_func
        if partial_H_func is None:
            raise MissingControlParameterError("partial_H_func is not configured.")
        matrix = np.asarray(partial_H_func(*args, **{**self._parameters, **kwargs}))

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValidationError(f"partial_H_func must return a 2D square matrix, got shape {matrix.shape}.")

        if not np.allclose(matrix, matrix.T.conj()):  # Hermitian
            raise ValidationError("partial_H_func must return a Hermitian matrix.")

        if not np.all(np.isfinite(matrix)):  # Non-finite values
            raise ValidationError("partial_H_func must return a matrix with finite values.")

        hamiltonian_temp = self.H_func(*args, **{**self._parameters, **kwargs})
        if np.shape(matrix) != hamiltonian_temp.shape:
            raise ValidationError("partial_H_func must return a matrix with the same shape as H_func.")

        return matrix

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        # Return the ControlModel function if the object is called directly, allowing for easy evaluation of the
        #   ControlModel at specific control values
        # Runtime kwargs override stored defaults for explicit one-off evaluations.
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
        """Evaluate and validate the Hamiltonian at one control value."""

        return self._call_hamiltonian(**self._evaluation_kwargs(control_value))

    # Setters and getters are defined for the critical attributes of the ControlModel class, so we have control over how
    # the user modifies them. It is important that the correct flags are updated when these attributes are changed
    @property
    def H_func(self) -> Callable[..., np.ndarray]:
        return self._H_func

    @H_func.setter
    def H_func(self, func: Callable[..., np.ndarray]) -> None:
        if self._H_func is None:
            if not callable(func):
                raise ValidationError("H_func must be callable.")
            self._H_func = func
        else:
            raise ImmutableConfigurationError("H_func is already set and cannot be changed. If you want to change it,"
                                              " please create a new instance of the ControlModel class.")

    @property
    def partial_H_func(self) -> Callable[..., np.ndarray] | None:
        return self._partial_H_func

    @partial_H_func.setter
    def partial_H_func(self, func: Callable[..., np.ndarray]) -> None:
        if self._partial_H_func is None:
            if not callable(func):
                raise ValidationError("partial_H_func must be callable.")
            self._partial_H_func = func
            self._flag_numerical_partial_H = False  # Update the numerical partial flag
            self._flags["eigenproblem_solved"] = False  # Reset the eigenproblem solved flag
        else:
            raise ImmutableConfigurationError(
                "partial_H_func is already set and cannot be changed. If you want to change it,"
                " please create a new instance of the ControlModel class.")

    @property
    def control_name(self) -> str | None:
        return self._control_name

    @control_name.setter
    def control_name(self, name: str | None) -> None:
        if name is None:  # Keep the previous value
            return
        if not isinstance(name, str) or not name.strip():
            raise InvalidControlParameterError("Control name must be a non-empty string.")
        if name in self._parameters:
            raise InvalidControlParameterError(f"Control name {name!r} collides with a stored Hamiltonian parameter.")
        if name == self._control_name:  # Keep the previous value
            return
        self._control_name = name
        self._flags["eigenproblem_solved"] = False  # Reset the eigenproblem solved flag if the control name changes

    @property
    def pulse_initial(self) -> float | None:
        return self._pulse_initial

    @pulse_initial.setter
    def pulse_initial(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("pulse_initial value must be a number.")
        value = float(value)
        if not np.isfinite(value):
            raise InvalidControlParameterError("pulse_initial value must be finite.")
        if value == self._pulse_initial:
            return
        if self._pulse_final is not None and value == self._pulse_final:
            raise InvalidControlParameterError("pulse_initial and pulse_final values must be different.")
        self._pulse_initial = value
        self._flags["eigenproblem_solved"] = (False
                                              # Reset the eigenproblem solved flag if the pulse initial value changes
                                              )

    @property
    def pulse_final(self) -> float | None:
        return self._pulse_final

    @pulse_final.setter
    def pulse_final(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("pulse_final value must be a number.")
        value = float(value)
        if not np.isfinite(value):
            raise InvalidControlParameterError("pulse_final value must be finite.")
        if value == self._pulse_final:
            return
        if self._pulse_initial is not None and value == self._pulse_initial:
            raise InvalidControlParameterError("pulse_initial and pulse_final values must be different.")
        self._pulse_final = value
        self._flags["eigenproblem_solved"] = (False
                                              # Reset the eigenproblem solved flag if the pulse final value changes
                                              )

    @property
    def initial_state(self) -> int | None:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: int | None) -> None:
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise InvalidControlParameterError("Initial state index must be an integer.")
        value = int(value)
        if value < 0:
            raise InvalidControlParameterError("Initial state index must be non-negative.")
        if value == self._initial_state:
            return

        self._initial_state = value

        if self._final_state is None:  # If not final state, assume for the moment that is the same as the initial one
            self.final_state = value

        self._flags["metric_computed"] = False  # Reset the  metric computed flag if the initial state index changes
        self._flags["dia_list_computed"] = False  # Reset the diabatic computed flag if the pulse initial value changes
        self._dia_list = None

        # If the initial state is the same as the final state, we can mark the diabatic passage list as computed
        if self._initial_state == self._final_state:
            self._flags["dia_list_computed"] = True

    @property
    def final_state(self) -> int | None:
        return self._final_state

    @final_state.setter
    def final_state(self, value: int | None) -> None:
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise InvalidControlParameterError("Final state index must be an integer.")
        value = int(value)
        if value < 0:
            raise InvalidControlParameterError("Final state index must be non-negative.")
        if value == self._final_state:
            return

        self._final_state = value
        self._flags["metric_computed"] = False  # Reset the metric computed flag if the final state index changes
        self._flags["dia_list_computed"] = False  # Reset the diabatic computed flag if the pulse initial value changes
        self._dia_list = None

        # If the initial state is the same as the final state, we can mark the diabatic passage list as computed
        if self._initial_state == self._final_state:
            self._flags["dia_list_computed"] = True

    @property
    def alpha(self) -> float | None:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("Alpha must be a number.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise InvalidControlParameterError("Alpha must be a finite non-negative number.")
        if value == self._alpha:
            return

        self._alpha = value
        self._flags["metric_computed"] = False

    @property
    def beta(self) -> float | None:
        return self._beta

    @beta.setter
    def beta(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("Beta must be a number.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise InvalidControlParameterError("Beta must be a finite non-negative number.")
        if value == self._beta:
            return

        self._beta = value
        self._flags["metric_computed"] = False  # Reset the metric computed flag if beta changes

    @property
    def dia_alpha(self) -> float | None:
        return self._dia_alpha

    @dia_alpha.setter
    def dia_alpha(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("Diabatic alpha must be a number.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise InvalidControlParameterError("Diabatic alpha must be a finite non-negative number.")
        if value == self._dia_alpha:
            return

        self._dia_alpha = value
        self._flags["metric_computed"] = False  # Reset the metric computed flag if alpha changes

    @property
    def dia_beta(self) -> float | None:
        return self._dia_beta

    @dia_beta.setter
    def dia_beta(self, value: float | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise InvalidControlParameterError("Diabatic beta must be a number.")
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise InvalidControlParameterError("Diabatic beta must be a finite non-negative number.")
        if value == self._dia_beta:
            return

        self._dia_beta = value
        self._flags["metric_computed"] = False  # Reset the metric computed flag if beta changes

    @property
    def num_steps(self) -> int | None:
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int | None) -> None:
        if value is None:  # Keep the previous value
            return
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) < 3:
            raise InvalidControlParameterError("Number of steps must be an integer >= 3 to support interpolation.")
        value = int(value)
        if value == self._num_steps:
            return

        self._num_steps = value
        self._flags["eigenproblem_solved"] = False  # Reset the eigenproblem solved flag if the number of steps changes

    @property
    def control_sol(self) -> np.ndarray:
        if not self._flags["ode_solved"]:
            self.solve_problem()  # Attempt to solve the ODE if not already solved
        if self._control_sol is None:
            raise SolverError("The control solution is unavailable after solving.")
        return self._control_sol.copy()

    @property
    def eigenenergies(self) -> np.ndarray:
        """Return ControlModel eigenenergies, solving the eigenproblem if needed."""
        if not self._flags["eigenproblem_solved"]:
            self._check_eigensystem_parameters()
            self._solve_eigenproblem()
        if self._energies is None:
            raise SolverError("Eigenenergies are unavailable after eigensystem solution.")
        return self._energies.copy()

    @property
    def control_pulse(self) -> np.ndarray:
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
        """

        if self.control_name in parameters:
            raise InvalidControlParameterError(
                f"{self.control_name!r} is the control variable and cannot also be supplied as a fixed parameter.")

        new_params = {**self._parameters, **parameters}

        if values_equal(new_params, self._parameters):
            return

        self._parameters = new_params
        self._flags["eigenproblem_solved"] = False

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return a read-only copy of the Hamiltonian parameters."""
        return MappingProxyType(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self._parameters.items()})

    def set_control(self, control_name: str | None = None, pulse_initial: float | None = None,
                    pulse_final: float | None = None, initial_state: int | None = None, final_state: int | None = None,
                    alpha: float | None = None, beta: float | None = None, dia_alpha: float | None = None,
                    dia_beta: float | None = None, num_steps: int | None = None, ) -> None:
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
            The index of the initial state in the energy spectrum. This is used to compute the metric tensor and the
             ODE for the control pulse.
        final_state : int | None
            The index of the final state in the energy spectrum. This is used to compute the metric tensor and the ODE
            for the control pulse. If not provided, it will be assumed to be the same as the initial state.
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

        candidate_name = self.control_name if control_name is None else control_name
        if candidate_name is not None and candidate_name in self._parameters:
            raise InvalidControlParameterError(
                f"Control name {candidate_name!r} collides with a stored Hamiltonian parameter.")

        # Update the final endpoint first when needed to avoid transient equality during range changes.
        self.control_name = control_name
        if pulse_initial is not None and pulse_final is not None and pulse_initial == self._pulse_final:
            self.pulse_final = pulse_final
            self.pulse_initial = pulse_initial
        else:
            self.pulse_initial = pulse_initial
            self.pulse_final = pulse_final
        self.initial_state = initial_state
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dia_alpha = dia_alpha
        self.dia_beta = dia_beta

        # Keep explicit user value; otherwise lazily initialize to default on first configuration.
        if num_steps is None and self._num_steps is None:
            self.num_steps = 2 ** 10 + 1
        else:
            self.num_steps = num_steps

    def solve_problem(self, pulse_accuracy: int = 1000, solver: Callable[..., Any] | None = None,
                      solver_kwargs: dict[str, Any] | None = None, metric_integrator: Callable[..., Any] | None = None,
                      metric_integrator_kwargs: dict[str, Any] | None = None, ) -> None:
        """
        Solve the optimization problem to find the optimal control pulse. This method computes the metric tensor based
        on the energies and matrix elements of the ControlModel, and then solves the ODE for the control pulse using the
        computed metric tensor. Note that this method does not synthesize the pulse itself, for that you need to call
        the synthesize_pulse() method after solving the problem.

        Parameters
        ----------
        pulse_accuracy : int
            The number of points to use in the numerical solution of the ODE for the control pulse. Higher values will
             yield a more accurate solution but will also increase the computational cost.
        solver : Callable[..., Any] | None
            Callable used to integrate the ODE. Defaults to ``scipy.integrate.solve_ivp``.
            Expected signature is solver(fun, t_span, y0, t_eval=..., **kwargs), and the return
            value must expose ``t`` and ``y`` (or be a ``(t, y)`` tuple).
        solver_kwargs : dict[str, Any] | None
            Additional keyword arguments forwarded to ``solver``.
        metric_integrator : Callable[..., Any] | None
            Callable used to compute ``a_tilde`` from ``sqrt(metric_tensor)``. Defaults to
            ``scipy.integrate.romb``.
        metric_integrator_kwargs : dict[str, Any] | None
            Additional keyword arguments forwarded to ``metric_integrator``.
        """
        if not isinstance(pulse_accuracy, (int, np.integer)) or isinstance(pulse_accuracy, bool):
            raise InvalidControlParameterError("pulse_accuracy must be an integer >= 3.")
        pulse_accuracy = int(pulse_accuracy)
        if pulse_accuracy < 3:
            raise InvalidControlParameterError("pulse_accuracy must be an integer >= 3.")
        config = self._check_control_parameters()

        self._configure_integration(solver=solver, solver_kwargs=solver_kwargs, metric_integrator=metric_integrator,
                                    metric_integrator_kwargs=metric_integrator_kwargs, )

        self._solve_eigenproblem(config)
        self._compute_metric_tensor(config)

        self._solve_ode(pulse_accuracy)

    def _configure_integration(self, solver: Callable[..., Any] | None, solver_kwargs: dict[str, Any] | None,
                               metric_integrator: Callable[..., Any] | None,
                               metric_integrator_kwargs: dict[str, Any] | None, ) -> None:
        if solver_kwargs is not None and not isinstance(solver_kwargs, dict):
            raise ValidationError("solver_kwargs must be a dictionary.")
        if metric_integrator_kwargs is not None and not isinstance(metric_integrator_kwargs, dict):
            raise ValidationError("metric_integrator_kwargs must be a dictionary.")
        selected_solver = self._solver if solver is None else solver
        selected_metric_integrator = self._metric_integrator if metric_integrator is None else metric_integrator
        selected_solver_kwargs = self._solver_kwargs if solver_kwargs is None else dict(solver_kwargs)
        selected_metric_kwargs = (
            self._metric_integrator_kwargs if metric_integrator_kwargs is None else dict(metric_integrator_kwargs))
        if not callable(selected_solver):
            raise ValidationError("solver must be a callable integration function.")

        if not callable(selected_metric_integrator):
            raise ValidationError("metric_integrator must be a callable integration function.")

        integrator_changed = selected_metric_integrator is not self._metric_integrator
        kwargs_changed = selected_metric_kwargs != self._metric_integrator_kwargs

        if integrator_changed or kwargs_changed:
            self._metric_integrator = selected_metric_integrator
            self._metric_integrator_kwargs = dict(selected_metric_kwargs)
            self._flags["metric_computed"] = False
        if selected_solver is not self._solver or selected_solver_kwargs != self._solver_kwargs:
            self._solver = selected_solver
            self._solver_kwargs = dict(selected_solver_kwargs)
            self._flags["ode_solved"] = False

    def _solve_dia_list(self, config: _ControlParameters) -> None:
        """Compute the diabatic-passage list for a validated control configuration."""
        if self._flags["dia_list_computed"]:
            return

        if self._energies is None:
            raise SolverError("Eigenenergies are unavailable before computing diabatic passages.")
        dim = self._energies.shape[1]
        self._dia_list = build_diab(initial_state=config.initial_state, final_state=config.final_state, dim=dim)
        self._flags["dia_list_computed"] = True

    def _solve_eigenproblem(self, config: _EigensystemParameters | None = None) -> None:
        """Solve the Hamiltonian eigenproblem over a validated control grid."""
        if self._flags["eigenproblem_solved"]:
            return

        if config is None:
            config = self._check_eigensystem_parameters()

        self._control_pulse = np.linspace(config.pulse_initial, config.pulse_final, num=config.num_steps, dtype=float, )
        full_hamiltonian = np.stack([self.evaluate_hamiltonian(value) for value in self._control_pulse])
        dimension = full_hamiltonian.shape[1]

        # State indices are required for a full solve, but not for plotting/eigenenergy access.
        if isinstance(config, _ControlParameters):
            for label, index in (("initial_state", config.initial_state), ("final_state", config.final_state)):
                if index >= dimension:
                    raise InvalidControlParameterError(
                        f"{label}={index} is out of range for a {dimension}-dimensional Hamiltonian.")

        try:
            energies, eigenvectors = np.linalg.eigh(full_hamiltonian)
        except np.linalg.LinAlgError as exc:
            raise SolverError("Hamiltonian eigendecomposition failed.") from exc

        energies = np.asarray(energies, dtype=float)
        if not np.all(np.isfinite(energies)):
            raise SolverError("Hamiltonian eigendecomposition produced non-finite eigenenergies.")
        self._energies = energies

        if self._flag_numerical_partial_H:
            full_partial_H = self._compute_numerical_partial_H()
        else:
            full_partial_H = np.array(
                [self._call_partial_hamiltonian(**self._evaluation_kwargs(lam)) for lam in self._control_pulse])

        matrix_elements = np.abs(
            np.einsum("...ij,...jk,...kl->...il", eigenvectors.conj().transpose(0, 2, 1), full_partial_H,
                      eigenvectors, ))
        if not np.all(np.isfinite(matrix_elements)):
            raise SolverError("Hamiltonian derivative matrix elements contain non-finite values.")
        self._matrix_elements = matrix_elements
        self._flags["eigenproblem_solved"] = True

    def _metric_ratio(self, numerator: np.ndarray, denominator: np.ndarray, alpha: float, beta: float,
                      transition: tuple[int, int], ) -> np.ndarray:
        if alpha > 0:
            gap_scale = max(1.0, float(np.max(np.abs(self._energies)))) if self._energies is not None else 1.0
            tolerance = self._SINGULARITY_RTOL * gap_scale
            if np.any(denominator <= tolerance):
                raise MetricComputationError("Degenerate or near-degenerate energy gap encountered for transition "
                                             f"{transition}; provide a regularized model or avoid the degeneracy.")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = numerator ** beta / denominator ** alpha
        return np.asarray(ratio, dtype=float)

    def _compute_metric_tensor(self, config: _ControlParameters) -> None:
        """
        Compute the metric tensor G_tensor based on the energies and matrix elements of the ControlModel. If the
        eigenproblem has not been solved yet, solve it first to obtain the energies and matrix elements.
        """
        if self._flags["metric_computed"]:
            return

        if config.initial_state == config.final_state:
            self._compute_G_adiabatic(config)
        else:
            self._solve_dia_list(config)
            self._compute_G_diabatic(config)

        if self._metric_tensor is None or self._control_pulse is None:
            raise MetricComputationError("Metric tensor was not produced.")
        metric = np.asarray(self._metric_tensor, dtype=float)
        if metric.ndim != 1 or metric.shape != self._control_pulse.shape:
            raise MetricComputationError(
                f"Metric tensor must have shape {self._control_pulse.shape}; received {metric.shape}.")
        if not np.all(np.isfinite(metric)):
            raise MetricComputationError("Metric tensor contains NaN or infinite values.")

        scale = max(1.0, float(np.max(np.abs(metric))))
        tolerance = self._SINGULARITY_RTOL * scale
        if np.any(metric < -tolerance):
            raise MetricComputationError("Metric tensor contains negative values.")
        metric = np.maximum(metric, 0.0)
        if np.any(metric <= tolerance):
            locations = self._control_pulse[metric <= tolerance]
            sample = ", ".join(f"{value:.6g}" for value in locations[:3])
            raise MetricComputationError("Metric tensor is zero or numerically singular" + (
                f" near control value(s) {sample}." if sample else "."))

        dx = float(np.abs(self._control_pulse[1] - self._control_pulse[0]))
        metric_values = np.sqrt(metric)

        # scipy.integrate.romb requires sample count n = 2**k + 1.
        if self._metric_integrator is romb:
            n_samples = metric_values.size
            power_minus_one = n_samples - 1
            if power_minus_one <= 0 or (power_minus_one & (power_minus_one - 1)) != 0:
                raise InvalidControlParameterError(
                    f"num_steps={n_samples} is incompatible with romb. Use num_steps=2**k+1, "
                    "or pass a different metric_integrator to solve_problem(...).")

        try:
            result = self._metric_integrator(metric_values, dx=dx, **self._metric_integrator_kwargs)
        except TypeError as exc:
            raise ValidationError(
                "Invalid metric_integrator call: ensure it accepts arguments like (values, dx=..., **kwargs).") from exc
        try:
            self._a_tilde = float(result)
        except (TypeError, ValueError) as exc:
            raise MetricComputationError("Metric integrator must return one finite real scalar.") from exc
        if not np.isfinite(self._a_tilde) or self._a_tilde <= 0:
            raise MetricComputationError("Metric normalization must be finite and strictly positive.")
        self._flags["metric_computed"] = True

    def _compute_G_diabatic(self, config: _ControlParameters) -> None:
        if self._energies is None or self._matrix_elements is None or self._dia_list is None:
            raise SolverError("Diabatic metric prerequisites are unavailable.")
        if config.dia_alpha is None or config.dia_beta is None:
            raise MissingControlParameterError("Diabatic metric exponents are required.")

        num, dim = self._energies.shape
        metric = np.zeros(num, dtype=float)

        for m in range(dim):
            for n in range(dim):
                if n == m:
                    continue
                adiabatic = bool(self._dia_list[m, n])
                denominator = np.abs(self._energies[:, n] - self._energies[:, m])
                numerator = self._matrix_elements[:, m, n]
                metric += self._metric_ratio(numerator, denominator,
                                             alpha=config.alpha if adiabatic else config.dia_alpha,
                                             beta=config.beta if adiabatic else config.dia_beta, transition=(m, n), )
        self._metric_tensor = metric

    def _compute_G_adiabatic(self, config: _ControlParameters) -> None:
        """Compute the adiabatic contribution to the metric tensor."""
        if self._energies is None or self._matrix_elements is None:
            raise SolverError("Adiabatic metric prerequisites are unavailable.")

        num, dim = self._energies.shape
        metric = np.zeros(num, dtype=float)
        for state in range(dim):
            if state == config.initial_state:
                continue
            denominator = np.abs(self._energies[:, state] - self._energies[:, config.initial_state])
            numerator = self._matrix_elements[:, config.initial_state, state]
            metric += self._metric_ratio(numerator, denominator, alpha=config.alpha, beta=config.beta,
                                         transition=(config.initial_state, state), )
        self._metric_tensor = metric

    def _compute_numerical_partial_H(self, order: int = 8, ) -> np.ndarray:
        """
        Evaluate dH/dx at every point of a real-valued grid.

        H_func is assumed not to be vectorized: it accepts one scalar x
        and returns a real- or complex-valued Hamiltonian.

        Parameters
        ----------
        order: int
            Order of the finite-difference formula.

        Returns
        -------
        dH_dx
            Derivative with shape (n_points, *H_shape).
        """
        if self._control_pulse is None:
            raise ValueError("x_grid is unavailable before the eigenproblem grid is initialized.")

        x_grid = np.asarray(self._control_pulse, dtype=float)

        if x_grid.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")

        if x_grid.size < 2:
            raise ValueError("x_grid must contain at least two points.")

        if not np.all(np.isfinite(x_grid)):
            raise ValueError("x_grid must contain only finite values.")

        unique_x = np.unique(x_grid)

        if unique_x.size < 2:
            raise ValueError("x_grid must contain at least two distinct points.")

        initial_step = float(np.min(np.diff(unique_x)))

        tolerances = {"atol": 1e-10, "rtol": 1e-8, }

        # Scalar evaluation to determine the Hamiltonian shape.
        H_reference = np.asarray(self.evaluate_hamiltonian(float(x_grid[0])), dtype=np.complex128, )

        H_shape = H_reference.shape
        n_elements = H_reference.size

        def evaluate_and_pack(x_column: np.ndarray, ) -> np.ndarray:
            """
            Evaluate the non-vectorized Hamiltonian at one scalar x.

            x_column has shape (1,), since x is the only independent
            variable.
            """
            x = float(x_column[0])

            H = np.asarray(self.evaluate_hamiltonian(float(x)), dtype=np.complex128, )

            if H.shape != H_shape:
                raise ValueError(f"H_func returned inconsistent shapes: expected {H_shape}, got {H.shape}.")

            H_flat = H.ravel()

            # scipy.differentiate.jacobian expects real outputs.
            return np.concatenate((H_flat.real, H_flat.imag,))

        def packed_hamiltonian(x_array: np.ndarray, ) -> np.ndarray:
            """
            Adapt the scalar H_func to SciPy's vectorized interface.

            x_array has shape (1, ...), where the trailing axes contain
            evaluation points introduced by SciPy.
            """
            return np.apply_along_axis(evaluate_and_pack, axis=0, arr=x_array, )

        # Use one-sided differences at the two domain boundaries.
        x_min = np.min(x_grid)
        x_max = np.max(x_grid)

        step_direction = np.zeros_like(x_grid, dtype=int)
        step_direction[x_grid - x_min < initial_step] = 1
        step_direction[x_max - x_grid < initial_step] = -1

        result = jacobian(packed_hamiltonian, x_grid[np.newaxis, :], order=order, initial_step=initial_step,
                          step_direction=step_direction[np.newaxis, :], tolerances=tolerances, )

        success = np.asarray(result.success, dtype=bool)
        if not np.all(success):
            status = np.asarray(result.status)
            error = np.asarray(result.error, dtype=float)

            failed_points = np.any(~success, axis=tuple(range(success.ndim - 1)), )

            failed_indices = np.flatnonzero(failed_points)

            details = []
            for index in failed_indices[:5]:
                point_failed = ~success[..., index]
                statuses = np.unique(status[..., index][point_failed])

                point_errors = error[..., index][point_failed]
                finite_errors = point_errors[np.isfinite(point_errors)]
                max_error = float(np.max(finite_errors)) if finite_errors.size else np.nan

                details.append(f"x={x_grid[index]:.6g}: "
                               f"status={statuses.tolist()}, "
                               f"max_error={max_error:.3e}")

            raise SolverError("Numerical differentiation failed to converge at "
                              f"{failed_indices.size} control point(s): " + "; ".join(details))

        # result.df shape:
        # (2 * n_elements, 1, n_points)
        derivative = np.asarray(result.df[:, 0, :])

        if not np.all(np.isfinite(derivative)):
            raise SolverError("Numerical Hamiltonian derivative contains non-finite values.")

        dH_flat = derivative[:n_elements] + 1j * derivative[n_elements:]

        # Convert from (*H_shape, n_points) to
        # (n_points, *H_shape).
        dH_dx = dH_flat.reshape(H_shape + (x_grid.size,))

        return np.moveaxis(dH_dx, -1, 0)

    def _solve_ode(self, pulse_accuracy: int) -> None:
        """Solve the normalized geodesic ODE and validate the solver result."""
        if self._previous_pulse_accuracy != pulse_accuracy:
            self._flags["ode_solved"] = False
        if self._flags["ode_solved"]:
            return
        if self._control_pulse is None or self._metric_tensor is None or self._a_tilde is None:
            raise SolverError("ODE cannot be solved before the control pulse and metric tensor are available.")

        factor = self._a_tilde / np.sqrt(self._metric_tensor)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
            raise MetricComputationError("ODE speed factor is non-finite or non-positive.")

        order = np.argsort(self._control_pulse)
        interpolation = PchipInterpolator(self._control_pulse[order], factor[order], extrapolate=False, )

        direction = float(np.sign(self._control_pulse[-1] - self._control_pulse[0]))
        lower = float(np.min(self._control_pulse))
        upper = float(np.max(self._control_pulse))

        def model(_: float, y: np.ndarray) -> np.ndarray:
            clipped = np.clip(y, lower, upper)
            return np.asarray(direction * interpolation(clipped), dtype=float)

        s = np.linspace(0.0, 1.0, pulse_accuracy)
        kwargs = dict(self._solver_kwargs)
        if self._solver is solve_ivp:
            reserved = {"dense_output", "events", "t_eval"}.intersection(kwargs)
            if reserved:
                names = ", ".join(sorted(reserved))
                raise ValidationError(f"solver_kwargs must not override internally managed option(s): {names}.")
            kwargs = {"method": "RK45", "atol": 1e-8, "rtol": 1e-6, **kwargs}

            target = float(self._control_pulse[-1])

            class EndpointEvent:
                def __init__(self, target: float, direction: float) -> None:
                    self.target = target
                    self.terminal = True
                    self.direction = direction

                def __call__(self, _: float, y: np.ndarray) -> float:
                    return float(y[0] - self.target)

            endpoint_event = EndpointEvent(target, direction)

            try:
                sol = solve_ivp(model, [0.0, 10.0], [self._control_pulse[0]], dense_output=True, events=endpoint_event,
                                **kwargs, )
            except TypeError as exc:
                raise ValidationError("Invalid solve_ivp keyword arguments.") from exc
            if not sol.success:
                raise SolverError(str(sol.message))
            if not sol.t_events or sol.t_events[0].size == 0 or sol.sol is None:
                raise SolverError("ODE solution did not reach the requested final control value.")
            endpoint_time = float(sol.t_events[0][0])
            if not np.isfinite(endpoint_time) or endpoint_time <= 0:
                raise SolverError("ODE endpoint time must be finite and strictly positive.")
            t = s
            y = np.asarray(sol.sol(s * endpoint_time), dtype=float)
        else:
            try:
                sol = self._solver(model, [0.0, 1.0], [self._control_pulse[0]], t_eval=s, **kwargs)
            except TypeError as exc:
                raise ValidationError(
                    "Invalid solver call: ensure solver accepts (fun, t_span, y0, t_eval=..., **kwargs).") from exc
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
        if self._solver is solve_ivp:
            control[0] = self._control_pulse[0]
            control[-1] = self._control_pulse[-1]
        self._s = t.copy()
        self._control_sol = control.copy()
        self._previous_pulse_accuracy = pulse_accuracy
        self._flags["ode_solved"] = True

    def _check_eigensystem_parameters(self) -> _EigensystemParameters:
        """Validate and return the configuration required by the eigensystem calculation."""
        control_name = self.control_name
        pulse_initial = self.pulse_initial
        pulse_final = self.pulse_final
        num_steps = self.num_steps

        if control_name is None or pulse_initial is None or pulse_final is None or num_steps is None:
            missing_params = [name for name, value in (("control_name", control_name), ("pulse_initial", pulse_initial),
                                                       ("pulse_final", pulse_final), ("num_steps", num_steps),) if
                              value is None]
            raise MissingControlParameterError(
                f"Missing control parameters for eigensystem: {', '.join(missing_params)}. "
                "Please set them using set_control(...).")

        return _EigensystemParameters(control_name=control_name, pulse_initial=pulse_initial, pulse_final=pulse_final,
                                      num_steps=num_steps, )

    def plot_eigenvalues(self, fig: Any = None, ax: Any = None, legend: bool = True,
                         legend_kwargs: dict[str, Any] | None = None, xlabel: str | None = None,
                         ylabel: str | None = None, title: str | None = None, **plot_kwargs: Any, ) -> tuple[Any, Any]:
        """
        Plot ControlModel eigenvalues as a function of the control parameter.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : dict | None
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : str | None
            Label for the x-axis. Defaults to ``control_name`` when not provided.
        ylabel : str | None
            Label for the y-axis. Defaults to ``"Energy"`` when not provided.
        title : str | None
            Plot title. Defaults to ``"ControlModel Eigenvalues"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot`` for each energy branch.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        config = self._check_eigensystem_parameters()
        control_name = config.control_name

        if ax is None:
            import matplotlib.pyplot as plt

            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        elif fig is None:
            fig = ax.figure

        for level in range(self.eigenenergies.shape[1]):
            ax.plot(self._control_pulse, self.eigenenergies[:, level], label=f"E{level}", **plot_kwargs)

        ax.set_xlabel(control_name if xlabel is None else xlabel)
        ax.set_ylabel("Energy" if ylabel is None else ylabel)
        ax.set_title("Hamiltonian eigenenergies" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax

    def plot_metric_tensor(self, fig: Any = None, ax: Any = None, legend: bool = True,
                           legend_kwargs: dict[str, Any] | None = None, xlabel: str | None = None,
                           ylabel: str | None = None, title: str | None = None, **plot_kwargs: Any, ) -> tuple[
        Any, Any]:
        """
        Plot the metric tensor (G tensor) as a function of the control parameter.

        If the metric tensor is not available yet, it is computed from the current
        ControlModel/control configuration.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created.
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
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        config = self._check_control_parameters()
        self._solve_eigenproblem(config)
        self._compute_metric_tensor(config)
        control_name = config.control_name

        if self._control_pulse is None:
            raise ValidationError("Control pulse can not be None.")

        if self._metric_tensor is None:
            raise ValidationError("Metric tensor can not be None.")

        if ax is None:
            import matplotlib.pyplot as plt

            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        elif fig is None:
            fig = ax.figure

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
        pulse = PulseControl(self._control_sol, duration)
        self._pulse = pulse
        return pulse

    def _check_control_parameters(self) -> _ControlParameters:
        """Validate and return all parameters required for a complete solve."""
        control_name = self.control_name
        pulse_initial = self.pulse_initial
        pulse_final = self.pulse_final
        initial_state = self.initial_state
        final_state = self.final_state
        alpha = self.alpha
        beta = self.beta
        dia_alpha = self.dia_alpha
        dia_beta = self.dia_beta
        num_steps = self.num_steps

        if (
                control_name is None or pulse_initial is None or pulse_final is None or num_steps is None or initial_state is None or final_state is None or alpha is None or beta is None):  # noqa: E501
            missing_params = [name for name, value in (("control_name", control_name), ("pulse_initial", pulse_initial),
                                                       ("pulse_final", pulse_final), ("initial_state", initial_state),
                                                       ("final_state", final_state), ("alpha", alpha), ("beta", beta),
                                                       ("num_steps", num_steps),) if value is None]
            raise MissingControlParameterError(
                f"Missing control parameters: {', '.join(missing_params)}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)}).")

        if initial_state != final_state and (dia_alpha is None or dia_beta is None):
            missing_params = [name for name, value in (("dia_alpha", dia_alpha), ("dia_beta", dia_beta)) if
                              value is None]
            raise MissingControlParameterError(
                f"Missing control parameters: {', '.join(missing_params)}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)}).")

        return _ControlParameters(control_name=control_name, pulse_initial=pulse_initial, pulse_final=pulse_final,
                                  num_steps=num_steps, initial_state=initial_state, final_state=final_state,
                                  alpha=alpha, beta=beta, dia_alpha=dia_alpha, dia_beta=dia_beta, )

    def _generate_summary(self) -> str:
        """
        Generate a summary string of the current control parameters and settings. This method creates a formatted
        string that provides a clear overview of the control parameters, their values, and any relevant settings for
        the optimization problem. The summary can be used for logging, debugging, or displaying the current state of the
        ControlModel object.
        """
        hamiltonian_params = (", ".join(
            f"{key}: {value}" for key, value in self._parameters.items()) if self._parameters else "❌ not set")
        alpha_beta = (f"({self.alpha if self.alpha is not None else '❌ not set'}, "
                      f"{self.beta if self.beta is not None else '❌ not set'})")
        diabatic_alpha_beta = ("("
                               f"{self.dia_alpha if self.dia_alpha is not None else '❌ not set'}, "
                               f"{self.dia_beta if self.dia_beta is not None else '❌ not set'}"
                               ")")

        summary_lines = ["------------------ ControlModel Control Summary ------------------",
                         f"Hamiltonian: {'✅ set' if self.H_func is not None else '❌ not set'}",
                         f"Partial Hamiltonian: {'✅ set' if self.partial_H_func is not None else '❌ not set'}",
                         f"Hamiltonian parameters: {hamiltonian_params}",
                         f"Control name → {self.control_name if self.control_name is not None else '❌ not set'}",
                         f"Pulse initial → {self.pulse_initial if self.pulse_initial is not None else '❌ not set'}",
                         f"Pulse final → {self.pulse_final if self.pulse_final is not None else '❌ not set'}",
                         f"Initial state index → "
                         f"{self.initial_state if self.initial_state is not None else '❌ not set'}",
                         f"Final state index → {self.final_state if self.final_state is not None else '❌ not set'}",
                         f"(Alpha, Beta) → {alpha_beta}", f"(Diabatic Alpha, Diabatic Beta) → {diabatic_alpha_beta}",
                         f"Eigenproblem solved → {'✅ yes' if self._flags['eigenproblem_solved'] else '❌ no'}",
                         f"Metric computed → {'✅ yes' if self._flags['metric_computed'] else '❌ no'}",
                         f"ODE solved → {'✅ yes' if self._flags['ode_solved'] else '❌ no'}",
                         "---------------------------------------------------------------", ]
        return "\n".join(summary_lines)

    def print_summary(self) -> None:
        """
        Print a summary of the current control parameters and settings. This method provides a clear overview of the
        control parameters, their values, and any relevant settings for the optimization problem.
        """
        print(self._generate_summary())

    def __str__(self) -> str:
        return (f"ControlModel(control_name={self.control_name}, pulse_initial={self.pulse_initial}, "
                f"pulse_final={self.pulse_final}, initial_state={self.initial_state}, alpha={self.alpha}, "
                f"beta={self.beta}, num_steps={self.num_steps})")

    def __repr__(self) -> str:
        return self._generate_summary()
