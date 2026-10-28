import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import qutip as qt
from numpy.typing import ArrayLike, NDArray
from qutip.solver.integrator import IntegratorException

from ._utils import figure_and_axes, is_hermitian, limit_blas_threads, real_samples, validate_state_index
from .controlmodel import ControlModel
from .decompose_hamiltonian import decompose_hamiltonian
from .exceptions import ConfigurationError, MissingArgsError, SolverError, ValidationError
from .warnings import GeodesiQWarning, NumericalStabilityWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


class Dynamics:
    """Simulate closed- or open-system evolution of a pulsed ControlModel.

    Use the model's optimal pulse rescaled to a physical duration, or supply a custom
    sampled pulse. Methods compute evolution operators, fidelities and state populations;
    collapse operators enable Lindblad evolution for state observables.
    """

    # A custom pulse whose endpoint is further than this fraction of the control range from the model's boundary
    # control triggers a warning.
    _ENDPOINT_RTOL = 1e-2
    # A selected endpoint eigenstate whose gap to a neighbouring level is below this fraction of the spectral
    # bandwidth is treated as degenerate, and a warning is emitted.
    _DEGENERACY_RTOL = 1e-8

    def __init__(
        self,
        model: ControlModel,
        duration: float | None = None,
        *,
        times: ArrayLike | None = None,
        pulse: ArrayLike | None = None,
        hbar: float = 1.0,
        order: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Dynamics object, which depends on an instance of the ControlModel class. This class deals with
        observables due to the time evolution of the pulsed ControlModel.

        The control pulse driving the evolution can be given in one of two (mutually exclusive) ways:

        - ``duration`` only: the optimal pulse solved by ``model`` is rescaled to the physical duration t_f.
        - ``times`` and ``pulse``: an arbitrary pulse sampled at the given physical times is used, e.g. the output of
          ``PulseControl.filtered_pulse()`` or ``PulseControl.discretized_pulse()``. The model then only provides the
          Hamiltonian and the eigenstate boundary conditions.

        Parameters
        ----------
        model : ControlModel
            An instance of the ControlModel class containing the Hamiltonian and the system parameters. Its
            Hamiltonian (copies of ``H_d``/``H_c``, or of the fixed parameters and ``control_name``), state indices and
            pulse are captured here, so later changes to the model do not affect this object; create a new
            ``Dynamics`` to use them. State kept inside ``H_func`` itself (e.g. in a closure) is not copied.
            Index-based and default states are eigenstates at the model's ``pulse_initial`` and ``pulse_final``, also
            for a custom pulse; without them, at the first and last pulse values.
        duration : float | None
            Duration of the optimal control pulse (t_f). Must not be combined with ``times`` and ``pulse``.
        times : ArrayLike | None
            Strictly increasing physical times at which ``pulse`` is sampled. Must be given together with ``pulse``.
        pulse : ArrayLike | None
            Control pulse values sampled at ``times``. Must be given together with ``times``.
        hbar : float
            Reduced Planck's constant (default is 1).
        order : int | None
            Interpolation of the pulse between its samples, for the propagation and for every Hamiltonian evaluated
            between samples: ``0`` holds each sample until the next one (as an arbitrary waveform generator does), ``1``
            interpolates linearly, and ``3`` uses a cubic spline. The default None uses ``3`` for the optimal pulse
            (``duration``), which is densely sampled and smooth, and ``1`` for a custom pulse (``times``/``pulse``), so
            that e.g. the piecewise-linear ``discretized_pulse()`` is simulated as the pulse it describes. A cubic
            spline through a few samples can deviate strongly between them and overshoot the sampled range.
        options : dict[str, Any] | None
            QuTiP solver options shared by all evolution methods, including ``atol``, ``rtol``, ``nsteps``,
            ``max_step`` and ``method``. None keeps QuTiP's numerical defaults. The dictionary is deep-copied;
            required output-storage settings are overridden by each method. Supported options depend on the
            integration method and are checked by QuTiP when evolution is requested. ``state_fidelity`` and the
            final-only propagator output only the end of the pulse; since ``nsteps`` bounds the steps between two
            outputs, it is then multiplied by the number of sample intervals (``len(times) - 1``).

        Notes
        -----
        For precision studies, tighten ``atol`` and ``rtol`` and check convergence of the observable. Limit
        ``max_step`` to less than half the narrowest pulse feature; sample times alone do not bound internal steps.
        QuTiP's default state normalization can hide norm drift; use ``normalize_output=False`` to inspect it.
        Propagated operators are not normalized. Solver options do not control pulse interpolation or SVD truncation.

        Raises
        ------
        ValidationError
            If ``model`` is not a ControlModel, pulse inputs or scalar settings are invalid, or optimal and custom
            pulse inputs are combined.
        MissingArgsError
            If neither pulse source is provided or only one of ``times`` and ``pulse`` is supplied.
        ConfigurationError
            If the model's required pulse or Hamiltonian configuration is unavailable. With ``duration``, the model is
            solved if needed: missing control parameters raise ``MissingControlParameterError``, and errors of
            ``ControlModel.solve_problem()`` (e.g. ``SolverError``) propagate unchanged.

        Warns
        -----
        GeodesiQWarning
            If a custom pulse starts or ends further than 1% of the control range from the model's ``pulse_initial``
            or ``pulse_final``. The fidelity against index-based states then also measures this endpoint mismatch.
        """

        if not isinstance(model, ControlModel):
            raise ValidationError("model must be an instance of ControlModel.")

        if options is not None and (not isinstance(options, dict) or not all(isinstance(key, str) for key in options)):
            raise ValidationError("options must be a dictionary with string keys or None.")
        self._options: dict[str, Any] = deepcopy(options) if options is not None else {}

        if not isinstance(hbar, (int, float, np.integer, np.floating)) or isinstance(hbar, bool):
            raise ValidationError("hbar must be a finite positive number.")
        try:
            hbar = float(hbar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("hbar must be a finite positive number.") from exc
        if not np.isfinite(hbar) or hbar <= 0:
            raise ValidationError("hbar must be a finite positive number.")
        self._hbar: float = hbar

        if order is not None and (
            not isinstance(order, (int, np.integer)) or isinstance(order, bool) or int(order) not in (0, 1, 3)
        ):
            raise ValidationError("order must be None, 0 (hold), 1 (linear) or 3 (cubic spline).")
        custom_pulse = times is not None or pulse is not None
        self._order: int = (1 if custom_pulse else 3) if order is None else int(order)

        self._pulse_times: NDArray[np.float64]
        self._pulse: NDArray[np.float64]
        self._duration: float
        # Control values whose eigenstates define the default initial and target states.
        self._boundary_controls: tuple[float, float]

        if custom_pulse:
            if duration is not None:
                raise ValidationError(
                    "Provide either duration (optimal pulse) or times and pulse (custom pulse), not both."
                )
            if times is None or pulse is None:
                raise MissingArgsError("times and pulse must be provided together.")

            self._pulse_times, self._pulse = self._validate_custom_pulse(times, pulse)
            self._duration = float(self._pulse_times[-1] - self._pulse_times[0])

            # Eigenstates are defined by the control problem of the model. Without one, use the pulse boundaries.
            if model.pulse_initial is not None and model.pulse_final is not None:
                self._boundary_controls = (float(model.pulse_initial), float(model.pulse_final))
                self._warn_moved_endpoints()
            else:
                self._boundary_controls = (float(self._pulse[0]), float(self._pulse[-1]))
        else:
            if duration is None:
                raise MissingArgsError(
                    "Provide either duration (optimal pulse) or times and pulse (custom pulse) to define the dynamics."
                )
            if not isinstance(duration, (int, float, np.integer, np.floating)) or isinstance(duration, bool):
                raise ValidationError("duration must be a finite positive number.")
            duration = float(duration)
            if not np.isfinite(duration) or duration <= 0:
                raise ValidationError("duration must be a finite positive number.")

            # Both properties solve the model if needed. They never return None: a missing configuration raises
            # MissingControlParameterError (a ConfigurationError), and a failed solve raises its own error.
            control_grid = np.asarray(model.control_pulse, dtype=float)
            self._pulse = np.asarray(model.control_sol, dtype=float)
            self._boundary_controls = (float(control_grid[0]), float(control_grid[-1]))
            self._duration = duration
            self._pulse_times = duration * np.linspace(0.0, 1.0, len(self._pulse), dtype=np.float64)

        # The pulse between samples, with the same interpolation as the propagated Hamiltonian.
        self._pulse_coefficient = qt.coefficient(self._pulse, tlist=self._pulse_times, order=self._order)

        # Snapshot of the model: later changes to it (parameters, control_name, states) do not reach this object, so
        # the propagation, the endpoint eigenstates and the adiabatic basis all use the same Hamiltonian (D-03).
        self._hamiltonian: Callable[[float], np.ndarray] = model._frozen_hamiltonian()
        self._initial_state: int | None = model.initial_state
        self._final_state: int | None = model.final_state
        if model.hamiltonian_dimension is None:
            self._hamiltonian(float(self._pulse[0]))  # Fixes the dimension of callable models
        self._hamiltonian_dimension: int | None = model.hamiltonian_dimension

        if model.affine_hamiltonian:
            H_d = model.H_d
            H_c = model.H_c

            if H_d is None or H_c is None:
                raise ConfigurationError("Affine ControlModel is missing its constant drift or control Hamiltonian.")

            self._qevo = qt.QobjEvo(
                [qt.Qobj(H_d) / self._hbar, [qt.Qobj(H_c) / self._hbar, self._pulse]],
                tlist=self._pulse_times,
                order=self._order,
            )
        else:
            with self._blas_threads():
                decomposition = decompose_hamiltonian(self._get_ham, self._pulse_times, drift="mean", rtol=1e-10)
            self._qevo = decomposition.qobjevo(order=self._order)

        # Eigenvectors of H at every pulse sample, computed on the first adiabatic populations() call (D-19).
        self._adiabatic_eigenvectors: NDArray[np.complex128] | None = None

    @staticmethod
    def _validate_custom_pulse(times: ArrayLike, pulse: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate a user-provided pulse and its sampling times."""
        time_values = np.asarray(times)
        if time_values.ndim != 1:
            raise ValidationError(f"times must be one-dimensional; received shape {time_values.shape}.")
        if time_values.size < 2:
            raise ValidationError("times must contain at least two samples.")
        if not np.issubdtype(time_values.dtype, np.number) or np.iscomplexobj(time_values):
            raise ValidationError("times must contain real-valued samples.")
        time_values = np.asarray(time_values, dtype=float)
        if not np.all(np.isfinite(time_values)):
            raise ValidationError("times contains NaN or infinite values.")
        if np.any(np.diff(time_values) <= 0):
            raise ValidationError("times must be strictly increasing.")

        pulse_values = np.asarray(pulse)
        if pulse_values.ndim != 1:
            raise ValidationError(f"pulse must be one-dimensional; received shape {pulse_values.shape}.")
        if pulse_values.shape != time_values.shape:
            raise ValidationError(
                f"pulse and times must have the same length; received {pulse_values.size} and {time_values.size}."
            )
        if not np.issubdtype(pulse_values.dtype, np.number):
            raise ValidationError("pulse must contain real-valued samples.")
        pulse_values = np.asarray(real_samples(pulse_values, "pulse"), dtype=float)
        if not np.all(np.isfinite(pulse_values)):
            raise ValidationError("pulse contains NaN or infinite values.")

        return time_values.copy(), pulse_values.copy()

    def _warn_moved_endpoints(self) -> None:
        """Warn when a custom pulse does not start or end at the model's boundary controls (D-14)."""
        control_initial, control_final = self._boundary_controls
        scale = max(abs(control_initial), abs(control_final), 1.0)
        tolerance = self._ENDPOINT_RTOL * abs(control_final - control_initial) + 100 * np.finfo(float).eps * scale
        start, end = float(self._pulse[0]), float(self._pulse[-1])
        if abs(start - control_initial) > tolerance or abs(end - control_final) > tolerance:
            warnings.warn(
                f"The pulse runs from {start:.6g} to {end:.6g}, but the model's boundary controls are "
                f"{control_initial:.6g} and {control_final:.6g}. Index-based (and default) initial and target states "
                "are the eigenstates at the boundary controls, so the fidelity also measures the endpoint mismatch.",
                GeodesiQWarning,
                stacklevel=3,
            )

    @property
    def times(self) -> NDArray[np.float64]:
        """Return a copy of the physical-time grid of the pulse."""
        return self._pulse_times.copy()

    @property
    def pulse(self) -> NDArray[np.float64]:
        """Return a copy of the control pulse samples driving the dynamics."""
        return self._pulse.copy()

    @property
    def duration(self) -> float:
        """Total duration of the pulse."""
        return self._duration

    def _blas_threads(self) -> Any:
        """Context limiting BLAS/LAPACK to one thread for small Hilbert spaces (see ``limit_blas_threads``)."""
        return limit_blas_threads(self._hamiltonian_dimension or 0)

    def _run_solver(self, solver: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Apply independent solver options, required storage settings and package errors."""
        options = deepcopy(self._options)
        options.update(kwargs.pop("options", {}))
        try:
            with self._blas_threads():
                return solver(*args, options=options, **kwargs)
        except IntegratorException as exc:
            raise SolverError(
                f"QuTiP time integration failed: {exc}. Adjust Dynamics(options=...), for example nsteps, "
                "atol, rtol, max_step or method."
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"QuTiP solver rejected the configuration or options: {exc}") from exc

    def _evaluate_hamiltonian(self, control_value: float) -> np.ndarray:
        """Evaluate the Hamiltonian of the model, as it was when this object was created, at one control value."""
        return self._hamiltonian(float(control_value))

    def _endpoint_options(self, solver: type[qt.SESolver]) -> dict[str, Any]:
        """
        Return the step budget for a run that outputs only ``[t0, tf]`` instead of every sample time (D-19).

        QuTiP's ``nsteps`` bounds the internal steps between two consecutive output times. It is therefore scaled by
        the number of sample intervals, which keeps the budget of a run through every sample time. The user's
        ``nsteps`` is scaled in the same way; methods without a step limit are left unchanged.
        """
        nsteps = self._options.get("nsteps")
        if nsteps is None:
            method = self._options.get("method", solver.solver_options.get("method"))
            integrator = solver.avail_integrators().get(method)
            nsteps = getattr(integrator, "integrator_options", {}).get("nsteps") if integrator is not None else None
        if nsteps is None:
            return {}
        return {"nsteps": int(nsteps) * max(1, len(self._pulse_times) - 1)}

    def _adiabatic_basis(self) -> NDArray[np.complex128]:
        """
        Return the eigenvectors of H at every pulse sample, shape ``(len(times), d, d)``, as columns sorted by energy.

        The Hamiltonian is captured at construction, so they are computed once per object and reused by every
        ``populations(basis="adiabatic")`` call, instead of re-evaluating and re-diagonalizing H each time.
        """
        if self._adiabatic_eigenvectors is None:
            hamiltonians = np.stack([np.asarray(self._evaluate_hamiltonian(value)) for value in self._pulse])
            with self._blas_threads():
                _, eigenvectors = np.linalg.eigh(hamiltonians)
            self._adiabatic_eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)
        return self._adiabatic_eigenvectors

    def _eigenstate(self, control_value: float, state_index: int) -> qt.Qobj:
        hamiltonian = qt.Qobj(self._evaluate_hamiltonian(control_value))
        with self._blas_threads():
            energies, eigenstates = hamiltonian.eigenstates()
        # Within a degenerate level the eigenvector is an arbitrary combination, and so is any fidelity against it.
        bandwidth = float(energies[-1] - energies[0])
        neighbours = energies[max(state_index - 1, 0) : state_index + 2]
        if neighbours.size > 1 and float(np.min(np.diff(neighbours))) <= self._DEGENERACY_RTOL * bandwidth:
            warnings.warn(
                f"Eigenstate {state_index} at control {control_value:.6g} is (nearly) degenerate with a neighbouring "
                "level, so it is an arbitrary vector of the degenerate subspace. Pass the state explicitly instead.",
                NumericalStabilityWarning,
                stacklevel=4,
            )
        return eigenstates[state_index]

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the Hamiltonian H(x(t))/hbar as a QuTiP Qobj, with the pulse interpolated between its samples in the
        same way as in the propagation (see ``order``).
        """
        control_val_t = float(np.real(self._pulse_coefficient(t)))

        return qt.Qobj(self._evaluate_hamiltonian(control_val_t)) / self._hbar

    def time_evolution_operator(self, final_only: bool = False) -> list[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse ControlModel.

        Parameters
        ----------
        final_only : bool
            If True, return only the propagator at the end of the pulse instead of one per time sample. This avoids
            storing ``len(times)`` dense matrices, which matters for large Hilbert spaces, and outputs nothing in
            between; the result agrees with the last full propagator within the solver tolerance.

        Returns
        -------
        propagators : list[qt.Qobj]
            Propagators U(t_i, t_0) for every time sample, or ``[U(t_f, t_0)]`` when ``final_only`` is True.

        Raises
        ------
        SolverError
            If QuTiP time integration fails. Configure solver settings through ``Dynamics(options=...)``.
        ValidationError
            If QuTiP rejects the solver configuration or options.
        """
        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()

        if final_only:
            # Only the end of the pulse is output. The sample times do not bound the solver steps (the Hamiltonian
            # carries the pulse), so this matches the last full propagator within the solver tolerance (D-19).
            identity = qt.qeye(self._qevo.dims[0])
            options = {"store_final_state": True, "store_states": False, **self._endpoint_options(qt.SESolver)}
            endpoints = [pulse_times[0], pulse_times[-1]]
            result = self._run_solver(qt.sesolve, self._qevo, identity, cast(Any, endpoints), options=options)
            if result.final_state is None:
                raise ValidationError("Time evolution did not return a final propagator.")
            return [result.final_state]

        propagator = self._run_solver(qt.propagator, self._qevo, cast(Any, pulse_times), options={"store_states": True})
        if isinstance(propagator, list):
            return propagator
        return [propagator]

    def state_fidelity(
        self,
        initial_state: np.ndarray | int | qt.Qobj | None = None,
        final_state: np.ndarray | int | qt.Qobj | None = None,
        c_ops: Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None = None,
    ) -> float:
        """
        Compute state transfer fidelity for closed- or open-system evolution.

        Without collapse operators the evolution is closed (QuTiP's ``mesolve`` then solves the Schrödinger equation
        for a ket). Supplying ``c_ops`` enables the Lindblad master equation. Integer state indices or omitted states
        select eigenstates at the pulse endpoints; explicit states are used directly.

        Parameters
        ----------
        initial_state : np.ndarray | int | qt.Qobj | None
            Initial state for pulsed time evolution: omitted (the model's ``initial_state`` index), an integer index
            (eigenstate at the start of the pulse), or an explicit state: a ket, bra or density matrix of the model
            dimension, as a numpy array or Qobj. Kets must be normalized and density matrices Hermitian with unit
            trace (within 1e-6); otherwise a ``ValidationError`` is raised.
        final_state : np.ndarray | int | qt.Qobj | None
            Target state, of any of the same kinds: omitted (the model's ``final_state`` index), an index (eigenstate
            at the end of the pulse) or an explicit state. The two arguments are resolved independently, so the kinds
            may be mixed.
        c_ops : Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None
            Collapse operators for the Lindblad master equation: a list or tuple of (d, d) Qobj, np.ndarray or
            QobjEvo, which may be mixed. None or an empty sequence gives closed evolution.

        Returns
        -------
        float
            Squared QuTiP (Uhlmann) fidelity between the target state ``sigma`` and the evolved final state ``rho``,
            ``F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2`` (the Jozsa fidelity). For pure states this is the
            transition probability ``|<target|psi_final>|**2``.

        Raises
        ------
        SolverError
            If QuTiP time integration fails. Configure solver settings through ``Dynamics(options=...)``.
        ValidationError
            If states, collapse operators or solver options are invalid.
        ConfigurationError
            If a model state index is needed but not configured, or the Hamiltonian dimension is unavailable.

        Warns
        -----
        NumericalStabilityWarning
            If an index-based (or default) state is degenerate with a neighbouring level, i.e. their gap is at most
            1e-8 times the spectral bandwidth. The eigenvector, and hence the fidelity, is then arbitrary within the
            degenerate subspace; pass the state explicitly.
        """
        dimension = self._dimension()
        collapse_operators = _as_collapse_operators(c_ops, dimension)

        # Only the final state is needed, so only the end of the pulse is output (D-19).
        endpoints = [float(self._pulse_times[0]), float(self._pulse_times[-1])]

        # Each state is resolved on its own, as in populations(), so the kinds may be mixed (D-15).
        psi_init = self._resolve_initial_state(initial_state)
        psi_target = self._resolve_target_state(final_state)

        options = {"store_final_state": True, "store_states": False, **self._endpoint_options(qt.MESolver)}

        result = self._run_solver(
            qt.mesolve,
            self._qevo,
            psi_init,
            cast(Any, endpoints),
            c_ops=cast(Any, collapse_operators),
            options=options,
        )

        psi_f = result.final_state
        if psi_f is None:
            raise ValidationError("Time evolution did not return a final state.")

        state_fidelity: float = qt.fidelity(psi_target, psi_f) ** 2

        return state_fidelity

    def average_gate_fidelity(
        self,
        gate: qt.Qobj | list[qt.Qobj] | None = None,
        *,
        target_gate: qt.Qobj | np.ndarray,
        final_only: bool = False,
    ) -> list[float]:
        """
        Compute average gate fidelity given the pulsed time evolution operator in the explicit real-time duration given.

        Parameters
        ----------
        gate : qt.Qobj | list[qt.Qobj] | None
            The resulting pulsed gate operation.
        target_gate : qt.Qobj | np.ndarray
            The target gate operation, with the dimension of the Hamiltonian. Required (keyword only): there is no
            default target, so that the fidelity is never computed against the identity by accident.
        final_only : bool
            Only used when ``gate`` is None: if True, compute the fidelity of the final propagator only (see
            ``time_evolution_operator``).

        Returns
        -------
        gate_fid : list[float]
            A list of average gate fidelities for each time step in the pulse duration (a single value when
            ``final_only`` is True).
        """

        dimension = self._dimension()
        target_gate = _as_operator(target_gate, dimension, "target_gate")

        if gate is None:
            operators = self.time_evolution_operator(final_only=final_only)
        elif isinstance(gate, qt.Qobj):
            operators = [_as_operator(gate, dimension, "gate")]
        elif isinstance(gate, list) and all(isinstance(g, qt.Qobj) for g in gate):
            operators = [_as_operator(g, dimension, f"gate[{index}]") for index, g in enumerate(gate)]
        else:
            raise ValidationError("Gate must be a Qobj or a list of Qobj instances.")

        gate_fid = [qt.average_gate_fidelity(oper=oper, target=target_gate) for oper in operators]

        return gate_fid

    def _resolve_initial_state(self, initial_state: np.ndarray | int | qt.Qobj | None) -> qt.Qobj:
        """Initial state of the evolution, following the conventions of ``state_fidelity``."""
        control_initial = self._boundary_controls[0]

        if initial_state is None:
            if self._initial_state is None:
                raise ConfigurationError(
                    "The initial state index must be configured in the ControlModel when no explicit initial state is "
                    "provided."
                )
            return self._eigenstate(control_initial, self._initial_state)

        dimension = self._dimension()
        if _is_index(initial_state):
            index = validate_state_index(cast(int, initial_state), dimension, "initial_state")
            return self._eigenstate(control_initial, index)

        return _as_state(initial_state, dimension, "initial_state")

    def _resolve_target_state(self, final_state: np.ndarray | int | qt.Qobj | None) -> qt.Qobj:
        """Target state, following the conventions of ``state_fidelity``: the eigenstates are at the final control."""
        control_final = self._boundary_controls[1]

        if final_state is None:
            if self._final_state is None:
                raise ConfigurationError(
                    "The final state index must be configured in the ControlModel when no explicit target state is "
                    "provided."
                )
            return self._eigenstate(control_final, self._final_state)

        dimension = self._dimension()
        if _is_index(final_state):
            index = validate_state_index(cast(int, final_state), dimension, "final_state")
            return self._eigenstate(control_final, index)

        return _as_state(final_state, dimension, "final_state")

    def _dimension(self) -> int:
        """Hilbert-space dimension of the model, which every explicit state and operator must match."""
        if self._hamiltonian_dimension is None:
            raise ConfigurationError("Hamiltonian dimension is unavailable.")
        return self._hamiltonian_dimension

    def populations(
        self,
        initial_state: np.ndarray | int | qt.Qobj | None = None,
        basis: Literal["adiabatic", "diabatic"] = "adiabatic",
        c_ops: Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute the population of every state at each time sample of the pulsed time evolution.

        Parameters
        ----------
        initial_state : np.ndarray | int | qt.Qobj | None
            Initial state of the evolution, with the same conventions as in ``state_fidelity``: None uses the initial
            state index of the ControlModel, an integer selects an eigenstate of the Hamiltonian at the initial control
            value, and a numpy array or Qobj is used as given (state vector or density matrix).
        basis : {"adiabatic", "diabatic"}
            Basis in which populations are measured. ``"adiabatic"`` uses the instantaneous eigenstates of the
            Hamiltonian at each time, sorted by increasing energy. ``"diabatic"`` uses the fixed basis in which the
            Hamiltonian matrix is written. Adiabatic populations of exactly degenerate levels are not unique.
        c_ops : Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None
            Collapse operators for the Lindblad master equation (see ``state_fidelity``).

        Returns
        -------
        populations : NDArray[np.float64]
            Array of shape ``(len(times), dimension)``, where ``populations[i, n]`` is the population of state ``n`` at
            ``times[i]``.

        Raises
        ------
        SolverError
            If QuTiP time integration fails. Configure solver settings through ``Dynamics(options=...)``.
        ValidationError
            If the basis, state, collapse operators or solver options are invalid.
        ConfigurationError
            If the model's initial state index is needed but not configured, or the Hamiltonian dimension
            is unavailable.

        Warns
        -----
        NumericalStabilityWarning
            If an index-based (or default) initial state is degenerate with a neighbouring level (see
            ``state_fidelity``).
        """
        if basis not in ("adiabatic", "diabatic"):
            raise ValidationError(f"basis must be 'adiabatic' or 'diabatic', got {basis!r}.")

        collapse_operators = _as_collapse_operators(c_ops, self._dimension())
        psi_init = self._resolve_initial_state(initial_state)

        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()
        options = {"store_states": True}

        result = self._run_solver(
            qt.mesolve,
            self._qevo,
            psi_init,
            cast(Any, pulse_times),
            c_ops=cast(Any, collapse_operators),
            options=options,
        )
        states = result.states
        if len(states) != len(pulse_times):
            raise ValidationError("Time evolution did not return a state for every time sample.")
        adiabatic_basis = self._adiabatic_basis() if basis == "adiabatic" else None

        with self._blas_threads():
            populations = np.empty((len(states), states[0].shape[0]))
            for i, state in enumerate(states):
                data = state.full()
                if adiabatic_basis is not None:
                    eigenvectors = adiabatic_basis[i]
                    data = eigenvectors.conj().T @ data
                    if not state.isket:
                        data = data @ eigenvectors

                if state.isket:
                    populations[i] = np.abs(data[:, 0]) ** 2
                else:
                    populations[i] = np.diagonal(data).real

        return populations

    def plot_populations(
        self,
        fig: "Figure | SubFigure | None" = None,
        ax: "Axes | None" = None,
        initial_state: np.ndarray | int | qt.Qobj | None = None,
        basis: Literal["adiabatic", "diabatic"] = "adiabatic",
        c_ops: Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None = None,
        states: Sequence[int] | None = None,
        legend: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        show: bool = True,
        **plot_kwargs: Any,
    ) -> "tuple[Figure | SubFigure, Axes]":
        """
        Plot the population of each state as a function of time during the pulsed time evolution.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created. A ``fig`` without ``ax`` is drawn on its
            single axis, or gets a new one if it has none; a figure with several axes needs ``ax``.
        initial_state : np.ndarray | int | qt.Qobj | None
            Initial state of the evolution (see ``populations``).
        basis : {"adiabatic", "diabatic"}
            Basis in which populations are measured (see ``populations``).
        c_ops : Sequence[qt.Qobj | qt.QobjEvo | np.ndarray] | None
            Collapse operators for the Lindblad master equation (see ``state_fidelity``).
        states : Sequence[int] | None
            Indices of the states to plot. All states are plotted when not provided.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : dict | None
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : str | None
            Label for the x-axis. Defaults to ``"Time $t$"`` when not provided.
        ylabel : str | None
            Label for the y-axis. Defaults to ``"Population"`` when not provided.
        title : str | None
            Plot title. Defaults to ``"Adiabatic populations"`` or ``"Diabatic populations"`` when not provided.
        show : bool
            Whether to call ``plt.show()``. Pass ``show=False`` to keep drawing on the figure.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot`` for each state.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        import matplotlib.pyplot as plt

        populations = self.populations(initial_state=initial_state, basis=basis, c_ops=c_ops)
        dimension = populations.shape[1]

        if states is None:
            indices = list(range(dimension))
        else:
            indices = [validate_state_index(index, dimension, "states") for index in states]

        fig, ax = figure_and_axes(fig, ax)

        for index in indices:
            label = rf"$|E_{{{index}}}\rangle$" if basis == "adiabatic" else rf"$|{index}\rangle$"
            ax.plot(self._pulse_times, populations[:, index], label=label, **plot_kwargs)

        ax.set_xlabel("Time $t$" if xlabel is None else xlabel)
        ax.set_ylabel("Population" if ylabel is None else ylabel)
        ax.set_title(f"{basis.capitalize()} populations" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        if show:
            plt.show()
        return fig, ax


def _is_index(value: Any) -> bool:
    """Whether a value is an integer state index (Python or NumPy integer, but not a bool)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


# Tolerance on the norm of an explicit ket, or the trace of an explicit density matrix.
_STATE_NORM_TOL = 1e-6


def _as_operator(value: Any, dimension: int, name: str) -> qt.Qobj:
    """Validate a (d, d) operator given as a Qobj or numpy array, and return it with the Hamiltonian's dimensions."""
    if not isinstance(value, (qt.Qobj, np.ndarray)):
        raise ValidationError(f"{name} must be a Qobj or a numpy array, got {type(value).__name__}.")
    array = value.full() if isinstance(value, qt.Qobj) else np.asarray(value)
    if array.shape != (dimension, dimension):
        raise ValidationError(f"{name} must have shape {(dimension, dimension)}, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite numbers.")
    return qt.Qobj(array)


def _as_state(value: Any, dimension: int, name: str) -> qt.Qobj:
    """
    Validate an explicit state given as a Qobj or numpy array: a ket (shape (d,) or (d, 1)), a bra (shape (1, d)),
    which is converted to a ket, or a density matrix (shape (d, d)). Kets must have unit norm and density matrices
    unit trace (within ``_STATE_NORM_TOL``), and density matrices must be Hermitian. The state is returned with the
    Hamiltonian's dimensions, so tensor-structured Qobj inputs are accepted.
    """
    if not isinstance(value, (qt.Qobj, np.ndarray)):
        raise ValidationError(f"{name} must be an integer index, a Qobj or a numpy array, got {type(value).__name__}.")
    array = value.full() if isinstance(value, qt.Qobj) else np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite numbers.")

    if array.shape in ((dimension,), (dimension, 1), (1, dimension)):
        ket = array.reshape(dimension, 1)
        if array.shape == (1, dimension):
            ket = ket.conj()
        norm = float(np.linalg.norm(ket))
        if abs(norm - 1.0) > _STATE_NORM_TOL:
            raise ValidationError(f"{name} must be normalized, got norm {norm:.6g}.")
        return qt.Qobj(ket)

    if array.shape == (dimension, dimension):
        trace = complex(np.trace(array))
        if abs(trace - 1.0) > _STATE_NORM_TOL:
            raise ValidationError(f"{name} as a density matrix must have unit trace, got {trace:.6g}.")
        if not is_hermitian(array):
            raise ValidationError(f"{name} as a density matrix must be Hermitian.")
        return qt.Qobj(array)

    raise ValidationError(
        f"{name} must be a ket of shape {(dimension,)} or {(dimension, 1)}, a bra of shape {(1, dimension)}, or a "
        f"density matrix of shape {(dimension, dimension)}; got {array.shape}."
    )


def _as_collapse_operators(c_ops: Any, dimension: int) -> list[qt.Qobj | qt.QobjEvo] | None:
    """Validate collapse operators (a list or tuple of (d, d) Qobj, numpy arrays or QobjEvo)."""
    if c_ops is None:
        return None
    if not isinstance(c_ops, (list, tuple)):
        raise ValidationError("Collapse operators must be provided as a list or tuple of Qobj or numpy arrays.")
    operators: list[qt.Qobj | qt.QobjEvo] = []
    for index, op in enumerate(c_ops):
        if isinstance(op, qt.QobjEvo):
            if tuple(op.shape) != (dimension, dimension):
                raise ValidationError(f"c_ops[{index}] must have shape {(dimension, dimension)}, got {op.shape}.")
            operators.append(op)
        else:
            operators.append(_as_operator(op, dimension, f"c_ops[{index}]"))
    return operators
