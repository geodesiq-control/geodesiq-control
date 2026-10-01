from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, cast

import numpy as np
import qutip as qt

from ._utils import limit_blas_threads, validate_state_index
from .controlmodel import ControlModel
from .decompose_hamiltonian import decompose_hamiltonian
from .exceptions import ConfigurationError, MissingArgsError, ValidationError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


class Dynamics:
    def __init__(
        self,
        model: ControlModel,
        duration: Optional[float] = None,
        *,
        times: Optional[np.ndarray] = None,
        pulse: Optional[np.ndarray] = None,
        hbar: float = 1.0,
    ):
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
        model: ControlModel
            An instance of the ControlModel class containing the Hamiltonian and the system parameters.
        duration : Optional[float]
            Duration of the optimal control pulse (t_f). Must not be combined with ``times`` and ``pulse``.
        times : Optional[np.ndarray]
            Strictly increasing physical times at which ``pulse`` is sampled. Must be given together with ``pulse``.
        pulse : Optional[np.ndarray]
            Control pulse values sampled at ``times``. Must be given together with ``times``.
        hbar : float
            Reduced Planck's constant (default is 1).
        """

        if not isinstance(model, ControlModel):
            raise ValidationError("model must be an instance of ControlModel.")

        if not isinstance(hbar, (int, float, np.integer, np.floating)) or isinstance(hbar, bool):
            raise ValidationError("hbar must be a finite positive number.")
        try:
            hbar = float(hbar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("hbar must be a finite positive number.") from exc
        if not np.isfinite(hbar) or hbar <= 0:
            raise ValidationError("hbar must be a finite positive number.")
        self._hbar: float = hbar

        self._pulse_times: np.ndarray
        self._pulse: np.ndarray
        self._duration: float
        # Control values whose eigenstates define the default initial and target states.
        self._boundary_controls: tuple[float, float]

        if times is not None or pulse is not None:
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

            # Attributes of the ControlModel instance
            if model.control_pulse is None or model.control_sol is None:
                raise ConfigurationError(
                    "Control pulse is unavailable. Solve the ControlModel before computing the dynamics."
                )

            control_grid = np.asarray(model.control_pulse, dtype=float)
            self._pulse = np.asarray(model.control_sol, dtype=float)
            self._boundary_controls = (float(control_grid[0]), float(control_grid[-1]))
            self._duration = duration
            self._pulse_times = duration * np.linspace(0.0, 1.0, len(self._pulse))

        self.evaluate_hamiltonian = lambda control_value: model.evaluate_hamiltonian(float(control_value))
        self._initial_state: int | None = model.initial_state
        self._final_state: int | None = model.final_state
        if model.hamiltonian_dimension is None:
            model.evaluate_hamiltonian(float(self._pulse[0]))  # Fixes the dimension of callable models
        self._hamiltonian_dimension: int | None = model.hamiltonian_dimension

        if model.affine_hamiltonian:
            H_d = model.H_d
            H_c = model.H_c

            if H_d is None or H_c is None:
                raise ConfigurationError("Affine ControlModel is missing its constant drift or control Hamiltonian.")

            self._qevo = qt.QobjEvo(
                [qt.Qobj(H_d) / self._hbar, [qt.Qobj(H_c) / self._hbar, self._pulse]],
                tlist=self._pulse_times,
                order=3,
            )
        else:
            with self._blas_threads():
                decomposition = decompose_hamiltonian(self._get_ham, self._pulse_times, drift="mean", rtol=1e-10)
            self._qevo = decomposition.qobjevo()

    @staticmethod
    def _validate_custom_pulse(times: np.ndarray, pulse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        if np.iscomplexobj(pulse_values):
            if not np.allclose(pulse_values.imag, 0.0):
                raise ValidationError("pulse must contain real-valued samples.")
            pulse_values = pulse_values.real
        pulse_values = np.asarray(pulse_values, dtype=float)
        if not np.all(np.isfinite(pulse_values)):
            raise ValidationError("pulse contains NaN or infinite values.")

        return time_values.copy(), pulse_values.copy()

    @property
    def times(self) -> np.ndarray:
        """Return a copy of the physical-time grid of the pulse."""
        return self._pulse_times.copy()

    @property
    def pulse(self) -> np.ndarray:
        """Return a copy of the control pulse samples driving the dynamics."""
        return self._pulse.copy()

    @property
    def duration(self) -> float:
        """Total duration of the pulse."""
        return self._duration

    def _blas_threads(self) -> Any:
        """Context limiting BLAS/LAPACK to one thread for small Hilbert spaces (see ``limit_blas_threads``)."""
        return limit_blas_threads(self._hamiltonian_dimension or 0)

    def _eigenstate(self, control_value: float, state_index: int) -> qt.Qobj:
        hamiltonian = qt.Qobj(self.evaluate_hamiltonian(control_value))
        with self._blas_threads():
            _, eigenstates = hamiltonian.eigenstates()
        return eigenstates[state_index]

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the time-dependent ControlModel using QuTiP Qobj
        """
        control_val_t = float(np.interp(t, self._pulse_times, self._pulse))

        return qt.Qobj(self.evaluate_hamiltonian(control_val_t)) / self._hbar

    def time_evolution_operator(self, final_only: bool = False) -> List[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse ControlModel.

        Parameters
        ----------
        final_only : bool
            If True, return only the propagator at the end of the pulse instead of one per time sample. This avoids
            storing ``len(times)`` dense matrices, which matters for large Hilbert spaces.

        Returns
        -------
        propagators : List[qt.Qobj]
            Propagators U(t_i, t_0) for every time sample, or ``[U(t_f, t_0)]`` when ``final_only`` is True.
        """
        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()

        if final_only:
            # Integrate through every sample time (as for the full propagator), but keep only the final operator.
            identity = qt.qeye(self._qevo.dims[0])
            options = {"store_final_state": True, "store_states": False}
            with self._blas_threads():
                result = qt.sesolve(self._qevo, identity, cast(Any, pulse_times), options=options)
            if result.final_state is None:
                raise ValidationError("Time evolution did not return a final propagator.")
            return [result.final_state]

        with self._blas_threads():
            propagator = qt.propagator(self._qevo, cast(Any, pulse_times))
        if isinstance(propagator, list):
            return propagator
        return [propagator]

    def state_fidelity(
        self,
        initial_state: Optional[np.ndarray | int | qt.Qobj] = None,
        final_state: Optional[np.ndarray | int | qt.Qobj] = None,
        c_ops: Optional[List[qt.Qobj] | List[np.ndarray]] = None,
    ) -> float:
        """
        Compute the state transfer fidelity with Lindblad master equation. Depending on whether initial/final states are
        explicitly given, then the time evolution is constructed. If integers of None are given then eigenstate
        evolution is assumed.

        Parameters
        ----------
        initial_state : Optional[np.ndarray, int or qt.Qobj]
            Initial state for pulsed time evolution.
        final_state : Optional[np.ndarray, int or qt.Qobj]
            Final state for pulsed time evolution.
        c_ops : Optional[list]
            Collapse operators (passed as a list of Qobj or np.ndarray) for the Lindblad master equation.

        """
        c_ops = _as_collapse_operators(c_ops)

        control_initial, control_final = self._boundary_controls

        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()

        if initial_state is None and final_state is None:
            initial_index = self._initial_state
            final_index = self._final_state

            if initial_index is None or final_index is None:
                raise ConfigurationError(
                    "Initial and final state indices must be configured in the ControlModel "
                    "when no explicit states are provided."
                )

            psi_init = self._eigenstate(control_initial, initial_index)
            psi_target = self._eigenstate(control_final, final_index)

        elif _is_index(initial_state) and _is_index(final_state):
            dimension = self._hamiltonian_dimension

            if dimension is None:
                raise ConfigurationError("Hamiltonian dimension is unavailable.")

            initial_index = validate_state_index(cast(int, initial_state), dimension, "initial_state")
            final_index = validate_state_index(cast(int, final_state), dimension, "final_state")

            psi_init = self._eigenstate(control_initial, initial_index)
            psi_target = self._eigenstate(control_final, final_index)

        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            if (
                initial_state.shape[0] != self._hamiltonian_dimension
                or final_state.shape[0] != self._hamiltonian_dimension
            ):
                raise ValidationError(
                    f"Initial and final states must have the same dimension as the ControlModel. Shape of ControlModel:"
                    f" {(self._hamiltonian_dimension, self._hamiltonian_dimension)}."
                    f" Shape of initial state: {initial_state.shape}"
                )

            psi_init = qt.Qobj(initial_state)
            psi_target = qt.Qobj(final_state)
        elif isinstance(initial_state, qt.Qobj) and isinstance(final_state, qt.Qobj):
            psi_init = initial_state
            psi_target = final_state
        else:
            raise ValidationError(
                "Initial and final states must be either integers, numpy arrays with correct dimensions or Qobj "
                "instances."
            )

        options = {"store_final_state": True, "store_states": False}

        with self._blas_threads():
            result = qt.mesolve(self._qevo, psi_init, cast(Any, pulse_times), c_ops=cast(Any, c_ops), options=options)

        psi_f = result.final_state
        if psi_f is None:
            raise ValidationError("Time evolution did not return a final state.")

        state_fidelity: float = qt.fidelity(psi_target, psi_f) ** 2

        return state_fidelity

    def average_gate_fidelity(
        self,
        gate: Optional[qt.Qobj | List[qt.Qobj]] = None,
        target_gate: Optional[qt.Qobj | np.ndarray] = None,
        final_only: bool = False,
    ) -> List[float]:
        """
        Compute average gate fidelity given the pulsed time evolution operator in the explicit real-time duration given.

        Parameters
        ----------
        gate : Optional[qt.Qobj]
            The resulting pulsed gate operation.
        target_gate : Optional[qt.Qobj | np.ndarray]
            The target gate operation.
        final_only : bool
            Only used when ``gate`` is None: if True, compute the fidelity of the final propagator only (see
            ``time_evolution_operator``).

        Returns
        -------
        gate_fid : List[float]
            A list of average gate fidelities for each time step in the pulse duration (a single value when
            ``final_only`` is True).
        """

        if isinstance(target_gate, np.ndarray):
            target_gate = qt.Qobj(target_gate)

        if gate is None:
            operators = self.time_evolution_operator(final_only=final_only)
        elif isinstance(gate, qt.Qobj):
            operators = [gate]
        elif isinstance(gate, list) and all(isinstance(g, qt.Qobj) for g in gate):
            operators = gate
        else:
            raise ValidationError("Gate must be a Qobj or a list of Qobj instances.")

        gate_fid = [qt.average_gate_fidelity(oper=oper, target=target_gate) for oper in operators]

        return gate_fid

    def _resolve_initial_state(self, initial_state: Optional[np.ndarray | int | qt.Qobj]) -> qt.Qobj:
        """Initial state of the evolution, following the conventions of ``state_fidelity``."""
        control_initial = self._boundary_controls[0]

        if initial_state is None:
            if self._initial_state is None:
                raise ConfigurationError(
                    "The initial state index must be configured in the ControlModel when no explicit initial state is "
                    "provided."
                )
            return self._eigenstate(control_initial, self._initial_state)

        if _is_index(initial_state):
            dimension = self._hamiltonian_dimension
            if dimension is None:
                raise ConfigurationError("Hamiltonian dimension is unavailable.")
            index = validate_state_index(cast(int, initial_state), dimension, "initial_state")
            return self._eigenstate(control_initial, index)

        if isinstance(initial_state, np.ndarray):
            if initial_state.shape[0] != self._hamiltonian_dimension:
                raise ValidationError(
                    f"Initial state must have the same dimension as the ControlModel. Shape of ControlModel:"
                    f" {(self._hamiltonian_dimension, self._hamiltonian_dimension)}."
                    f" Shape of initial state: {initial_state.shape}"
                )
            return qt.Qobj(initial_state)

        if isinstance(initial_state, qt.Qobj):
            return initial_state

        raise ValidationError(
            "Initial state must be either an integer, a numpy array with correct dimensions or a Qobj instance."
        )

    def populations(
        self,
        initial_state: Optional[np.ndarray | int | qt.Qobj] = None,
        basis: Literal["adiabatic", "diabatic"] = "adiabatic",
        c_ops: Optional[List[qt.Qobj] | List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Compute the population of every state at each time sample of the pulsed time evolution.

        Parameters
        ----------
        initial_state : Optional[np.ndarray, int or qt.Qobj]
            Initial state of the evolution, with the same conventions as in ``state_fidelity``: None uses the initial
            state index of the ControlModel, an integer selects an eigenstate of the Hamiltonian at the initial control
            value, and a numpy array or Qobj is used as given (state vector or density matrix).
        basis : {"adiabatic", "diabatic"}
            Basis in which populations are measured. ``"adiabatic"`` uses the instantaneous eigenstates of the
            Hamiltonian at each time, sorted by increasing energy. ``"diabatic"`` uses the fixed basis in which the
            Hamiltonian matrix is written. Adiabatic populations of exactly degenerate levels are not unique.
        c_ops : Optional[list]
            Collapse operators (passed as a list of Qobj or np.ndarray) for the Lindblad master equation.

        Returns
        -------
        populations : np.ndarray
            Array of shape ``(len(times), dimension)``, where ``populations[i, n]`` is the population of state ``n`` at
            ``times[i]``.
        """
        if basis not in ("adiabatic", "diabatic"):
            raise ValidationError(f"basis must be 'adiabatic' or 'diabatic', got {basis!r}.")

        c_ops = _as_collapse_operators(c_ops)
        psi_init = self._resolve_initial_state(initial_state)

        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()
        options = {"store_states": True}

        with self._blas_threads():
            result = qt.mesolve(self._qevo, psi_init, cast(Any, pulse_times), c_ops=cast(Any, c_ops), options=options)

            states = result.states
            if len(states) != len(pulse_times):
                raise ValidationError("Time evolution did not return a state for every time sample.")

            populations = np.empty((len(states), states[0].shape[0]))
            for i, (state, control_value) in enumerate(zip(states, self._pulse, strict=True)):
                data = state.full()
                if basis == "adiabatic":
                    _, eigenvectors = np.linalg.eigh(np.asarray(self.evaluate_hamiltonian(control_value)))
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
        initial_state: Optional[np.ndarray | int | qt.Qobj] = None,
        basis: Literal["adiabatic", "diabatic"] = "adiabatic",
        c_ops: Optional[List[qt.Qobj] | List[np.ndarray]] = None,
        states: Optional[Sequence[int]] = None,
        legend: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        **plot_kwargs: Any,
    ) -> "tuple[Figure | SubFigure, Axes]":
        """
        Plot the population of each state as a function of time during the pulsed time evolution.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created.
        initial_state : Optional[np.ndarray, int or qt.Qobj]
            Initial state of the evolution (see ``populations``).
        basis : {"adiabatic", "diabatic"}
            Basis in which populations are measured (see ``populations``).
        c_ops : Optional[list]
            Collapse operators for the Lindblad master equation (see ``populations``).
        states : Optional[Sequence[int]]
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

        if ax is None:
            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        fig = ax.figure

        for index in indices:
            label = rf"$|E_{{{index}}}\rangle$" if basis == "adiabatic" else rf"$|{index}\rangle$"
            ax.plot(self._pulse_times, populations[:, index], label=label, **plot_kwargs)

        ax.set_xlabel("Time $t$" if xlabel is None else xlabel)
        ax.set_ylabel("Population" if ylabel is None else ylabel)
        ax.set_title(f"{basis.capitalize()} populations" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax


def _is_index(value: Any) -> bool:
    """Whether a value is an integer state index (Python or NumPy integer, but not a bool)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _as_collapse_operators(c_ops: Any) -> Optional[List[qt.Qobj]]:
    """Validate collapse operators, converting numpy arrays to Qobj."""
    if c_ops is None:
        return None
    if isinstance(c_ops, list):
        return [qt.Qobj(op) if isinstance(op, np.ndarray) else op for op in c_ops]
    raise ValidationError("Collapse operators must be provided as a list of Qobj or numpy arrays.")
