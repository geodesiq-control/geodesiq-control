from typing import Any, List, Optional, Tuple, cast

import numpy as np
import qutip as qt

from ._utils import validate_state_index
from .controlmodel import ControlModel
from .decompose_hamiltonian import decompose_hamiltonian
from .exceptions import ConfigurationError, MissingArgsError, ValidationError


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

        Parameters:
        -----------
        model: ControlModel
            An instance of the ControlModel class containing the Hamiltonian and the system parameters.
        duration: Optional[float]
            Duration of the optimal control pulse (t_f). Must not be combined with ``times`` and ``pulse``.
        times: Optional[np.ndarray]
            Strictly increasing physical times at which ``pulse`` is sampled. Must be given together with ``pulse``.
        pulse: Optional[np.ndarray]
            Control pulse values sampled at ``times``. Must be given together with ``times``.
        hbar: float
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
            self._qevo = decompose_hamiltonian(self._get_ham, self._pulse_times, drift="mean", rtol=1e-10).qobjevo()

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

    def _eigenstate(self, control_value: float, state_index: int) -> qt.Qobj:
        hamiltonian = qt.Qobj(self.evaluate_hamiltonian(control_value))
        _, eigenstates = hamiltonian.eigenstates()
        return eigenstates[state_index]

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the time-dependent ControlModel using QuTiP Qobj
        """
        control_val_t = float(np.interp(t, self._pulse_times, self._pulse))

        return qt.Qobj(self.evaluate_hamiltonian(control_val_t)) / self._hbar

    def time_evolution_operator(self) -> List[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse ControlModel.
        """
        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()
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

        Parameters:
        -----------
        initial_state: Optional[np.ndarray, int or qt.Qobj]
            Initial state for pulsed time evolution.
        final_state: Optional[np.ndarray, int or qt.Qobj]
            Final state for pulsed time evolution.
        c_ops: Optional[list]
            Collapse operators (passed as a list of Qobj or np.ndarray) for the Lindblad master equation.

        """
        if c_ops is None:
            c_ops = None
        elif isinstance(c_ops, list):
            c_ops = [qt.Qobj(op) if isinstance(op, np.ndarray) else op for op in c_ops]
        else:
            raise ValidationError("Collapse operators must be provided as a list of Qobj or numpy arrays.")

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

        elif isinstance(initial_state, int) and isinstance(final_state, int):
            dimension = self._hamiltonian_dimension

            if dimension is None:
                raise ConfigurationError("Hamiltonian dimension is unavailable.")

            validate_state_index(initial_state, dimension, "initial_state")
            validate_state_index(final_state, dimension, "final_state")

            psi_init = self._eigenstate(control_initial, initial_state)
            psi_target = self._eigenstate(control_final, final_state)

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

        result = qt.mesolve(self._qevo, psi_init, cast(Any, pulse_times), c_ops=cast(Any, c_ops), options=options)


        psi_f = result.final_state
        if psi_f is None:
            raise ValidationError("Time evolution did not return a final state.")

        state_fidelity: float = qt.fidelity(psi_target, psi_f) ** 2

        return state_fidelity

    def average_gate_fidelity(
        self, gate: Optional[qt.Qobj | List[qt.Qobj]] = None, target_gate: Optional[qt.Qobj | np.ndarray] = None
    ) -> List[float]:
        """
        Compute average gate fidelity given the pulsed time evolution operator in the explicit real-time duration given.

        Parameters:
        -----------
        gate: Optional[qt.Qobj]
            The resulting pulsed gate operation.
        target_gate: Optional[qt.Qobj | np.ndarray]
            The target gate operation.


        Returns:
        --------
        gate_fid: List[float]
            A list of average gate fidelities for each time step in the pulse duration.
        """

        if isinstance(target_gate, np.ndarray):
            target_gate = qt.Qobj(target_gate)

        if gate is None:
            operators = self.time_evolution_operator()
        elif isinstance(gate, qt.Qobj):
            operators = [gate]
        elif isinstance(gate, list) and all(isinstance(g, qt.Qobj) for g in gate):
            operators = gate
        else:
            raise ValidationError("Gate must be a Qobj or a list of Qobj instances.")

        gate_fid = [qt.average_gate_fidelity(oper=oper, target=target_gate) for oper in operators]

        return gate_fid
