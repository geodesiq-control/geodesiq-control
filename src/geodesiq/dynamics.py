from typing import List, Optional

import numpy as np
import qutip as qt

from ._utils import validate_state_index
from .controlmodel import ControlModel
from .exceptions import ValidationError, ConfigurationError


class Dynamics:
    def __init__(self, duration: float, model: ControlModel, hbar: float = 1.0):
        """
        Initialize the Dynamics object, which depends on an instance of the ControlModel class. This class deals with
        observables due to the time evolution of the pulsed ControlModel.

        Parameters:
        -----------
        duration: float
            Duration of the control pulse (t_f).
        model: ControlModel
            An instance of the ControlModel class containing the control pulse and system parameters.
        hbar: float
            Reduced Planck's constant (default is 1).
        """

        # Attributes of the ControlModel instance
        self.evaluate_hamiltonian = model.evaluate_hamiltonian
        self._control_pulse: np.ndarray | None = (
            np.asarray(model.control_pulse) if model.control_pulse is not None else None)
        self._control_sol: np.ndarray | None = np.asarray(model.control_sol) if model.control_sol is not None else None
        self._initial_state: int | None = model.initial_state
        self._final_state: int | None = model.final_state
        self._hamiltonian_dimension: int | None = model.hamiltonian_dimension

        if not isinstance(hbar, (int, float, np.integer, np.floating)) or isinstance(hbar, bool):
            raise ValidationError("hbar must be a finite positive number.")
        try:
            hbar = float(hbar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("hbar must be a finite positive number.") from exc
        if not np.isfinite(hbar) or hbar <= 0:
            raise ValidationError("hbar must be a finite positive number.")
        self._hbar: float = hbar

        self._control_pulse = np.asarray(self._control_pulse, dtype=float)
        self._control_sol = np.asarray(self._control_sol, dtype=float)

        if not isinstance(duration, (int, float, np.integer, np.floating)) or isinstance(duration, bool):
            raise ValidationError("duration must be a finite positive number.")

        duration = float(duration)

        if not np.isfinite(duration) or duration <= 0:
            raise ValidationError("duration must be a finite positive number.")
        self._duration: float = duration
        self._pulse_times: np.ndarray = duration * np.linspace(0.0, 1.0, len(self._control_sol))

    def _eigenstate(self, control_value: float, state_index: int) -> qt.Qobj:
        hamiltonian = qt.Qobj(self.evaluate_hamiltonian(control_value))
        _, eigenstates = hamiltonian.eigenstates()
        return eigenstates[state_index]

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the time-dependent ControlModel using QuTiP Qobj
        """
        pulse_times: np.ndarray = np.asarray(self._pulse_times, dtype=float).tolist()
        control_sol: np.ndarray = np.asarray(self._control_sol, dtype=float).tolist()
        control_val_t = float(np.interp(t, pulse_times, control_sol))

        return qt.Qobj(self.evaluate_hamiltonian(control_val_t)) / self._hbar

    def time_evolution_operator(self) -> List[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse ControlModel.
        """
        pulse_times: np.ndarray = np.asarray(self._pulse_times, dtype=float).tolist()
        propagator = qt.propagator(self._get_ham, pulse_times)
        if isinstance(propagator, list):
            return propagator
        return [propagator]

    def state_fidelity(self, initial_state: Optional[np.ndarray | int | qt.Qobj] = None,
                       final_state: Optional[np.ndarray | int | qt.Qobj] = None,
                       c_ops: Optional[List[qt.Qobj] | List[np.ndarray]] = None, ) -> float:
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

        control_pulse = self._control_pulse

        if control_pulse is None:
            raise ConfigurationError(
                "Control pulse is unavailable. Solve the ControlModel before computing the dynamics.")

        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()

        if initial_state is None and final_state is None:
            initial_index = self._initial_state
            final_index = self._final_state

            if initial_index is None or final_index is None:
                raise ConfigurationError("Initial and final state indices must be configured in the ControlModel "
                                         "when no explicit states are provided.")

            psi_init = self._eigenstate(float(control_pulse[0]), initial_index, )
            psi_target = self._eigenstate(float(control_pulse[-1]), final_index, )

        elif isinstance(initial_state, int) and isinstance(final_state, int):
            dimension = self._hamiltonian_dimension

            if dimension is None:
                raise ConfigurationError("Hamiltonian dimension is unavailable.")

            validate_state_index(initial_state, dimension, "initial_state")
            validate_state_index(final_state, dimension, "final_state")

            psi_init = self._eigenstate(float(control_pulse[0]), initial_state)
            psi_target = self._eigenstate(float(control_pulse[-1]), final_state)

        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            if (initial_state.shape[0] != self._hamiltonian_dimension or final_state.shape[
                0] != self._hamiltonian_dimension):
                raise ValidationError(
                    f"Initial and final states must have the same dimension as the ControlModel. Shape of ControlModel:"
                    f" {(self._hamiltonian_dimension, self._hamiltonian_dimension)}."
                    f" Shape of initial state: {initial_state.shape}")

            psi_init = qt.Qobj(initial_state)
            psi_target = qt.Qobj(final_state)
        elif isinstance(initial_state, qt.Qobj) and isinstance(final_state, qt.Qobj):
            psi_init = initial_state
            psi_target = final_state
        else:
            raise ValidationError(
                "Initial and final states must be either integers, numpy arrays with correct dimensions or Qobj "
                "instances.")

        options = {"store_final_state": True, "store_states": False}

        result = qt.mesolve(self._get_ham, psi_init, pulse_times, c_ops=c_ops, options=options)
        psi_f = result.final_state
        if psi_f is None:
            raise ValidationError("Time evolution did not return a final state.")

        state_fidelity: float = qt.fidelity(psi_target, psi_f) ** 2

        return state_fidelity

    def average_gate_fidelity(self, gate: Optional[qt.Qobj | List[qt.Qobj]] = None,
                              target_gate: Optional[qt.Qobj | np.ndarray] = None) -> List[float]:
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
