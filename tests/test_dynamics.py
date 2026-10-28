from typing import Any, cast

import numpy as np
import pytest
import qutip as qt
from qutip.solver.integrator import IntegratorException

import geodesiq.dynamics as dynamics_module
from geodesiq import ControlModel
from geodesiq.dynamics import Dynamics
from geodesiq.exceptions import (
    ConfigurationError,
    MissingArgsError,
    MissingControlParameterError,
    SolverError,
    ValidationError,
)
from geodesiq.warnings import GeodesiQWarning, NumericalStabilityWarning

# The fixture models sweep lam from 1 to 3 with final_state=1, so their diabatic passage lies at the start of the grid
# (no anticrossing inside the sweep), which needs allow_endpoint_passages=True. They only need a solved model.

# ------------------------------------------------------------
# Real ControlModel() fixtures
# ------------------------------------------------------------


def lz_hamiltonian(lam: float, delta: float = 1.0) -> np.ndarray:
    """Simple 2-level Landau-Zener Hamiltonian."""
    return np.array([[lam, delta], [delta, -lam]], dtype=float)


def _build_solved_model(pulse_accuracy: int = 3) -> ControlModel:
    model = ControlModel(lz_hamiltonian)
    model.set_parameters(delta=1.0)
    model.set_control(control_name="lam", pulse_initial=1.0, pulse_final=3.0, initial_state=0, final_state=1, alpha=2.0,
                      beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33, allow_endpoint_passages=True, )
    model.solve_problem(pulse_accuracy=pulse_accuracy)
    return model


def _build_solved_affine_model(pulse_accuracy: int = 5) -> ControlModel:
    H_d = np.array([[0.0, 1.0], [1.0, 0.0]])
    H_c = np.array([[1.0, 0.0], [0.0, -1.0]])
    model = ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(pulse_initial=1.0, pulse_final=3.0, initial_state=0, final_state=1, alpha=2.0,
                      beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33, allow_endpoint_passages=True, )
    model.solve_problem(pulse_accuracy=pulse_accuracy)
    return model


@pytest.fixture
def real_model():
    return _build_solved_model()


@pytest.fixture
def affine_model():
    return _build_solved_affine_model()


@pytest.fixture
def default_dynamics(real_model):
    duration = 2.0
    return Dynamics(duration=duration, model=real_model)


@pytest.fixture
def varying_dynamics():
    duration = 2.0
    varying_model = _build_solved_model()
    # Keep interpolation test deterministic independent of ODE solver details.
    varying_model._control_sol = np.array([0.0, 2.0, 4.0], dtype=float)
    return Dynamics(duration=duration, model=varying_model)


@pytest.fixture
def affine_dynamics(affine_model):
    return Dynamics(duration=2.0, model=affine_model)


# ------------------------------------------------------------
# Testing initialization and internal method _get_ham()
# ------------------------------------------------------------


class TestSolverOptions:
    @pytest.mark.parametrize("options", [[], "tight", 1, {1: 1e-8}])
    def test_invalid_options_container(self, affine_model, options):
        with pytest.raises(ValidationError, match="dictionary with string keys"):
            Dynamics(affine_model, duration=2, options=options)

    @pytest.mark.parametrize("fixture_name", ["real_model", "affine_model"])
    @pytest.mark.parametrize("operation", ["full", "final", "gate", "state", "open_state", "populations",
                                          "open_populations"])
    def test_options_reach_every_solver_and_storage_is_preserved(self, request, monkeypatch, fixture_name, operation):
        options = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 5000, "max_step": 0.05, "method": "dop853",
                   "store_states": False, "store_final_state": False, "progress_kwargs": {"chunk_size": 7}}
        dynamics = Dynamics(request.getfixturevalue(fixture_name), duration=2, options=options)
        calls = []
        solver_name = "propagator" if operation == "full" else "sesolve" if operation == "final" else "mesolve"
        if operation == "gate":
            solver_name = "sesolve"
        original = getattr(qt, solver_name)

        def tracked_solver(*args, **kwargs):
            calls.append(kwargs["options"].copy())
            kwargs["options"]["progress_kwargs"]["chunk_size"] = 8
            return original(*args, **kwargs)

        monkeypatch.setattr(qt, solver_name, tracked_solver)
        # Caller mutation, including a nested dictionary, must not change the stored configuration.
        options["atol"] = 0.1
        options["progress_kwargs"]["chunk_size"] = 99
        for _ in range(2):
            if operation in ("full", "final"):
                assert len(dynamics.time_evolution_operator(final_only=operation == "final")) > 0
            elif operation == "gate":
                assert len(dynamics.average_gate_fidelity(target_gate=qt.sigmax(), final_only=True)) == 1
            elif "state" in operation:
                dynamics.state_fidelity(c_ops=[0.1 * qt.sigmaz()] if operation == "open_state" else None)
            else:
                c_ops = [0.1 * qt.sigmaz()] if operation == "open_populations" else None
                assert dynamics.populations(c_ops=c_ops).shape[1] == 2
        assert len(calls) == 2
        # Runs that output only the end of the pulse scale nsteps by the number of sample intervals (D-19).
        final_state_run = operation in ("final", "gate", "state", "open_state")
        expected_nsteps = 5000 * (len(dynamics.times) - 1) if final_state_run else 5000
        for received in calls:
            assert received["atol"] == 1e-10
            assert received["rtol"] == 1e-8
            assert received["nsteps"] == expected_nsteps
            assert received["max_step"] == 0.05
            assert received["method"] == "dop853"
            assert received["store_states"] is (operation in ("full", "populations", "open_populations"))
            if operation in ("final", "gate", "state", "open_state"):
                assert received["store_final_state"] is True
        assert dynamics._options["progress_kwargs"]["chunk_size"] == 7

    @pytest.mark.parametrize("operation", ["full", "final", "gate", "state", "populations"])
    def test_integration_failure_is_chained(self, affine_model, monkeypatch, operation):
        failure = IntegratorException("Excess work done")

        def failed_solver(*args, **kwargs):
            raise failure

        for name in ("propagator", "sesolve", "mesolve"):
            monkeypatch.setattr(qt, name, failed_solver)
        dynamics = Dynamics(affine_model, duration=2)
        with pytest.raises(SolverError, match="nsteps") as error:
            if operation in ("full", "final"):
                dynamics.time_evolution_operator(final_only=operation == "final")
            elif operation == "gate":
                dynamics.average_gate_fidelity(target_gate=qt.sigmax())
            elif operation == "state":
                dynamics.state_fidelity()
            else:
                dynamics.populations()
        assert error.value.__cause__ is failure

    @pytest.mark.parametrize("options", [{"rtlo": 1e-8}, {"method": "unknown_method"}])
    def test_unsupported_options_raise_package_error(self, affine_model, options):
        dynamics = Dynamics(affine_model, duration=2, options=options)
        with pytest.raises(ValidationError, match="QuTiP solver rejected"):
            dynamics.time_evolution_operator()

    def test_long_evolution_converges_to_exact_unitary(self):
        model = ControlModel(H_d=qt.sigmax().full(), H_c=np.zeros((2, 2)))
        times = np.linspace(0, 500, 101)
        pulse = np.zeros_like(times)
        default = Dynamics(model, times=times, pulse=pulse).time_evolution_operator(final_only=True)[0]
        tight = Dynamics(model, times=times, pulse=pulse,
                         options={"atol": 1e-12, "rtol": 1e-12, "nsteps": 10000, "normalize_output": False})
        final = tight.time_evolution_operator(final_only=True)[0]
        exact = (-500j * qt.sigmax()).expm().full()
        loose_error = np.linalg.norm(default.full() - exact)
        tight_error = np.linalg.norm(final.full() - exact)
        assert tight_error < loose_error / 100
        assert tight_error < 1e-6
        np.testing.assert_allclose(final.dag().full() @ final.full(), np.eye(2), atol=1e-7)
        populations = tight.populations(initial_state=qt.basis(2, 0), basis="diabatic")
        np.testing.assert_allclose(populations.sum(axis=1), 1, atol=1e-7)

    @pytest.mark.filterwarnings("ignore:.*Excess work done.*")
    def test_nsteps_can_recover_a_real_integration_failure(self):
        model = ControlModel(H_d=100 * qt.sigmax().full(), H_c=np.zeros((2, 2)))
        times = [0, 1]
        pulse = [0, 0]
        with pytest.raises(SolverError) as error:
            Dynamics(model, times=times, pulse=pulse, options={"nsteps": 1}).time_evolution_operator(final_only=True)
        assert isinstance(error.value.__cause__, IntegratorException)
        final = Dynamics(model, times=times, pulse=pulse, options={"nsteps": 10000, "atol": 1e-11, "rtol": 1e-11})
        np.testing.assert_allclose(final.time_evolution_operator(final_only=True)[0].full(),
                                   (-100j * qt.sigmax()).expm().full(), atol=1e-7)

    def test_max_step_resolves_a_narrow_hold_pulse(self):
        model = ControlModel(H_d=np.zeros((2, 2)), H_c=qt.sigmax().full())
        dynamics = Dynamics(model, times=[0, 0.49, 0.5, 0.51, 1], pulse=[0, 0, 50, 0, 0], order=0,
                            options={"max_step": 0.002, "atol": 1e-11, "rtol": 1e-11, "nsteps": 10000})
        final = dynamics.time_evolution_operator(final_only=True)[0]
        np.testing.assert_allclose(final.full(), (-0.5j * qt.sigmax()).expm().full(), atol=1e-7)


def test_initialization(real_model):
    """Verify attributes are correctly extracted from the ControlModel object."""
    duration = 5.0
    dyn = Dynamics(duration=duration, model=real_model)

    assert dyn._duration == 5.0
    assert len(dyn._pulse_times) == len(real_model.control_sol)
    assert dyn._pulse_times[-1] == 5.0  # Check proper scaling of time array


def test_get_ham(default_dynamics):
    """Test the internal QuTiP ControlModel constructor method."""
    H_qobj = default_dynamics._get_ham(t=1.0)

    assert isinstance(H_qobj, qt.Qobj)


def test_get_ham_interpolation(varying_dynamics):
    """ControlModel values should follow linear interpolation of the solved control pulse."""
    H_qobj = varying_dynamics._get_ham(t=0.5)
    expected = np.array([[1.0, 1.0], [1.0, -1.0]])

    np.testing.assert_allclose(H_qobj.full(), expected)


def test_affine_initialization_bypasses_general_decomposition(affine_model, monkeypatch):
    """Affine models should be converted directly into a QobjEvo."""

    def unexpected_decomposition(*args, **kwargs):
        raise AssertionError("decompose_hamiltonian should not be called for an affine model.")

    monkeypatch.setattr(dynamics_module, "decompose_hamiltonian", unexpected_decomposition)

    dynamics = Dynamics(duration=2.0, model=affine_model)

    assert isinstance(dynamics._qevo, qt.QobjEvo)


def test_callable_initialization_keeps_general_decomposition(real_model, monkeypatch):
    """Callable Hamiltonians should retain the existing decomposition route."""
    original_decomposition = dynamics_module.decompose_hamiltonian
    calls = []

    def tracked_decomposition(*args, **kwargs):
        calls.append((args, kwargs))
        return original_decomposition(*args, **kwargs)

    monkeypatch.setattr(dynamics_module, "decompose_hamiltonian", tracked_decomposition)

    dynamics = Dynamics(duration=2.0, model=real_model)

    assert len(calls) == 1
    assert isinstance(dynamics._qevo, qt.QobjEvo)


def test_affine_dynamics_propagator_starts_as_identity(affine_dynamics):
    propagators = affine_dynamics.time_evolution_operator()

    assert len(propagators) == len(affine_dynamics._pulse_times)
    np.testing.assert_allclose(propagators[0].full(), np.eye(2))


def test_affine_initialization_rejects_missing_internal_matrix(affine_model):
    affine_model._H_c = None

    with pytest.raises(ConfigurationError, match="missing its constant drift or control Hamiltonian"):
        Dynamics(duration=2.0, model=affine_model)


def test_unconfigured_model_raises_missing_control_parameters():
    # D-18: control_pulse/control_sol never return None; the lazy solve reports the missing settings itself.
    model = ControlModel(H_d=np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.array([[1.0, 0.0], [0.0, -1.0]]))

    with pytest.raises(MissingControlParameterError, match="pulse_initial, pulse_final"):
        Dynamics(duration=2.0, model=model)

    model.set_control(control_name="z", pulse_initial=-1.0, pulse_final=1.0, initial_state=0)
    with pytest.raises(MissingControlParameterError, match="alpha, beta"):
        Dynamics(duration=2.0, model=model)


@pytest.mark.parametrize("duration", [0, -1, -1.5, np.nan, np.inf, -np.inf, True, False, "1.0", 1 + 2j, ], )
def test_invalid_duration_raises_validation_error(real_model, duration: Any) -> None:
    with pytest.raises(ValidationError, match="duration must be a finite positive number", ):
        Dynamics(duration=duration, model=real_model)


@pytest.mark.parametrize("duration", [1, 1.5, np.int64(2), np.float64(2.5), ], )
def test_valid_duration_is_accepted(real_model, duration: Any) -> None:
    dynamics = Dynamics(duration=duration, model=real_model)

    assert dynamics._duration == float(duration)


# ------------------------------------------------------------
# Testing pulse sources: optimal pulse (duration) vs. custom pulse (times, pulse)
# ------------------------------------------------------------


class TestOptimalPulse:
    """Only a duration is given: the solved optimal pulse is rescaled to physical time."""

    def test_model_can_be_passed_positionally(self, real_model):
        dynamics = Dynamics(real_model, 3.0)

        assert dynamics.duration == 3.0

    def test_uses_solved_pulse_on_uniform_grid(self, real_model):
        dynamics = Dynamics(real_model, duration=4.0)

        np.testing.assert_allclose(dynamics.pulse, real_model.control_sol)
        np.testing.assert_allclose(dynamics.times, np.linspace(0.0, 4.0, len(real_model.control_sol)))

    def test_eigenstate_boundaries_follow_control_grid(self, real_model):
        dynamics = Dynamics(real_model, duration=2.0)

        assert dynamics._boundary_controls == (real_model.pulse_initial, real_model.pulse_final)

    def test_hamiltonian_evaluation_is_private(self, real_model):
        # The public, unannotated evaluate_hamiltonian lambda was replaced by a private method (P-06, D-17).
        dynamics = Dynamics(real_model, duration=2.0)

        assert not hasattr(dynamics, "evaluate_hamiltonian")
        np.testing.assert_allclose(dynamics._evaluate_hamiltonian(1.5), real_model.evaluate_hamiltonian(1.5))

    def test_later_model_changes_do_not_reach_an_existing_dynamics(self):
        # The propagation was frozen at construction, but the endpoint eigenstates and the adiabatic basis used the
        # live model, so a changed parameter mixed two Hamiltonians without an error (D-03).
        model = ControlModel(lz_hamiltonian)
        model.set_parameters(delta=1.0)
        model.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=129)
        model.solve_problem(pulse_accuracy=50)
        dynamics = Dynamics(model, duration=10.0)
        fidelity = dynamics.state_fidelity()
        populations = dynamics.populations()

        model.set_parameters(delta=0.2)
        model.set_control(control_name="lam", initial_state=1)

        assert dynamics.state_fidelity() == fidelity
        np.testing.assert_array_equal(dynamics.populations(), populations)
        np.testing.assert_allclose(dynamics._evaluate_hamiltonian(0.0), lz_hamiltonian(0.0, delta=1.0))
        model.solve_problem(pulse_accuracy=50)
        assert Dynamics(model, duration=10.0).state_fidelity() != fidelity

    def test_parameters_are_copied_into_the_snapshot(self):
        # A mutable parameter changed in place after construction must not reach the snapshot either.
        def hamiltonian(lam: float, couplings: list) -> np.ndarray:
            return np.array([[lam, couplings[0]], [couplings[0], -lam]], dtype=float)

        couplings = [1.0]
        model = ControlModel(hamiltonian)
        model.set_parameters(couplings=couplings)
        model.set_control(control_name="lam", pulse_initial=-2.0, pulse_final=2.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=33)
        model.solve_problem(pulse_accuracy=20)
        dynamics = Dynamics(model, duration=5.0)

        model.set_parameters(couplings=[0.3])

        np.testing.assert_allclose(dynamics._evaluate_hamiltonian(0.0), [[0.0, 1.0], [1.0, 0.0]])

    def test_matches_synthesized_pulse_control(self, real_model):
        duration = 2.5
        dynamics = Dynamics(real_model, duration=duration)
        pulse_control = real_model.synthesize_pulse(duration=duration)

        np.testing.assert_allclose(dynamics.times, pulse_control.times)
        np.testing.assert_allclose(dynamics.pulse, pulse_control.pulse)

    def test_fidelity_increases_with_duration(self):
        model = _build_solved_model(pulse_accuracy=200)

        short = Dynamics(model, duration=0.5).state_fidelity(initial_state=0, final_state=0)
        long = Dynamics(model, duration=50.0).state_fidelity(initial_state=0, final_state=0)

        assert long > short
        assert long == pytest.approx(1.0, abs=1e-3)

    def test_properties_return_copies(self, default_dynamics):
        default_dynamics.times[0] = 100.0
        default_dynamics.pulse[0] = 100.0

        assert default_dynamics.times[0] == 0.0
        assert default_dynamics.pulse[0] != 100.0

    def test_missing_duration_and_pulse_raises(self, real_model):
        with pytest.raises(MissingArgsError, match="Provide either duration"):
            Dynamics(real_model)

    def test_invalid_model_raises(self):
        with pytest.raises(ValidationError, match="model must be an instance of ControlModel"):
            Dynamics(cast(Any, 2.0), duration=1.0)

    def test_unsolved_model_is_solved_lazily(self):
        model = ControlModel(lz_hamiltonian)
        model.set_parameters(delta=1.0)
        model.set_control(control_name="lam", pulse_initial=1.0, pulse_final=3.0, initial_state=0, final_state=1,
                          alpha=2.0, beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33,
                          allow_endpoint_passages=True, )

        dynamics = Dynamics(model, duration=2.0)

        np.testing.assert_allclose(dynamics.pulse, model.control_sol)


class TestCustomPulse:
    """Explicit times and pulse are given, e.g. after filtering the pulse with PulseControl."""

    def test_reproduces_optimal_pulse_dynamics(self, real_model):
        """Feeding the optimal pulse explicitly must give the same result as passing only the duration."""
        optimal = Dynamics(real_model, duration=2.0)
        # A custom pulse is interpolated linearly by default; use the cubic spline of the optimal pulse.
        custom = Dynamics(real_model, times=optimal.times, pulse=optimal.pulse, order=3)

        assert custom.state_fidelity() == pytest.approx(optimal.state_fidelity(), rel=1e-8)
        np.testing.assert_allclose(
            custom.time_evolution_operator()[-1].full(), optimal.time_evolution_operator()[-1].full(), atol=1e-8
        )

    def test_reproduces_optimal_pulse_dynamics_affine(self, affine_model):
        optimal = Dynamics(affine_model, duration=2.0)
        custom = Dynamics(affine_model, times=optimal.times, pulse=optimal.pulse, order=3)

        assert custom.state_fidelity() == pytest.approx(optimal.state_fidelity(), rel=1e-8)

    # The low cutoffs move the steep endpoints of these pulses, which filtered_pulse() reports (D-05).
    @pytest.mark.filterwarnings("ignore:.*The filtered pulse:geodesiq.warnings.GeodesiQWarning")
    @pytest.mark.filterwarnings("ignore:.*boundary controls:geodesiq.warnings.GeodesiQWarning")
    def test_filtered_pulse_from_pulse_control(self):
        model = _build_solved_model(pulse_accuracy=200)
        duration = 50.0
        pulse_control = model.synthesize_pulse(duration=duration)
        times, filtered = pulse_control.filtered_pulse(cutoff_freq=0.1)

        dynamics = Dynamics(model, times=times, pulse=filtered)

        np.testing.assert_allclose(dynamics.times, times)
        np.testing.assert_allclose(dynamics.pulse, filtered)
        assert dynamics.duration == pytest.approx(duration)
        assert not np.allclose(filtered, pulse_control.pulse)
        fidelity = dynamics.state_fidelity(initial_state=0, final_state=0)
        # A smooth, slow pulse should still drive an (almost) adiabatic evolution.
        optimal = Dynamics(model, duration=duration).state_fidelity(initial_state=0, final_state=0)
        assert fidelity == pytest.approx(optimal, abs=1e-2)

    @pytest.mark.filterwarnings("ignore:.*The filtered pulse:geodesiq.warnings.GeodesiQWarning")
    @pytest.mark.filterwarnings("ignore:.*boundary controls:geodesiq.warnings.GeodesiQWarning")
    def test_filtered_pulse_from_pulse_control_affine(self):
        model = _build_solved_affine_model(pulse_accuracy=200)
        pulse_control = model.synthesize_pulse(duration=20.0)
        times, filtered = pulse_control.filtered_pulse(cutoff_freq=0.5)

        dynamics = Dynamics(model, times=times, pulse=filtered)
        gate_fid = dynamics.average_gate_fidelity(target_gate=qt.identity(2))

        assert len(gate_fid) == len(times)
        assert gate_fid[0] == pytest.approx(1.0)

    def test_discretized_pulse_uses_its_own_grid(self, real_model):
        times, values = real_model.synthesize_pulse(duration=10.0).discretized_pulse(linear_steps=7)

        dynamics = Dynamics(real_model, times=times, pulse=values)

        assert len(dynamics.time_evolution_operator()) == 7
        np.testing.assert_allclose(dynamics.times, times)

    def test_non_uniform_and_shifted_time_grid(self, affine_model):
        times = 1.0 + 4.0 * np.linspace(0.0, 1.0, 40) ** 2
        pulse = np.linspace(1.0, 3.0, times.size)

        dynamics = Dynamics(affine_model, times=times, pulse=pulse)
        propagators = dynamics.time_evolution_operator()

        assert dynamics.duration == pytest.approx(4.0)
        np.testing.assert_allclose(propagators[0].full(), np.eye(2), atol=1e-12)
        np.testing.assert_allclose(dynamics._get_ham(times[-1]).full(), lz_hamiltonian(3.0), atol=1e-12)

    def test_linear_ramp_matches_analytical_hamiltonian(self, real_model):
        times = np.linspace(0.0, 2.0, 21)
        pulse = np.linspace(1.0, 3.0, times.size)

        dynamics = Dynamics(real_model, times=times, pulse=pulse)

        np.testing.assert_allclose(dynamics._get_ham(1.0).full(), lz_hamiltonian(2.0))

    def test_eigenstate_boundaries_follow_model_control_problem(self, real_model):
        """Initial and target eigenstates are those of the model, not of the (filtered) pulse endpoints."""
        times = np.linspace(0.0, 1.0, 11)
        pulse = np.linspace(1.2, 2.8, times.size)

        message = r"runs from 1\.2 to 2\.8, but the model's boundary controls are 1 and 3"
        with pytest.warns(GeodesiQWarning, match=message):
            dynamics = Dynamics(real_model, times=times, pulse=pulse)

        assert dynamics._boundary_controls == (1.0, 3.0)

    def test_pulse_ending_at_the_boundary_controls_does_not_warn(self, real_model):
        times = np.linspace(0.0, 1.0, 11)
        # Within 1% of the control range of the model (lam from 1 to 3).
        pulse = np.linspace(1.0 + 0.019, 3.0 - 0.019, times.size)

        Dynamics(real_model, times=times, pulse=pulse)  # GeodesiQWarning is an error in the tests

    def test_degenerate_endpoint_eigenstate_warns(self):
        # Levels 1 and 2 of diag(-1, 0, 0) + lam diag(0, 1, 1) are degenerate for every lam; level 0 is not.
        model = ControlModel(H_d=np.diag([-1.0, 0.0, 0.0]), H_c=np.diag([0.0, 1.0, 1.0]))
        dynamics = Dynamics(model, times=np.linspace(0.0, 1.0, 11), pulse=np.linspace(0.0, 1.0, 11))

        with pytest.warns(NumericalStabilityWarning, match="Eigenstate 1 at control 1 is \\(nearly\\) degenerate"):
            dynamics.state_fidelity(initial_state=0, final_state=1)
        assert dynamics.state_fidelity(initial_state=0, final_state=0) == pytest.approx(1.0)

    def test_custom_pulse_does_not_require_solved_model(self):
        model = ControlModel(H_d=lz_hamiltonian(0.0), H_c=np.diag([1.0, -1.0]))
        times = np.linspace(0.0, 1.0, 11)
        pulse = np.linspace(-1.0, 1.0, times.size)

        dynamics = Dynamics(model, times=times, pulse=pulse)

        assert dynamics._boundary_controls == (-1.0, 1.0)
        assert len(dynamics.average_gate_fidelity(target_gate=qt.identity(2))) == times.size

    def test_custom_inputs_are_copied(self, real_model):
        times = np.linspace(0.0, 1.0, 11)
        pulse = np.linspace(1.0, 3.0, times.size)

        dynamics = Dynamics(real_model, times=times, pulse=pulse)
        times[0] = -5.0
        pulse[0] = -5.0

        assert dynamics.times[0] == 0.0
        assert dynamics.pulse[0] == 1.0

    def test_accepts_lists_and_real_complex_values(self, real_model):
        # times and pulse are annotated as ArrayLike, so lists also pass the type checker (P-08).
        dynamics = Dynamics(real_model, times=[0.0, 0.5, 1.0], pulse=np.array([1.0, 2.0, 3.0]) + 0j)
        from_lists = Dynamics(real_model, times=[0.0, 0.5, 1.0], pulse=[1.0, 2.0, 3.0])

        assert dynamics.pulse.dtype == float
        np.testing.assert_allclose(dynamics.pulse, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(from_lists.pulse, dynamics.pulse)

    def test_duration_with_custom_pulse_raises(self, real_model):
        times = np.linspace(0.0, 1.0, 5)

        with pytest.raises(ValidationError, match="not both"):
            Dynamics(real_model, duration=1.0, times=times, pulse=np.ones(5))

    @pytest.mark.parametrize("missing", ["times", "pulse"])
    def test_times_and_pulse_must_be_given_together(self, real_model, missing: str):
        kwargs: dict[str, Any] = {"times": np.linspace(0.0, 1.0, 5), "pulse": np.ones(5)}
        kwargs.pop(missing)

        with pytest.raises(MissingArgsError, match="must be provided together"):
            Dynamics(real_model, **kwargs)

    @pytest.mark.parametrize(
        ("times", "pulse", "match"),
        [
            (np.zeros((2, 2)), np.ones(4), "times must be one-dimensional"),
            (np.array([0.0]), np.array([1.0]), "at least two samples"),
            (np.array(["a", "b"]), np.ones(2), "times must contain real-valued samples"),
            (np.array([0.0, 1.0j]), np.ones(2), "times must contain real-valued samples"),
            (np.array([0.0, np.nan]), np.ones(2), "times contains NaN or infinite"),
            (np.array([0.0, 1.0, 1.0]), np.ones(3), "strictly increasing"),
            (np.array([1.0, 0.0]), np.ones(2), "strictly increasing"),
            (np.linspace(0.0, 1.0, 4), np.ones((2, 2)), "pulse must be one-dimensional"),
            (np.linspace(0.0, 1.0, 4), np.ones(5), "same length"),
            (np.linspace(0.0, 1.0, 2), np.array(["a", "b"]), "pulse must contain real-valued samples"),
            (np.linspace(0.0, 1.0, 2), np.array([1.0, 1.0j]), "pulse must contain real-valued samples"),
            (np.linspace(0.0, 1.0, 2), 1e-9 * np.array([1.0, 1.0j]), "pulse must contain real-valued samples"),
            (np.linspace(0.0, 1.0, 2), np.array([1.0, np.inf]), "pulse contains NaN or infinite"),
        ],
    )
    def test_invalid_custom_pulse_raises(self, real_model, times, pulse, match: str):
        with pytest.raises(ValidationError, match=match):
            Dynamics(real_model, times=times, pulse=pulse)


# ------------------------------------------------------------
# Interpolation of the pulse between samples (D-01, D-10)
# ------------------------------------------------------------


class TestInterpolationOrder:
    # A coarse piecewise-linear pulse, like discretized_pulse() returns: 4 samples on [0, 5].
    TIMES = np.array([0.0, 5.0 / 3.0, 10.0 / 3.0, 5.0])
    PULSE = np.array([1.0, 2.5, 2.0, 3.0])

    @staticmethod
    def _propagated_hamiltonian(dynamics: Dynamics, t: float) -> np.ndarray:
        return dynamics._qevo(t).full() * dynamics._hbar

    def test_default_order_is_cubic_for_optimal_and_linear_for_custom_pulses(self, real_model):
        assert Dynamics(real_model, duration=2.0)._order == 3
        assert Dynamics(real_model, times=self.TIMES, pulse=self.PULSE)._order == 1

    @pytest.mark.parametrize("model_fixture", ["real_model", "affine_model"])
    @pytest.mark.parametrize("order", [0, 1, 3])
    def test_propagated_hamiltonian_matches_get_ham_between_samples(self, request, model_fixture: str, order: int):
        # Before, the propagation always used a cubic spline while _get_ham interpolated linearly (D-10). For the LZ
        # model H is affine in the control, so both must agree exactly for every order and both model kinds.
        model = request.getfixturevalue(model_fixture)
        dynamics = Dynamics(model, times=self.TIMES, pulse=self.PULSE, order=order)

        for t in np.linspace(self.TIMES[0], self.TIMES[-1], 37):
            np.testing.assert_allclose(self._propagated_hamiltonian(dynamics, t), dynamics._get_ham(t).full(),
                                       atol=1e-10)

    @pytest.mark.parametrize("model_fixture", ["real_model", "affine_model"])
    def test_default_custom_pulse_is_piecewise_linear(self, request, model_fixture: str):
        # The cubic spline deviated from the piecewise-linear discretized pulse between samples (D-01).
        model = request.getfixturevalue(model_fixture)
        dynamics = Dynamics(model, times=self.TIMES, pulse=self.PULSE)

        for t in np.linspace(self.TIMES[0], self.TIMES[-1], 37):
            expected = lz_hamiltonian(float(np.interp(t, self.TIMES, self.PULSE)))
            np.testing.assert_allclose(self._propagated_hamiltonian(dynamics, t), expected, atol=1e-10)

    def test_order_zero_holds_each_sample(self, affine_model):
        dynamics = Dynamics(affine_model, times=self.TIMES, pulse=self.PULSE, order=0)

        for start, end, value in zip(self.TIMES[:-1], self.TIMES[1:], self.PULSE[:-1], strict=True):
            for t in np.linspace(start, end, 5)[:-1]:
                np.testing.assert_allclose(self._propagated_hamiltonian(dynamics, t), lz_hamiltonian(value),
                                           atol=1e-12)

    @pytest.mark.filterwarnings("ignore:.*boundary controls:geodesiq.warnings.GeodesiQWarning")
    def test_linear_interpolation_stays_within_the_sampled_range(self, affine_model):
        # A coarsely sampled step in [-1, 1]: a cubic spline overshoots it, linear interpolation does not.
        times = np.linspace(0.0, 4.0, 9)
        step = np.where(times < 2.0, -1.0, 1.0)
        t_fine = np.linspace(0.0, 4.0, 401)

        linear = Dynamics(affine_model, times=times, pulse=step)
        cubic = Dynamics(affine_model, times=times, pulse=step, order=3)
        linear_values = [self._propagated_hamiltonian(linear, t)[0, 0].real for t in t_fine]
        cubic_values = [self._propagated_hamiltonian(cubic, t)[0, 0].real for t in t_fine]

        assert max(np.abs(linear_values)) <= 1.0 + 1e-12
        assert max(np.abs(cubic_values)) > 1.05

    def test_discretized_pulse_fidelity_depends_on_the_interpolation(self, real_model):
        times, values = real_model.synthesize_pulse(duration=5.0).discretized_pulse(linear_steps=4)

        default = Dynamics(real_model, times=times, pulse=values).state_fidelity()
        linear = Dynamics(real_model, times=times, pulse=values, order=1).state_fidelity()
        cubic = Dynamics(real_model, times=times, pulse=values, order=3).state_fidelity()

        assert default == pytest.approx(linear, rel=1e-12)
        assert abs(cubic - linear) > 1e-6

    @pytest.mark.parametrize("order", [2, 4, -1, 1.0, True, "linear"])
    def test_invalid_order_is_rejected(self, real_model, order: Any):
        with pytest.raises(ValidationError, match="order must be"):
            Dynamics(real_model, duration=2.0, order=order)


# ------------------------------------------------------------
# Validation of explicit states, gates and collapse operators (D-08)
# ------------------------------------------------------------


class TestExplicitStateValidation:
    KET_0 = np.array([1.0, 0.0])
    KET_1 = np.array([0.0, 1.0])

    def test_unnormalized_states_are_rejected(self, default_dynamics):
        # state_fidelity(2a, 2b) returned 15.85 and populations(2a) summed to 4 (D-08).
        with pytest.raises(ValidationError, match="must be normalized"):
            default_dynamics.state_fidelity(initial_state=2 * self.KET_0, final_state=2 * self.KET_1)
        with pytest.raises(ValidationError, match="must be normalized"):
            default_dynamics.populations(initial_state=2 * self.KET_0)
        with pytest.raises(ValidationError, match="unit trace"):
            default_dynamics.state_fidelity(initial_state=2 * np.diag([1.0, 0.0]), final_state=self.KET_1)

    def test_non_hermitian_density_matrix_is_rejected(self, default_dynamics):
        with pytest.raises(ValidationError, match="must be Hermitian"):
            default_dynamics.populations(initial_state=np.array([[0.5, 0.5], [0.0, 0.5]]))

    @pytest.mark.parametrize("bad_state", [qt.basis(3, 0), qt.Qobj(np.ones(4) / 2), 0.5j * qt.basis(2, 0).proj()])
    def test_wrong_qobj_states_raise_validation_error_not_type_error(self, default_dynamics, bad_state):
        # Qobj states skipped the dimension check and failed inside QuTiP with a raw TypeError, or were used as given.
        with pytest.raises(ValidationError):
            default_dynamics.state_fidelity(initial_state=bad_state, final_state=qt.basis(2, 1))

    def test_ket_bra_density_matrix_and_qobj_give_the_same_fidelity(self, default_dynamics):
        reference = default_dynamics.state_fidelity(initial_state=self.KET_0, final_state=self.KET_1)

        equivalent_inputs = [
            (qt.basis(2, 0), qt.basis(2, 1)),
            (self.KET_0.reshape(1, 2), self.KET_1.reshape(1, 2)),
            (np.outer(self.KET_0, self.KET_0), qt.basis(2, 1)),
            (self.KET_0.reshape(2, 1), self.KET_1),
        ]
        # A density matrix is propagated with a different equation, so agreement is at the solver tolerance (1e-6).
        for initial, final in equivalent_inputs:
            fidelity = default_dynamics.state_fidelity(initial_state=initial, final_state=final)
            assert fidelity == pytest.approx(reference, abs=1e-5)

    def test_wrong_gate_or_target_dimension_raises_validation_error(self, default_dynamics):
        with pytest.raises(ValidationError, match="target_gate must have shape"):
            default_dynamics.average_gate_fidelity(target_gate=qt.qeye(3), final_only=True)
        with pytest.raises(ValidationError, match="gate must have shape"):
            default_dynamics.average_gate_fidelity(gate=qt.qeye(3), target_gate=qt.sigmax())
        with pytest.raises(ValidationError, match=r"gate\[1\] must have shape"):
            default_dynamics.average_gate_fidelity(gate=[qt.qeye(2), qt.qeye(3)], target_gate=qt.sigmax())

    def test_collapse_operators_are_validated_and_accept_tuples(self, default_dynamics):
        dephasing = np.sqrt(1e-3) * qt.sigmaz()

        from_list = default_dynamics.state_fidelity(c_ops=[dephasing])
        from_tuple = default_dynamics.state_fidelity(c_ops=(dephasing,))
        from_array = default_dynamics.state_fidelity(c_ops=[dephasing.full()])
        assert from_tuple == pytest.approx(from_list, abs=1e-12)
        assert from_array == pytest.approx(from_list, abs=1e-12)

        # A mixed tuple of Qobj and array matches the c_ops annotation (P-07); two halves of the rate add up.
        half = np.sqrt(0.5e-3) * qt.sigmaz()
        from_mixed = default_dynamics.state_fidelity(c_ops=(half, half.full()))
        assert from_mixed == pytest.approx(from_list, abs=1e-8)

        with pytest.raises(ValidationError, match=r"c_ops\[0\] must have shape"):
            default_dynamics.state_fidelity(c_ops=[qt.qeye(3)])
        with pytest.raises(ValidationError, match=r"c_ops\[0\] must be a Qobj or a numpy array"):
            default_dynamics.populations(c_ops=["dephasing"])


# ------------------------------------------------------------
# Testing gate and state transfer fidelity
# ------------------------------------------------------------


def test_time_evolution_operator(default_dynamics):
    """Ensure the propagator computes successfully and returns expected elements."""
    U_list = default_dynamics.time_evolution_operator()

    assert isinstance(U_list, list)
    assert len(U_list) == len(default_dynamics._pulse_times)
    assert isinstance(U_list[0], qt.Qobj)
    assert U_list[0].isket is False  # Propagators must be operators (matrices)


def test_time_evolution_operator_starts_as_identity(default_dynamics):
    """The propagator at t=0 should be the identity operator."""
    U_list = default_dynamics.time_evolution_operator()

    np.testing.assert_allclose(U_list[0].full(), np.eye(2))


def test_time_evolution_operator_wraps_single_qobj(default_dynamics, monkeypatch):
    """When qutip.propagator returns a single Qobj, the API should still return a list."""
    monkeypatch.setattr(qt, "propagator", lambda *args, **kwargs: qt.identity(2))

    U_list = default_dynamics.time_evolution_operator()

    assert isinstance(U_list, list)
    assert len(U_list) == 1
    assert isinstance(U_list[0], qt.Qobj)


def test_state_fidelity_eigenstates(default_dynamics):
    """Test fidelity execution when initial_state/final_state are implicitly None."""
    # This triggers the if-branch using the initial/final state indices
    fidelity = default_dynamics.state_fidelity(initial_state=None, final_state=None)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity < 1.0


def test_state_fidelity_explicit_arrays(default_dynamics):
    """Test fidelity calculation with explicitly passed state vector numpy arrays."""
    # Create 2-level state vectors: ground |0> and excited |1>
    psi_i = np.array([[1.0], [0.0]])
    psi_f = np.array([[0.0], [1.0]])

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f)
    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity < 1.0


def test_state_fidelity_qobj_states(default_dynamics):
    """Qobj inputs should be accepted directly without array conversion."""
    psi_i = qt.basis(2, 0)
    psi_f = qt.basis(2, 1)

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_invalid_dimensions(default_dynamics):
    """Verify that array dimensional mismatches throw a ValidationError."""
    # Our ControlModel has dimensions 2x2. Let's pass a 3-level state.
    bad_state = np.array([[1.0], [0.0], [0.0]])
    valid_state = np.array([[1.0], [0.0]])

    with pytest.raises(ValidationError, match="must be a ket of shape"):
        default_dynamics.state_fidelity(initial_state=bad_state, final_state=valid_state)


def test_state_fidelity_type_mismatch_error(default_dynamics):
    """Passing invalid types (like strings) must raise a ValidationError."""
    with pytest.raises(ValidationError, match="initial_state must be an integer index, a Qobj or a numpy array"):
        default_dynamics.state_fidelity(initial_state=cast(Any, "invalid_type"), final_state=1)


def test_state_fidelity_accepts_mixed_state_kinds(default_dynamics):
    # D-15: the two states are resolved independently, as in populations(); mixing kinds gives the same result as
    # the equivalent indices.
    initial_control, final_control = default_dynamics._boundary_controls
    psi_0 = default_dynamics._eigenstate(initial_control, 0)
    psi_1_final = default_dynamics._eigenstate(final_control, 1)
    reference = default_dynamics.state_fidelity(0, 1)

    assert default_dynamics.state_fidelity(0, psi_1_final) == pytest.approx(reference, abs=1e-10)
    assert default_dynamics.state_fidelity(psi_0.full(), 1) == pytest.approx(reference, abs=1e-10)
    # Omitted initial state: the model's index (0); explicit target
    assert default_dynamics.state_fidelity(final_state=psi_1_final) == pytest.approx(reference, abs=1e-10)
    # A density-matrix target gives the same transition probability for a pure state
    rho_target = psi_1_final * psi_1_final.dag()
    assert default_dynamics.state_fidelity(0, rho_target) == pytest.approx(reference, abs=1e-8)


def test_state_fidelity_invalid_c_ops_container(default_dynamics):
    """Collapse operators must be provided as a list container."""
    with pytest.raises(ValidationError, match="Collapse operators must be provided as a list"):
        default_dynamics.state_fidelity(c_ops=cast(Any, "not_a_list"))


def test_state_fidelity_c_ops_numpy_array_conversion(default_dynamics):
    """Numpy-array collapse operators should be converted and accepted by mesolve."""
    psi_i = qt.basis(2, 0)
    psi_f = qt.basis(2, 1)
    c_ops = [0.1 * qt.sigmaz().full()]

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f, c_ops=c_ops)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_integer_state_indices(default_dynamics):
    """Integer state indices should select eigenstates at pulse boundaries."""
    fidelity = default_dynamics.state_fidelity(initial_state=0, final_state=1)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_raises_when_mesolve_returns_no_final_state(default_dynamics, monkeypatch):
    """A mesolve result without final_state should raise a ValidationError."""

    class DummyResult:
        final_state = None

    monkeypatch.setattr(qt, "mesolve", lambda *args, **kwargs: DummyResult())

    with pytest.raises(ValidationError, match="did not return a final state"):
        default_dynamics.state_fidelity(initial_state=qt.basis(2, 0), final_state=qt.basis(2, 1))


def test_average_gate_fidelity(default_dynamics):
    """Ensure average gate fidelity is calculated correctly against a target operator."""
    # Target gate: Identity gate for a 2-level system
    target = qt.identity(2)

    gate_fid_list = default_dynamics.average_gate_fidelity(target_gate=target)

    assert isinstance(gate_fid_list, list)
    assert len(gate_fid_list) == len(default_dynamics._pulse_times)
    assert all(0.0 <= f <= 1.0 for f in gate_fid_list)


def test_average_gate_fidelity_accepts_numpy_target(default_dynamics):
    """A numpy target gate should be promoted to Qobj internally."""
    gate_fid_list = default_dynamics.average_gate_fidelity(target_gate=np.eye(2))

    assert isinstance(gate_fid_list, list)
    assert len(gate_fid_list) == len(default_dynamics._pulse_times)
    assert all(0.0 <= f <= 1.0 for f in gate_fid_list)


def test_average_gate_fidelity_single_qobj_gate(default_dynamics):
    """A single Qobj gate should be accepted and treated as a one-item list."""
    gate_fid_list = default_dynamics.average_gate_fidelity(gate=qt.identity(2), target_gate=qt.identity(2))

    assert gate_fid_list == [pytest.approx(1.0)]


def test_average_gate_fidelity_list_of_qobj_gate(default_dynamics):
    """A list of Qobj gates should be accepted directly."""
    gate_fid_list = default_dynamics.average_gate_fidelity(gate=[qt.identity(2)], target_gate=qt.identity(2))

    assert gate_fid_list == [pytest.approx(1.0)]


def test_average_gate_fidelity_invalid_gate_type(default_dynamics):
    """Invalid gate types should raise a ValidationError."""
    with pytest.raises(ValidationError, match="Gate must be a Qobj or a list of Qobj instances"):
        default_dynamics.average_gate_fidelity(gate=cast(Any, [qt.identity(2), np.eye(2)]), target_gate=qt.identity(2))


def test_average_gate_fidelity_requires_a_target_gate(default_dynamics):
    # Without a target, QuTiP silently compared with the identity (D-09).
    with pytest.raises(TypeError, match="target_gate"):
        cast(Any, default_dynamics).average_gate_fidelity()
    with pytest.raises(ValidationError, match="target_gate must be a Qobj or a numpy array"):
        default_dynamics.average_gate_fidelity(target_gate=cast(Any, None))


def test_average_gate_fidelity_target_gate_is_keyword_only(default_dynamics):
    # A second positional argument was the target gate before; it must not be read as something else silently.
    with pytest.raises(TypeError):
        cast(Any, default_dynamics).average_gate_fidelity(qt.identity(2), qt.sigmax())


# ---------------------------------------------------------------------------
# Testing populations and plot_populations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["adiabatic", "diabatic"])
def test_populations_shape_and_normalization(default_dynamics, basis):
    """One row per time sample, one column per state, and every row sums to one."""
    populations = default_dynamics.populations(basis=basis)

    assert populations.shape == (len(default_dynamics.times), 2)
    assert np.all(populations >= -1e-12)
    np.testing.assert_allclose(populations.sum(axis=1), 1.0, atol=1e-6)


def test_populations_start_in_the_initial_eigenstate(default_dynamics):
    """The default initial state is the configured eigenstate, fully populated at t = 0."""
    populations = default_dynamics.populations()

    np.testing.assert_allclose(populations[0], [1.0, 0.0], atol=1e-10)


def test_final_adiabatic_population_matches_state_fidelity(default_dynamics):
    """For a pure state, the final population of eigenstate n is the fidelity of the transfer to it."""
    populations = default_dynamics.populations(initial_state=0)

    assert populations[-1, 1] == pytest.approx(default_dynamics.state_fidelity(0, 1), abs=1e-6)
    assert populations[-1, 0] == pytest.approx(default_dynamics.state_fidelity(0, 0), abs=1e-6)


def test_diabatic_populations_of_a_basis_state(default_dynamics):
    """An explicit computational basis state is fully populated at t = 0 in the diabatic basis."""
    populations = default_dynamics.populations(initial_state=np.array([0.0, 1.0]), basis="diabatic")

    np.testing.assert_allclose(populations[0], [0.0, 1.0], atol=1e-12)


def test_populations_of_density_matrix_match_state_vector(default_dynamics):
    """With vanishing collapse operators the density-matrix branch reproduces the closed-system populations."""
    closed = default_dynamics.populations(initial_state=0)
    open_system = default_dynamics.populations(initial_state=0, c_ops=[np.zeros((2, 2))])

    np.testing.assert_allclose(open_system, closed, atol=1e-6)


def test_populations_reject_unknown_basis(default_dynamics):
    with pytest.raises(ValidationError, match="basis must be"):
        default_dynamics.populations(basis=cast(Any, "computational"))


def test_populations_reject_invalid_initial_state(default_dynamics):
    with pytest.raises(ValidationError, match="must be a ket of shape"):
        default_dynamics.populations(initial_state=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValidationError, match="must be between"):
        default_dynamics.populations(initial_state=2)
    with pytest.raises(ValidationError, match="initial_state must be an integer index"):
        default_dynamics.populations(initial_state=cast(Any, "ground"))


def test_populations_require_a_configured_initial_state(default_dynamics):
    default_dynamics._initial_state = None

    with pytest.raises(ConfigurationError, match="initial state index"):
        default_dynamics.populations()


def test_populations_raise_when_mesolve_misses_states(default_dynamics, monkeypatch):
    class DummyResult:
        states = [qt.basis(2, 0)]

    monkeypatch.setattr(qt, "mesolve", lambda *args, **kwargs: DummyResult())

    with pytest.raises(ValidationError, match="state for every time sample"):
        default_dynamics.populations()


def test_plot_populations_draws_one_line_per_state(default_dynamics):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    fig, ax = default_dynamics.plot_populations()

    assert isinstance(ax, Axes)
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == [r"$|E_{0}\rangle$", r"$|E_{1}\rangle$"]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), default_dynamics.times)
    assert ax.get_xlabel() == "Time $t$"
    assert ax.get_title() == "Adiabatic populations"
    plt.close(fig)


def test_plot_populations_on_provided_axes_with_selected_states(default_dynamics):
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2)
    out_fig, out_ax = default_dynamics.plot_populations(ax=ax_right, basis="diabatic", states=[1], legend=False)

    assert out_ax is ax_right
    assert out_fig is fig
    assert len(ax_right.lines) == 1
    assert ax_right.lines[0].get_label() == r"$|1\rangle$"
    assert ax_right.get_legend() is None
    assert len(ax_left.lines) == 0
    plt.close(fig)


def test_plot_populations_on_provided_figure(default_dynamics):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    out_fig, ax = default_dynamics.plot_populations(fig=fig, title="LZ", ylabel="P")

    assert out_fig is fig
    assert ax in fig.axes
    assert ax.get_title() == "LZ"
    assert ax.get_ylabel() == "P"
    plt.close(fig)


@pytest.mark.parametrize(("kwargs", "expected_calls"), [({}, 1), ({"show": False}, 0)])
def test_plot_populations_calls_show_by_default(default_dynamics, monkeypatch, kwargs, expected_calls):
    """Like plot_pulse(), plot_populations() calls plt.show() unless show=False (W2-102)."""
    import matplotlib.pyplot as plt

    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(1))

    fig, _ = default_dynamics.plot_populations(**kwargs)

    assert len(calls) == expected_calls
    plt.close(fig)


def test_plot_populations_rejects_out_of_range_states(default_dynamics):
    with pytest.raises(ValidationError, match="states must be between"):
        default_dynamics.plot_populations(states=[0, 2])


# ---------------------------------------------------------------------------
# Test initial and final indices
# ---------------------------------------------------------------------------

class TestIndices:
    @pytest.mark.parametrize(("initial_state", "final_state"), [(2, 0), (0, 2), (2, 2)], )
    def test_out_of_range_control_state_indices_raise(self,
                                                      default_dynamics,
                                                      initial_state: int,
                                                      final_state: int, ) -> None:
        with pytest.raises(ValidationError, match="must be between", ):
            default_dynamics.state_fidelity(initial_state=initial_state, final_state=final_state)

    @pytest.mark.parametrize(("initial_state", "final_state"), [(0, 0), (0, 1), (1, 0), (1, 1), ], )
    def test_valid_control_state_indices_are_accepted(self,
                                                      default_dynamics,
                                                      initial_state: int,
                                                      final_state: int, ) -> None:
        default_dynamics.state_fidelity(initial_state=initial_state, final_state=final_state)


# ---------------------------------------------------------------------------
# NumPy indices, final-only propagators and unit consistency
# ---------------------------------------------------------------------------


def test_state_fidelity_accepts_numpy_integer_indices(default_dynamics):
    expected = default_dynamics.state_fidelity(initial_state=0, final_state=1)

    fidelity = default_dynamics.state_fidelity(initial_state=np.int64(0), final_state=np.int32(1))

    assert fidelity == pytest.approx(expected)


def test_state_fidelity_rejects_bool_indices(default_dynamics):
    with pytest.raises(ValidationError, match="integer index, a Qobj or a numpy array"):
        default_dynamics.state_fidelity(initial_state=cast(Any, True), final_state=cast(Any, False))


def test_adiabatic_basis_is_computed_once_per_object(default_dynamics, monkeypatch):
    """populations(basis="adiabatic") evaluates H at every sample once, not on every call (D-19)."""
    hamiltonian = default_dynamics._hamiltonian
    evaluations = []

    def counting_hamiltonian(control_value):
        evaluations.append(control_value)
        return hamiltonian(control_value)

    monkeypatch.setattr(default_dynamics, "_hamiltonian", counting_hamiltonian)
    # An explicit initial state, so that only the basis evaluates H (an index would add one eigenstate per call).
    psi0 = np.array([1.0, 0.0])

    first = default_dynamics.populations(initial_state=psi0, basis="adiabatic")
    assert len(evaluations) == len(default_dynamics.times)
    second = default_dynamics.populations(initial_state=psi0, basis="adiabatic")
    assert len(evaluations) == len(default_dynamics.times)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("fixture_name", ["real_model", "affine_model"])
def test_final_only_propagator_matches_last_full_propagator(request, fixture_name: str):
    model = request.getfixturevalue(fixture_name)
    dynamics = Dynamics(model, duration=2.0)

    full = dynamics.time_evolution_operator()
    final = dynamics.time_evolution_operator(final_only=True)

    assert len(final) == 1
    # final_only outputs only the end of the pulse, so the two agree within the solver tolerance (rtol=1e-6 for the
    # default adams method), not to round-off (D-19). Tighter tolerances bring them together.
    np.testing.assert_allclose(final[0].full(), full[-1].full(), atol=1e-4)
    tight = Dynamics(model, duration=2.0, options={"atol": 1e-12, "rtol": 1e-10})
    np.testing.assert_allclose(tight.time_evolution_operator(final_only=True)[0].full(),
                               tight.time_evolution_operator()[-1].full(), atol=1e-8)


def test_final_only_gate_fidelity_matches_last_value(affine_dynamics):
    full = affine_dynamics.average_gate_fidelity(target_gate=qt.sigmax())
    final = affine_dynamics.average_gate_fidelity(target_gate=qt.sigmax(), final_only=True)

    assert len(final) == 1
    assert final[0] == pytest.approx(full[-1], abs=1e-6)


@pytest.mark.parametrize("scale", [6.62607015e-25, 1e9])
def test_dynamics_is_unit_consistent(scale: float):
    """Scaling H and hbar by the same factor leaves the dynamics in physical time unchanged."""
    H_d = np.array([[0.0, 1.0], [1.0, 0.0]])
    H_c = np.array([[1.0, 0.0], [0.0, -1.0]])

    def build(factor: float) -> ControlModel:
        model = ControlModel(H_d=factor * H_d, H_c=factor * H_c)
        model.set_control(pulse_initial=1.0, pulse_final=3.0, initial_state=0, alpha=2.0, beta=2.0, num_steps=33, )
        model.solve_problem(pulse_accuracy=50)
        return model

    reference = Dynamics(build(1.0), duration=2.0)
    scaled = Dynamics(build(scale), duration=2.0, hbar=scale)

    np.testing.assert_allclose(scaled.pulse, reference.pulse, rtol=1e-7)
    assert scaled.state_fidelity(0, 1) == pytest.approx(reference.state_fidelity(0, 1), abs=1e-6)


# ---------------------------------------------------------------------------
# Physics: Landau-Zener transition probability
# ---------------------------------------------------------------------------


class TestLandauZenerFormula:
    """
    For H = lam(t) sigma_z + Delta sigma_x with a linear sweep lam(t) = v t, the probability of a diabatic transition
    (ending in the excited adiabatic state) is P = exp(-pi Delta^2 / (hbar v)). The adiabatic ground-to-ground
    fidelity is therefore 1 - P. The finite sweep range adds small oscillatory corrections (~1e-4 for L = 40 Delta).
    """

    DELTA = 1.0
    SWEEP = 40.0  # lam goes from -SWEEP to +SWEEP

    def _linear_ramp_fidelity(self, rate: float, hbar: float) -> float:
        model = ControlModel(H_d=self.DELTA * np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
        model.set_control(pulse_initial=-self.SWEEP, pulse_final=self.SWEEP, initial_state=0, alpha=2.0, beta=2.0)

        duration = 2 * self.SWEEP / rate
        times = np.linspace(0.0, duration, 4001)
        pulse = np.linspace(-self.SWEEP, self.SWEEP, times.size)

        return Dynamics(model, times=times, pulse=pulse, hbar=hbar).state_fidelity(initial_state=0, final_state=0)

    @pytest.mark.parametrize("rate", [1.0, 2.6, 6.0, 20.0])
    def test_linear_ramp_matches_landau_zener(self, rate: float):
        expected = 1.0 - np.exp(-np.pi * self.DELTA**2 / rate)

        assert self._linear_ramp_fidelity(rate, hbar=1.0) == pytest.approx(expected, abs=2e-3)

    def test_final_excited_population_matches_landau_zener(self):
        """The final population of the excited adiabatic state is the Landau-Zener probability."""
        rate = 2.6
        model = ControlModel(H_d=self.DELTA * np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
        model.set_control(pulse_initial=-self.SWEEP, pulse_final=self.SWEEP, initial_state=0, alpha=2.0, beta=2.0)
        times = np.linspace(0.0, 2 * self.SWEEP / rate, 4001)
        pulse = np.linspace(-self.SWEEP, self.SWEEP, times.size)

        populations = Dynamics(model, times=times, pulse=pulse).populations(initial_state=0)

        assert populations[-1, 1] == pytest.approx(np.exp(-np.pi * self.DELTA**2 / rate), abs=2e-3)

    def test_hbar_enters_the_landau_zener_exponent(self):
        rate, hbar = 2.0, 0.5
        expected = 1.0 - np.exp(-np.pi * self.DELTA**2 / (hbar * rate))

        assert self._linear_ramp_fidelity(rate, hbar=hbar) == pytest.approx(expected, abs=2e-3)

    def test_optimal_pulse_beats_linear_ramp_of_same_duration(self):
        """The geodesic pulse is designed to be more adiabatic than a linear ramp of the same duration."""
        model = ControlModel(H_d=self.DELTA * np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
        model.set_control(pulse_initial=-self.SWEEP, pulse_final=self.SWEEP, initial_state=0, alpha=2.0, beta=2.0)
        model.solve_problem(pulse_accuracy=2001)
        rate = 6.0
        duration = 2 * self.SWEEP / rate

        optimal = Dynamics(model, duration=duration).state_fidelity(initial_state=0, final_state=0)

        assert optimal > self._linear_ramp_fidelity(rate, hbar=1.0) + 0.1


class TestCallableHamiltonianDynamics:
    """Callable models propagate the SVD decomposition of the sampled H(x(t)) (D-11, D-20 (ii) and (iii))."""

    TIGHT = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000}
    SX = np.array([[0.0, 1.0], [1.0, 0.0]])
    SZ = np.diag([1.0, -1.0])

    @classmethod
    def _nonlinear_hamiltonian(cls, lam: float) -> np.ndarray:
        return lam * cls.SZ + (1.0 + 0.3 * lam**2) * cls.SX

    def test_converges_to_the_exact_hamiltonian_of_the_pulse(self):
        """Against H(x(t)) evaluated exactly, the error falls as the square of the sample spacing."""
        model = ControlModel(self._nonlinear_hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-2.0, pulse_final=2.0, initial_state=0, alpha=2.0,
                          beta=2.0)
        duration = 5.0
        errors = []
        for samples in (11, 41, 161):
            times = np.linspace(0.0, duration, samples)
            pulse = -2.0 + 4.0 * (times / duration) ** 2
            propagator = Dynamics(model, times=times, pulse=pulse, options=self.TIGHT).time_evolution_operator(
                final_only=True)[0].full()

            # Exact reference: the order=1 pulse (linear interpolation) inserted into the nonlinear H.
            exact_hamiltonian = qt.QobjEvo(
                lambda t, times=times, pulse=pulse: qt.Qobj(self._nonlinear_hamiltonian(np.interp(t, times, pulse))))
            reference = qt.sesolve(exact_hamiltonian, qt.qeye(2), [0.0, duration],
                                   options={**self.TIGHT, "max_step": 0.01}).states[-1].full()
            errors.append(np.max(np.abs(propagator - reference)))

        assert errors[-1] < 5e-4
        # Four times more samples reduce the error about 16-fold (second order); allow some slack.
        assert errors[1] < errors[0] / 10
        assert errors[2] < errors[1] / 10

    def test_callable_and_affine_models_give_the_same_dynamics(self):
        """The same Landau-Zener problem as a callable and as an affine model gives the same pulse and evolution."""
        callable_model = ControlModel(lambda lam: lam * self.SZ + self.SX)
        callable_model.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0,
                                   alpha=2.0, beta=2.0)
        affine_model = ControlModel(H_d=self.SX, H_c=self.SZ)
        affine_model.set_control(pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0, beta=2.0)

        callable_dynamics = Dynamics(callable_model, duration=10.0, options=self.TIGHT)
        affine_dynamics = Dynamics(affine_model, duration=10.0, options=self.TIGHT)

        np.testing.assert_allclose(callable_dynamics.pulse, affine_dynamics.pulse, atol=1e-10)
        np.testing.assert_allclose(callable_dynamics.time_evolution_operator(final_only=True)[0].full(),
                                   affine_dynamics.time_evolution_operator(final_only=True)[0].full(), atol=1e-7)
        np.testing.assert_allclose(callable_dynamics.populations(), affine_dynamics.populations(), atol=1e-8)
        assert callable_dynamics.state_fidelity() == pytest.approx(affine_dynamics.state_fidelity(), abs=1e-8)
