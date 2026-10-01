from typing import Any, cast

import numpy as np
import pytest
import qutip as qt

import geodesiq.dynamics as dynamics_module
from geodesiq import ControlModel
from geodesiq.dynamics import Dynamics
from geodesiq.exceptions import ConfigurationError, MissingArgsError, ValidationError

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
                      beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33, )
    model.solve_problem(pulse_accuracy=pulse_accuracy)
    return model


def _build_solved_affine_model(pulse_accuracy: int = 5) -> ControlModel:
    H_d = np.array([[0.0, 1.0], [1.0, 0.0]])
    H_c = np.array([[1.0, 0.0], [0.0, -1.0]])
    model = ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(pulse_initial=1.0, pulse_final=3.0, initial_state=0, final_state=1, alpha=2.0,
                      beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33, )
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
                          alpha=2.0, beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=33, )

        dynamics = Dynamics(model, duration=2.0)

        np.testing.assert_allclose(dynamics.pulse, model.control_sol)


class TestCustomPulse:
    """Explicit times and pulse are given, e.g. after filtering the pulse with PulseControl."""

    def test_reproduces_optimal_pulse_dynamics(self, real_model):
        """Feeding the optimal pulse explicitly must give the same result as passing only the duration."""
        optimal = Dynamics(real_model, duration=2.0)
        custom = Dynamics(real_model, times=optimal.times, pulse=optimal.pulse)

        assert custom.state_fidelity() == pytest.approx(optimal.state_fidelity(), rel=1e-8)
        np.testing.assert_allclose(
            custom.time_evolution_operator()[-1].full(), optimal.time_evolution_operator()[-1].full(), atol=1e-8
        )

    def test_reproduces_optimal_pulse_dynamics_affine(self, affine_model):
        optimal = Dynamics(affine_model, duration=2.0)
        custom = Dynamics(affine_model, times=optimal.times, pulse=optimal.pulse)

        assert custom.state_fidelity() == pytest.approx(optimal.state_fidelity(), rel=1e-8)

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

        dynamics = Dynamics(real_model, times=times, pulse=pulse)

        assert dynamics._boundary_controls == (1.0, 3.0)

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
        dynamics = Dynamics(real_model, times=cast(Any, [0.0, 0.5, 1.0]), pulse=np.array([1.0, 2.0, 3.0]) + 0j)

        assert dynamics.pulse.dtype == float
        np.testing.assert_allclose(dynamics.pulse, [1.0, 2.0, 3.0])

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
            (np.linspace(0.0, 1.0, 2), np.array([1.0, np.inf]), "pulse contains NaN or infinite"),
        ],
    )
    def test_invalid_custom_pulse_raises(self, real_model, times, pulse, match: str):
        with pytest.raises(ValidationError, match=match):
            Dynamics(real_model, times=times, pulse=pulse)


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

    with pytest.raises(ValidationError, match="must have the same dimension"):
        default_dynamics.state_fidelity(initial_state=bad_state, final_state=valid_state)


def test_state_fidelity_type_mismatch_error(default_dynamics):
    """Passing mixed or invalid types (like strings) must raise a ValidationError."""
    with pytest.raises(ValidationError, match="either integers, numpy arrays with correct dimensions"):
        default_dynamics.state_fidelity(initial_state=cast(Any, "invalid_type"), final_state=1)


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
    with pytest.raises(ValidationError, match="either integers"):
        default_dynamics.state_fidelity(initial_state=cast(Any, True), final_state=cast(Any, False))


@pytest.mark.parametrize("fixture_name", ["default_dynamics", "affine_dynamics"])
def test_final_only_propagator_matches_last_full_propagator(request, fixture_name: str):
    dynamics = request.getfixturevalue(fixture_name)

    full = dynamics.time_evolution_operator()
    final = dynamics.time_evolution_operator(final_only=True)

    assert len(final) == 1
    np.testing.assert_allclose(final[0].full(), full[-1].full(), atol=1e-6)


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
