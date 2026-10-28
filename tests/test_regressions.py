import functools
import tracemalloc
import warnings
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.integrate import romb, solve_ivp
from scipy.interpolate import PchipInterpolator

from geodesiq import ControlModel
from geodesiq._utils import _gap_minima, is_hermitian
from geodesiq.exceptions import (
    InvalidControlParameterError,
    MetricComputationError,
    MissingControlParameterError,
    SolverError,
    ValidationError,
)
from geodesiq.warnings import NumericalStabilityWarning

# ---------------------------------------------------------------------------
# Configures Hamiltonians and models
# ---------------------------------------------------------------------------

def lz_hamiltonian(lam: float, delta: float = 0.5) -> np.ndarray:
    return np.array([[lam, delta], [delta, -lam]], dtype=float)


def lz_partial(lam: float, delta: float = 0.5) -> np.ndarray:
    del lam, delta
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)


def configured_model(*, analytical: bool = True) -> ControlModel:
    model = ControlModel(lz_hamiltonian, lz_partial if analytical else None)
    model.set_parameters(delta=0.5)
    model.set_control(control_name="lam", pulse_initial=-3.0, pulse_final=3.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=65, )
    return model


def shifted_hamiltonian(lam: float, shift: float, ) -> np.ndarray:
    return np.array([[lam + shift, 1.0], [1.0, -lam + shift], ], dtype=float, )


def shifted_partial(lam: float, shift: float, ) -> np.ndarray:
    del lam, shift

    return np.array([[1.0, 0.0], [0.0, -1.0], ], dtype=float, )


def make_model(shift: float) -> ControlModel:
    model = ControlModel(shifted_hamiltonian, shifted_partial, )

    model.set_parameters(shift=shift)

    model.set_control(control_name="lam", pulse_initial=-2.0, pulse_final=2.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=65, )

    model.solve_problem(pulse_accuracy=50)

    return model


def degenerate_hamiltonian(lam: float, ) -> np.ndarray:
    # Two copies of a Landau-Zener system: every level, including the occupied one, is degenerate at every point. (A
    # crossing at a single point, such as lam * sz at lam = 0, is a true crossing and is followed instead, C-07.)
    return np.kron(np.eye(2), np.array([[lam, 0.1], [0.1, -lam], ], dtype=float, ))


def degenerate_partial(lam: float, ) -> np.ndarray:
    del lam

    return np.kron(np.eye(2), np.array([[1.0, 0.0], [0.0, -1.0], ], dtype=float, ))


def dqd_hamiltonian(eps: float, U: float, tc: float, Ez: float, dEz: float, dEx: float):
    ham = np.array([[U - eps, 0, -tc, tc, 0], [0, Ez, dEx, -dEx, 0], [-tc, dEx, dEz, 0, dEx], [tc, -dEx, 0, -dEz, -dEx],
                    [0, 0, dEx, -dEx, -Ez]])
    return ham


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_numerical_derivative_supports_complex_hamiltonians():
    def hamiltonian(z: float) -> np.ndarray:
        return np.array([[z, 1j], [-1j, -z]], dtype=complex)

    model = ControlModel(hamiltonian)
    model.set_control(control_name="z", pulse_initial=-2.0, pulse_final=2.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=65, )
    model.solve_problem(pulse_accuracy=80)

    assert model._flags["ode_solved"]
    assert np.all(np.isfinite(model.control_sol))


def test_numerical_derivative_is_complex_preserving_and_accurate():
    def hamiltonian(z: float) -> np.ndarray:
        return np.array([[z ** 2, 1j * z], [-1j * z, -(z ** 2)]], dtype=complex)

    model = ControlModel(hamiltonian)
    model.set_control(control_name="z", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=0.0, beta=0.0,
                      num_steps=65, )
    model._solve_eigenproblem()
    derivative = model._compute_numerical_partial_H()
    z = model.control_pulse
    expected = np.array([[[2 * value, 1j], [-1j, -2 * value]] for value in z])
    np.testing.assert_allclose(derivative, expected, atol=1e-10)


def test_parameter_cannot_override_control_sweep():
    model = ControlModel(lz_hamiltonian, lz_partial)
    model.set_control(control_name="lam")
    with pytest.raises(InvalidControlParameterError, match="as a fixed parameter"):
        model.set_parameters(lam=7.0)


def test_control_name_cannot_collide_with_existing_parameter():
    model = ControlModel(lz_hamiltonian, lz_partial)
    model.set_parameters(lam=7.0)
    with pytest.raises(InvalidControlParameterError, match="collides"):
        model.set_control(control_name="lam")


def test_control_value_wins_in_internal_evaluation():
    model = configured_model()
    np.testing.assert_allclose(model.evaluate_hamiltonian(-1.0), lz_hamiltonian(-1.0, delta=0.5))
    np.testing.assert_allclose(model.evaluate_hamiltonian(1.0), lz_hamiltonian(1.0, delta=0.5))


def test_zero_metric_raises_instead_of_entering_solver():
    def constant_hamiltonian(z: float) -> np.ndarray:
        del z
        return np.diag([0.0, 1.0])

    def zero_partial(z: float) -> np.ndarray:
        return np.zeros((2, 2))

    model = ControlModel(constant_hamiltonian, zero_partial)
    model.set_control(control_name="z", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=33, )
    with pytest.raises(MetricComputationError, match="zero or numerically singular"):
        model.solve_problem()


def _isolated_zero_metric_model() -> ControlModel:
    # The coupling x**3 vanishes exactly at the grid point x = 0, so the metric is zero there (C-04).
    sz = np.diag([1.0, -1.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    model = ControlModel(lambda x: x ** 3 * sx + sz, partial_H_func=lambda x: 3 * x ** 2 * sx)
    model.set_control(control_name="x", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=129, )
    return model


@pytest.mark.parametrize("solver_kwargs", [None, {"rtol": 1e-10}])
def test_isolated_zero_of_the_metric_is_passed_by_the_default_solver(solver_kwargs):
    # The default solver integrates dtau/dx = sqrt(G) / a_tilde, which is finite at G = 0. Before, the speed
    # a_tilde / sqrt(G) was built first and raised "ODE speed factor is non-finite or non-positive".
    model = _isolated_zero_metric_model()

    with pytest.warns(NumericalStabilityWarning, match="near control value\\(s\\) 0"):
        model.solve_problem(pulse_accuracy=401, solver_kwargs=solver_kwargs)

    pulse = model.control_sol
    assert pulse[0] == -1.0 and pulse[-1] == 1.0
    assert np.all(np.diff(pulse) >= 0)
    # The problem is symmetric in x, so the zero at x = 0 is passed at half the normalized time. The pulse jumps
    # through it, so tau(x = 0) is only resolved to one sample of the pulse.
    assert abs(np.interp(0.0, pulse, model.tau) - 0.5) <= 1.0 / (401 - 1)


@pytest.mark.parametrize("solver_kwargs", [None, {"rtol": 1e-10}])
def test_trailing_zero_metric_interpolates_to_the_final_control_value(monkeypatch, solver_kwargs):
    # The coupling vanishes for x >= 1, so the metric is zero on (1, 2] and tau(x) is flat there. The x(tau)
    # interpolant used to keep the first node of that stretch, ending at x(1) ~ 1 instead of 2, and only the
    # endpoint overwrite hid it (C-14).
    sz = np.diag([1.0, -1.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    model = ControlModel(lambda x: x * sz + 0.3 * max(0.0, 1.0 - x) ** 3 * sx,
                         partial_H_func=lambda x: sz - 0.9 * max(0.0, 1.0 - x) ** 2 * sx)
    model.set_control(control_name="x", pulse_initial=-1.0, pulse_final=2.0, initial_state=0, alpha=2.0, beta=1.0,
                      num_steps=513, )

    # Record the nodes of each interpolant; the last one built in _solve_ode is x(tau).
    calls = []

    def recording_pchip(x, y, *args, **kwargs):
        calls.append((np.asarray(x), np.asarray(y)))
        return PchipInterpolator(x, y, *args, **kwargs)

    monkeypatch.setattr("geodesiq.controlmodel.PchipInterpolator", recording_pchip)

    with pytest.warns(NumericalStabilityWarning, match="Metric tensor is zero or numerically singular"):
        model.solve_problem(pulse_accuracy=201, solver_kwargs=solver_kwargs)

    tau_nodes, x_nodes = calls[-1]
    assert tau_nodes[-1] == 1.0 and x_nodes[-1] == 2.0
    assert np.all(np.diff(tau_nodes) > 0)
    pulse = model.control_sol
    assert pulse[0] == -1.0 and pulse[-1] == 2.0
    assert np.all(np.diff(pulse) >= 0)
    # The geodesic passes the zero-metric stretch in zero time, so the pulse still jumps from about 1 to 2 there.
    assert pulse[-2] < 1.0


def test_isolated_zero_of_the_metric_is_rejected_clearly_by_a_custom_solver():
    model = _isolated_zero_metric_model()

    with pytest.warns(NumericalStabilityWarning), pytest.raises(MetricComputationError, match="forward equation"):
        model.solve_problem(solver=functools.partial(solve_ivp, rtol=1e-8))


def test_custom_solver_with_its_own_time_grid_is_resampled_onto_a_uniform_tau():
    # The solver ignores the requested t_eval and returns tau = linspace(0, 1, n)**2. That grid was stored as it was,
    # while Dynamics and synthesize_pulse rebuilt a uniform one from the sample count, which distorted the pulse (D-04).
    from geodesiq import Dynamics

    raw = {}

    def squared_grid_solver(fun, t_span, y0, t_eval, **kwargs):
        sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(0.0, 1.0, len(t_eval)) ** 2, rtol=1e-10, atol=1e-12)
        raw["t"], raw["x"] = sol.t, sol.y[0]
        return sol

    def lz_model() -> ControlModel:
        model = ControlModel(H_d=np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
        model.set_control(pulse_initial=-10.0, pulse_final=10.0, initial_state=0, alpha=2.0, beta=2.0)
        return model

    model = lz_model()
    model.solve_problem(pulse_accuracy=200, solver=squared_grid_solver)
    reference = lz_model()
    reference.solve_problem(pulse_accuracy=200)

    np.testing.assert_array_equal(model.tau, np.linspace(0.0, 1.0, 200))
    np.testing.assert_allclose(model.control_sol, reference.control_sol, rtol=0, atol=0.1)
    # The optimal pulse at a duration is the solver's own samples at their physical times.
    fidelity = Dynamics(model, duration=20.0).state_fidelity()
    exact = Dynamics(model, times=20.0 * raw["t"], pulse=raw["x"], order=3).state_fidelity()
    assert fidelity == pytest.approx(exact, abs=1e-6)


def test_degenerate_gap_raises_clear_metric_error():
    model = ControlModel(degenerate_hamiltonian, degenerate_partial)
    model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=33, )
    with pytest.raises(MetricComputationError, match="Degenerate or near-degenerate"):
        model.solve_problem()


def test_equal_pulse_endpoints_are_rejected():
    model = ControlModel(lz_hamiltonian, lz_partial)
    with pytest.raises(InvalidControlParameterError, match="must be different"):
        model.set_control(control_name="lam", pulse_initial=1.0, pulse_final=1.0)


@pytest.mark.parametrize("state", [-1, cast(Any, 1.5), cast(Any, True)])
def test_invalid_state_indices_are_rejected_early(state: Any):
    model = ControlModel(lz_hamiltonian, lz_partial)
    with pytest.raises(InvalidControlParameterError):
        model.initial_state = state


def test_synthesis_preserves_custom_solver_configuration():
    model = configured_model()
    calls = {"count": 0}

    def custom_solver(fun, t_span, y0, t_eval=None, **kwargs):
        calls["count"] += 1
        return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)

    model.solve_problem(pulse_accuracy=30, solver=custom_solver, solver_kwargs={"rtol": 1e-8, "atol": 1e-10}, )
    model.synthesize_pulse(duration=1.0)

    assert calls["count"] == 1


def test_solver_choice_persists_and_defaults_can_be_restored():
    # Documented behaviour of solve_problem (C-19): a solver or integrator passed once is reused by later calls and by
    # automatic re-solves, until the defaults are passed explicitly.
    model = configured_model()
    calls = {"solver": 0, "integrator": 0}

    def custom_solver(fun, t_span, y0, t_eval=None, **kwargs):
        calls["solver"] += 1
        return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)

    def custom_integrator(values, dx):
        calls["integrator"] += 1
        return romb(values, dx=dx)

    model.solve_problem(pulse_accuracy=30, solver=custom_solver, solver_kwargs={"rtol": 1e-10, "atol": 1e-12},
                        metric_integrator=custom_integrator, )
    model.alpha = 3.0  # invalidates the metric and the pulse
    _ = model.control_sol
    assert calls == {"solver": 2, "integrator": 2}

    model.solve_problem(solver=solve_ivp, solver_kwargs={}, metric_integrator=romb, metric_integrator_kwargs={})
    model.alpha = 2.0
    _ = model.control_sol
    assert calls == {"solver": 2, "integrator": 2}
    assert model._solver is solve_ivp and model._solver_kwargs == {}
    assert model._metric_integrator is romb and model._metric_integrator_kwargs == {}


def affine_lz_model(num_steps: int, alpha: float, beta: float, pulse_initial: float = -5.0,
                    pulse_final: float = 5.0) -> ControlModel:
    model = ControlModel(H_d=0.1 * np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
    model.set_control(control_name="x", pulse_initial=pulse_initial, pulse_final=pulse_final, initial_state=0,
                      alpha=alpha, beta=beta, num_steps=num_steps, )
    return model


def wrapped_solve_ivp(fun, t_span, y0, t_eval=None, **kwargs):
    return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)


@pytest.mark.parametrize(("num_steps", "alpha", "beta"), [(17, 2.0, 2.0), (17, 4.0, 2.0), (65, 2.0, 2.0),
                                                          (1025, 4.0, 2.0), ], )
@pytest.mark.parametrize(("pulse_initial", "pulse_final"), [(-5.0, 5.0), (5.0, -5.0)])
def test_custom_solver_ends_at_pulse_final_and_matches_default(num_steps: int, alpha: float, beta: float,
                                                               pulse_initial: float, pulse_final: float, ) -> None:
    # The forward ODE was normalized with a_tilde from romb instead of the integral of its interpolant, so a custom
    # solver ended far from pulse_final, e.g. at x = 414 or 34801 instead of 5 for num_steps=17 (C-03).
    reference = affine_lz_model(num_steps, alpha, beta, pulse_initial, pulse_final)
    reference.solve_problem(pulse_accuracy=200)

    model = affine_lz_model(num_steps, alpha, beta, pulse_initial, pulse_final)
    model.solve_problem(pulse_accuracy=200, solver=wrapped_solve_ivp, solver_kwargs={"rtol": 1e-10, "atol": 1e-12}, )

    control = model.control_sol
    assert control[0] == pulse_initial
    assert control[-1] == pulse_final
    np.testing.assert_allclose(control, reference.control_sol, atol=1e-4)


def test_partial_solve_ivp_with_default_tolerances_ends_at_pulse_final() -> None:
    model = affine_lz_model(1025, 4.0, 2.0)
    model.solve_problem(pulse_accuracy=200, solver=functools.partial(solve_ivp))

    control = model.control_sol
    assert control[0] == -5.0
    assert control[-1] == 5.0
    assert np.all(np.diff(control) >= 0)
    assert np.all((control >= -5.0) & (control <= 5.0))


def test_custom_solver_that_leaves_the_geodesic_is_rejected() -> None:
    model = affine_lz_model(65, 2.0, 2.0)

    def early_arrival(fun, t_span, y0, t_eval=None, **kwargs):
        # Reaches pulse_final at tau = 0.5 and then stays there: right endpoint, wrong pulse.
        del fun, t_span, kwargs
        assert t_eval is not None
        return t_eval, np.interp(t_eval, [0.0, 0.5, 1.0], [y0[0], 5.0, 5.0])[np.newaxis, :]

    with pytest.raises(SolverError, match="deviates from the geodesic"):
        model.solve_problem(pulse_accuracy=100, solver=early_arrival)


def complex_hamiltonian(x: float) -> np.ndarray:
    # sz + 0.3 sy + x sx + 2 * identity: complex entries and a nonzero trace.
    return np.array([[3.0, x - 0.3j], [x + 0.3j, 1.0]], dtype=complex)


@pytest.mark.parametrize("mode", ["affine", "callable"])
def test_eigenenergies_of_complex_hamiltonian_are_real(mode: str) -> None:
    # The trace offset was kept complex, so eigenenergies came out as complex128 (C-06).
    if mode == "affine":
        model = ControlModel(H_d=complex_hamiltonian(0.0), H_c=complex_hamiltonian(1.0) - complex_hamiltonian(0.0))
    else:
        model = ControlModel(complex_hamiltonian)
    model.set_control(control_name="x", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=33, )

    # numpy.exceptions exists only from numpy 1.25, and numpy 2.0 removed the top-level alias (P-02).
    complex_warning = np.exceptions.ComplexWarning if hasattr(np, "exceptions") else np.ComplexWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", complex_warning)
        energies = model.eigenenergies
        model.solve_problem(pulse_accuracy=50)

    assert energies.dtype == np.float64
    expected = np.array([np.linalg.eigvalsh(complex_hamiltonian(x)) for x in model.control_pulse])
    np.testing.assert_allclose(energies, expected, atol=1e-12)


def test_decreasing_control_sweep_is_supported():
    model = ControlModel(lz_hamiltonian, lz_partial)
    model.set_parameters(delta=0.5)
    model.set_control(control_name="lam", pulse_initial=3.0, pulse_final=-3.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=65, )
    model.solve_problem(pulse_accuracy=80)
    assert np.all(np.diff(model.control_sol) <= 1e-8)


@pytest.mark.parametrize(("status_code", "error_value"), [(-1, 1e-4), (-2, 1e-2), (-3, np.inf), ], )
def test_numerical_derivative_rejects_failed_jacobian(monkeypatch, status_code: int, error_value: float, ) -> None:
    def hamiltonian(lam: float) -> np.ndarray:
        return np.array([[lam, 1.0], [1.0, -lam], ], dtype=float, )

    model = ControlModel(hamiltonian)

    model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                      num_steps=5, )

    def failed_jacobian(func, x, **kwargs):
        del func, kwargs

        n_points = x.shape[-1]

        # 2x2 Hamiltonian -> 4 complex elements
        # packed as 4 real + 4 imaginary = 8 outputs
        shape = (8, 1, n_points)

        df = np.zeros(shape, dtype=float)
        success = np.ones(shape, dtype=bool)
        status = np.zeros(shape, dtype=int)
        error = np.zeros(shape, dtype=float)

        # Make the middle control point fail.
        success[..., 2] = False
        status[..., 2] = status_code
        error[..., 2] = error_value

        return SimpleNamespace(df=df, success=success, status=status, error=error, )

    monkeypatch.setattr("geodesiq.controlmodel.jacobian", failed_jacobian, )
    # Grids smaller than _FD_MIN_POINTS skip the finite-difference stage and use the adaptive jacobian everywhere.
    monkeypatch.setattr(ControlModel, "_FD_MIN_POINTS", 100)

    with pytest.raises(SolverError) as exc_info:
        model._solve_eigenproblem()

    assert "Numerical differentiation failed to converge" in str(exc_info.value)
    assert str(status_code) in str(exc_info.value)


@pytest.mark.parametrize(("alpha", "beta"), [(2, 4), (10 / 3, 4), (4, 4), ], )
def test_dqd_model_singularity(alpha: float, beta: float, ) -> None:
    dqd_model = ControlModel(dqd_hamiltonian)

    U, tc, Ez, dEz, dEx = 10, 1, .9, .1, .01
    eps0, epsf = 15, 0

    dqd_model.set_parameters(U=U, tc=tc, Ez=Ez, dEz=dEz, dEx=dEx)
    dqd_model.set_control(control_name='eps', pulse_initial=eps0, pulse_final=epsf, initial_state=0)

    dqd_model.set_control(alpha=alpha, beta=beta)
    dqd_model.solve_problem(pulse_accuracy=100)


# ---------------------------------------------------------------------------
# Degeneracies
# ---------------------------------------------------------------------------
class TestDegeneracies:
    def test_global_energy_shift_does_not_create_degeneracy(self):
        model = ControlModel(shifted_hamiltonian, shifted_partial, )

        model.set_parameters(shift=1e10)

        model.set_control(control_name="lam", pulse_initial=-2.0, pulse_final=2.0, initial_state=0, alpha=2.0, beta=2.0,
                          num_steps=65, )

        model.solve_problem(pulse_accuracy=50)

        assert np.all(np.isfinite(model.control_sol))

    def test_global_energy_shift_leaves_control_solution_unchanged(self):
        model_0 = make_model(0.0)
        model_small_shifted = make_model(10)
        model_large_shifted = make_model(1e8)

        np.testing.assert_allclose(model_small_shifted.control_sol, model_0.control_sol, rtol=1e-7, atol=1e-9, )
        np.testing.assert_allclose(model_large_shifted.control_sol, model_0.control_sol, rtol=1e-7, atol=1e-9, )

    def test_true_degeneracy_is_rejected(self):
        model = ControlModel(degenerate_hamiltonian, degenerate_partial, )

        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
                          num_steps=65, )

        with pytest.raises(MetricComputationError, match="Degenerate or near-degenerate", ):
            model.solve_problem()

    def test_physical_energies(self):
        model_0 = make_model(0.0)
        model_small_shifted = make_model(10)

        np.testing.assert_allclose(model_0.eigenenergies + 10, model_small_shifted.eigenenergies)


# ---------------------------------------------------------------------------
# True (symmetry-protected) level crossings (C-07)
# ---------------------------------------------------------------------------
LZ_BLOCK_D = np.array([[0.0, 0.1], [0.1, 0.0]])
LZ_BLOCK_C = np.diag([1.0, -1.0])


def lz_with_uncoupled_level(level: float, slope: float) -> ControlModel:
    """Landau-Zener block plus a third level ``level + slope * x`` that the control does not couple to it."""
    h_d = np.zeros((3, 3))
    h_d[:2, :2] = LZ_BLOCK_D
    h_d[2, 2] = level
    return ControlModel(H_d=h_d, H_c=np.diag([1.0, -1.0, slope]))


def metric_of(model: ControlModel) -> np.ndarray:
    assert model._metric_tensor is not None
    return model._metric_tensor


def lz_block_model(pulse_initial: float, pulse_final: float) -> ControlModel:
    model = ControlModel(H_d=LZ_BLOCK_D, H_c=LZ_BLOCK_C)
    model.set_control(pulse_initial=pulse_initial, pulse_final=pulse_final, initial_state=0, alpha=2.0, beta=2.0)
    model.solve_problem()
    return model


class TestTrueCrossings:
    @pytest.mark.filterwarnings("error::geodesiq.warnings.NumericalStabilityWarning")
    def test_occupied_state_is_followed_through_crossings_with_an_uncoupled_level(self):
        # Review reproduction: the uncoupled level -0.5 is the lowest one for |x| < 0.49. Energy-ordered labels made
        # state 0 that level there, so the metric vanished over the whole interval.
        model = lz_with_uncoupled_level(-0.5, 0.0)
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0)
        model.solve_problem()

        x = model.control_pulse
        crossing = np.sqrt(0.25 - 0.01)
        assert model._occupied_states is not None
        np.testing.assert_array_equal(model._occupied_states, np.where(np.abs(x) < crossing, 1, 0))
        assert model.final_state == 0
        reference = lz_block_model(-1.0, 1.0)
        np.testing.assert_allclose(metric_of(model), metric_of(reference), rtol=1e-12)
        np.testing.assert_allclose(model.control_sol, reference.control_sol, rtol=1e-12, atol=1e-12)

    def test_implicit_final_state_is_the_index_the_followed_state_ends_in(self):
        from geodesiq import Dynamics

        model = lz_with_uncoupled_level(-0.5, 0.0)
        model.set_control(pulse_initial=-1.0, pulse_final=0.0, initial_state=0, alpha=2.0, beta=2.0)
        assert model.final_state == 0

        model.solve_problem()

        assert model.final_state == 1
        assert model._dia_list is None
        # The default target of the dynamics is the followed state, not the uncoupled ground state at x = 0.
        assert Dynamics(duration=200.0, model=model).state_fidelity() > 0.99
        # Changing the initial state releases the resolved value until the next solve.
        model.initial_state = 2
        assert model.final_state == 2

    def test_explicit_final_state_at_the_end_of_the_followed_state_is_adiabatic(self):
        model = lz_with_uncoupled_level(-0.5, 0.0)
        model.set_control(pulse_initial=-1.0, pulse_final=0.0, initial_state=0, final_state=1, alpha=2.0, beta=2.0)

        model.solve_problem()  # no dia_alpha/dia_beta needed

        assert model._dia_list is None
        np.testing.assert_allclose(model.control_sol, lz_block_model(-1.0, 0.0).control_sol, rtol=1e-12, atol=1e-12)

    def test_explicit_final_state_elsewhere_needs_the_diabatic_exponents(self):
        model = lz_with_uncoupled_level(-0.5, 0.0)
        model.set_control(pulse_initial=-1.0, pulse_final=0.0, initial_state=0, alpha=2.0, beta=2.0)
        model.solve_problem()
        flags = dict(model._flags._values)

        # Making the resolved value explicit keeps the cached results; a different value does not.
        model.final_state = 1
        assert dict(model._flags._values) == flags
        model.final_state = 2
        assert not model._flags["dia_list_computed"]
        with pytest.raises(MissingControlParameterError, match="ends in state 1"):
            model.solve_problem()

    @pytest.mark.filterwarnings("error::geodesiq.warnings.NumericalStabilityWarning")
    def test_crossing_on_a_grid_point_is_followed(self):
        # The lower LZ level -sqrt(x^2 + 0.01) crosses the uncoupled level -0.1 - x at x = 0, a grid point, where the
        # two states are degenerate.
        model = lz_with_uncoupled_level(-0.1, -1.0)
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0)
        model.solve_problem()

        x = model.control_pulse
        assert np.min(np.diff(model.eigenenergies[x == 0.0])) < 1e-15
        assert model._occupied_states is not None
        np.testing.assert_array_equal(model._occupied_states, np.where(x < 0.0, 0, 1))
        assert model.final_state == 1
        reference = lz_block_model(-1.0, 1.0)
        np.testing.assert_allclose(metric_of(model), metric_of(reference), rtol=1e-12)

    @pytest.mark.parametrize("partial", [True, False])
    def test_crossing_is_followed_for_callable_models(self, partial: bool):
        def hamiltonian(x: float) -> np.ndarray:
            return np.diag([0.0, 0.0, -0.5]) + x * np.diag([1.0, -1.0, 0.0]) + np.pad(LZ_BLOCK_D, (0, 1))

        model = ControlModel(hamiltonian, (lambda x: np.diag([1.0, -1.0, 0.0])) if partial else None)
        model.set_control(control_name="x", pulse_initial=-1.0, pulse_final=0.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=257)
        model.solve_problem()

        assert model.final_state == 1
        reference = ControlModel(H_d=LZ_BLOCK_D, H_c=LZ_BLOCK_C)
        reference.set_control(pulse_initial=-1.0, pulse_final=0.0, initial_state=0, alpha=2.0, beta=2.0, num_steps=257)
        reference.solve_problem()
        np.testing.assert_allclose(metric_of(model), metric_of(reference), rtol=1e-6)

    @pytest.mark.filterwarnings("ignore::geodesiq.warnings.NumericalStabilityWarning")
    def test_unresolved_anticrossing_keeps_the_energy_order(self):
        # The eigenvectors swap between two grid points as at a crossing, but dH/dx couples the two states.
        model = ControlModel(H_d=np.array([[0.0, 1e-6], [1e-6, 0.0]]), H_c=LZ_BLOCK_C)
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0, num_steps=33)
        model.solve_problem()

        assert model._occupied_states is not None
        assert np.all(model._occupied_states == 0)
        assert model.final_state == 0

    def test_diabatic_transfer_through_a_crossing(self):
        # Level a = x anticrosses c = 0.5 at x = 0.5 (coupling 0.05) and crosses the uncoupled level b = -0.5 at
        # x = -0.5. From a (state 0), the crossing is followed and only the anticrossing is passed diabatically.
        model = ControlModel(H_d=np.array([[0.0, 0.0, 0.05], [0.0, -0.5, 0.0], [0.05, 0.0, 0.5]]),
                             H_c=np.diag([1.0, 0.0, 0.0]))
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, final_state=2, alpha=2.0, beta=2.0,
                          dia_alpha=-1.0, dia_beta=0.0)
        model.solve_problem()

        x = model.control_pulse
        dia_list = model._dia_list
        assert dia_list is not None
        # The level repulsion from c moves the crossing slightly below x = -0.5.
        changes = np.flatnonzero(np.diff(dia_list.occupied)) + 1
        assert dia_list.occupied[0] == 0 and dia_list.occupied[changes].tolist() == [1, 2]
        assert abs(x[changes[0]] + 0.5) < 0.01 and abs(x[changes[1]] - 0.5) < 0.01
        # The transition to the crossing level is adiabatic; only the anticrossing is diabatic.
        near_crossing = np.abs(x + 0.5) < 0.1
        assert np.all(dia_list.dense()[near_crossing, 0, 1] == 1)
        assert np.all(dia_list.dense()[np.abs(x - 0.5) < 0.05, 1, 2] == 0)


# ---------------------------------------------------------------------------
# Diabatic passages at resolved interior anticrossings only (C-01)
# ---------------------------------------------------------------------------
SIGMA_Z = np.diag([1.0, -1.0])
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0


def rising_gap_model(pulse_initial: float, pulse_final: float) -> ControlModel:
    """Review reproduction R1: the gap rises on [0, 0.5], and the anticrossing is at x = (1 + sqrt(5)) / 2."""
    model = ControlModel(lambda x: (1.0 + x - x * x) * SIGMA_Z + 0.05 * SIGMA_X,
                         partial_H_func=lambda x: (1.0 - 2.0 * x) * SIGMA_Z)
    model.set_control(control_name="x", pulse_initial=pulse_initial, pulse_final=pulse_final, initial_state=0,
                      final_state=1, alpha=2.0, beta=2.0, dia_alpha=0.0, dia_beta=0.0)
    return model


def rotation(x: float) -> np.ndarray:
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


def flat_noisy_gap_model(pulse_initial: float, pulse_final: float) -> ControlModel:
    """
    Review reproduction R14: in a basis rotating with x, the gap is constant for x < 1, so its samples differ only by
    rounding errors, and the anticrossing is at x = 2.
    """

    def h(x: float) -> float:
        return 1.0 if x < 1.0 else 2.0 - x

    def hamiltonian(x: float) -> np.ndarray:
        return rotation(x) @ (h(x) * SIGMA_Z + 0.05 * SIGMA_X) @ rotation(x).T

    def partial(x: float) -> np.ndarray:
        drift = h(x) * SIGMA_Z + 0.05 * SIGMA_X
        d_rotation = np.array([[-np.sin(x), -np.cos(x)], [np.cos(x), -np.sin(x)]])
        return (d_rotation @ drift @ rotation(x).T + rotation(x) @ drift @ d_rotation.T
                + (0.0 if x < 1.0 else -1.0) * rotation(x) @ SIGMA_Z @ rotation(x).T)

    model = ControlModel(hamiltonian, partial_H_func=partial)
    model.set_control(control_name="x", pulse_initial=pulse_initial, pulse_final=pulse_final, initial_state=0,
                      final_state=1, alpha=2.0, beta=2.0, dia_alpha=0.0, dia_beta=0.0)
    return model


def passage_points(model: ControlModel) -> np.ndarray:
    model.solve_problem()
    assert model._dia_list is not None
    return model.control_pulse[np.flatnonzero(np.diff(model._dia_list.occupied)) + 1]


class TestResolvedAnticrossings:
    @pytest.mark.parametrize("sweep", [(0.0, 2.5), (2.5, 0.0)])
    def test_rising_gap_is_passed_at_the_anticrossing(self, sweep):
        # R1: the passage used to be made at x = 0, and the anticrossing was then passed with the adiabatic exponents.
        model = rising_gap_model(*sweep)
        np.testing.assert_allclose(passage_points(model), [GOLDEN], atol=0.01)

        x = model.control_pulse
        assert model._dia_list is not None
        fraction = model._dia_list.adiabatic_fraction
        assert np.all(fraction[np.abs(x - GOLDEN) < 0.1, 1 - model._dia_list.occupied[np.abs(x - GOLDEN) < 0.1]] == 0)

    @pytest.mark.parametrize("sweep", [(0.0, 3.0), (3.0, 0.0)])
    def test_rounding_dips_of_a_flat_gap_are_not_anticrossings(self, sweep):
        model = flat_noisy_gap_model(*sweep)
        x = model.control_pulse
        gap = np.diff(model.eigenenergies, axis=1)[:, 0]
        # The precondition: the flat part has local minima at the level of rounding errors.
        flat = x < 0.95
        assert np.ptp(gap[flat]) < 1e-13
        assert np.any(_gap_minima(gap)[flat])

        np.testing.assert_allclose(passage_points(model), [2.0], atol=0.01)
        # The passage acts only within its anticrossing, which the flat gap bounds where it begins.
        assert model._dia_list is not None
        transitions = 1 - model._dia_list.occupied
        np.testing.assert_array_equal(model._dia_list.adiabatic_fraction[flat, transitions[flat]], 1.0)


# ---------------------------------------------------------------------------
# Transitions that dH/dx does not couple (A-12)
# ---------------------------------------------------------------------------
def rotated(model: ControlModel) -> ControlModel:
    """The same model in a rotated basis, where the uncoupled matrix elements are rounding errors, not exact zeros."""
    rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    assert model._H_d is not None and model._H_c is not None
    return ControlModel(H_d=rotation @ model._H_d @ rotation.T, H_c=rotation @ model._H_c @ rotation.T)


class TestUncoupledTransitions:
    @pytest.mark.parametrize("basis", [lambda model: model, rotated])
    def test_beta_zero_keeps_uncoupled_transitions(self, basis):
        # 0**0 = 1: the uncoupled level 5 contributes 1/|E_2 - E_0|^alpha.
        model = basis(lz_with_uncoupled_level(5.0, 0.0))
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=0.0)
        model.solve_problem()

        gaps = np.abs(model.eigenenergies[:, 1:] - model.eigenenergies[:, :1])
        np.testing.assert_allclose(metric_of(model), np.sum(gaps ** -2.0, axis=1), rtol=1e-10)

    def test_zero_exponents_give_a_linear_ramp(self):
        model = lz_with_uncoupled_level(5.0, 0.0)
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=0.0, beta=0.0)
        model.solve_problem()

        np.testing.assert_array_equal(metric_of(model), 2.0)
        np.testing.assert_allclose(model.control_sol, -1.0 + 2.0 * model.tau, atol=1e-10)

    @pytest.mark.parametrize("basis", [lambda model: model, rotated])
    def test_negative_beta_drops_uncoupled_transitions(self, basis):
        # Before, an exact zero raised MetricComputationError (inf), and a rounding error gave a metric of ~1e16.
        model = basis(lz_with_uncoupled_level(5.0, 0.0))
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=-1.0)
        model.solve_problem()

        reference = ControlModel(H_d=LZ_BLOCK_D, H_c=LZ_BLOCK_C)
        reference.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=-1.0)
        reference.solve_problem()
        np.testing.assert_allclose(metric_of(model), metric_of(reference), rtol=1e-10)

    def test_negative_beta_at_a_crossing_on_a_grid_point(self):
        model = lz_with_uncoupled_level(-0.1, -1.0)
        model.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=-1.0)
        model.solve_problem()

        reference = ControlModel(H_d=LZ_BLOCK_D, H_c=LZ_BLOCK_C)
        reference.set_control(pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=-1.0)
        reference.solve_problem()
        np.testing.assert_allclose(metric_of(model), metric_of(reference), rtol=1e-12)


# ---------------------------------------------------------------------------
# Unit independence of the numerical tolerances
# ---------------------------------------------------------------------------
def scaled_lz_hamiltonian(lam: float, scale: float, delta: float = 0.5) -> np.ndarray:
    """Landau-Zener Hamiltonian H = scale * (lam sigma_z + delta sigma_x), e.g. scale = h * 1 GHz in Joules."""
    return scale * lz_hamiltonian(lam, delta=delta)


def scaled_lz_partial(lam: float, scale: float, delta: float = 0.5) -> np.ndarray:
    return scale * lz_partial(lam, delta=delta)


def scaled_model(scale: float, *, analytical: bool = True, initial_state: int = 0, final_state: int = 0,
                 ) -> ControlModel:
    model = ControlModel(scaled_lz_hamiltonian, scaled_lz_partial if analytical else None)
    model.set_parameters(scale=scale, delta=0.5)
    model.set_control(control_name="lam", pulse_initial=-3.0, pulse_final=3.0, initial_state=initial_state,
                      final_state=final_state, alpha=2.0, beta=2.0, dia_alpha=2.0, dia_beta=2.0, num_steps=65, )
    model.solve_problem(pulse_accuracy=50)
    return model


class TestUnitIndependence:
    """The optimal pulse only depends on the shape of the spectrum, not on the units of the Hamiltonian."""

    @pytest.mark.parametrize("scale", [6.62607015e-25, 1e-12, 1e9])
    @pytest.mark.parametrize("analytical", [True, False])
    def test_control_solution_does_not_depend_on_energy_units(self, scale: float, analytical: bool):
        reference = scaled_model(1.0, analytical=analytical)
        scaled = scaled_model(scale, analytical=analytical)

        np.testing.assert_allclose(scaled.control_sol, reference.control_sol, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("scale", [6.62607015e-25, 1e9])
    def test_diabatic_solution_does_not_depend_on_energy_units(self, scale: float):
        reference = scaled_model(1.0, initial_state=0, final_state=1)
        scaled = scaled_model(scale, initial_state=0, final_state=1)

        np.testing.assert_allclose(scaled.control_sol, reference.control_sol, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("scale", [6.62607015e-25, 1.0, 1e9])
    def test_true_degeneracy_is_rejected_in_any_units(self, scale: float):
        model = ControlModel(lambda lam: scale * degenerate_hamiltonian(lam),
                             lambda lam: scale * degenerate_partial(lam))
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=65, )

        with pytest.raises(MetricComputationError, match="Degenerate or near-degenerate"):
            model.solve_problem()

    @pytest.mark.filterwarnings("ignore::geodesiq.warnings.NumericalStabilityWarning")
    def test_small_but_resolved_gap_is_not_a_degeneracy(self):
        """A minimum gap of ~1e-7 relative to the bandwidth is physical and must not be flagged as degenerate."""
        delta = 1e-6

        def hamiltonian(lam: float) -> np.ndarray:
            return np.array([[lam, delta, 0.0], [delta, -lam, 0.0], [0.0, 0.0, 10.0]])

        def partial(lam: float) -> np.ndarray:
            return np.diag([1.0, -1.0, 0.0])

        model = ControlModel(hamiltonian, partial)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=65, )

        model.solve_problem(pulse_accuracy=50)

        gaps = np.diff(np.sort(model.eigenenergies, axis=1), axis=1)
        assert gaps.min() / np.ptp(model.eigenenergies, axis=1).max() < 1e-6
        assert np.all(np.isfinite(model.control_sol))


# ---------------------------------------------------------------------------
# Hermiticity checks
# ---------------------------------------------------------------------------
class TestHermiticity:
    def test_non_hermitian_matrix_is_rejected_at_small_scales(self):
        """At 1e-12 energy scales the old absolute tolerance accepted any matrix."""

        def hamiltonian(lam: float) -> np.ndarray:
            return 1e-12 * np.array([[lam, 1.0], [0.0, -lam]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=33, )

        with pytest.raises(ValidationError, match="Hermitian"):
            model.evaluate_hamiltonian(0.5)

    def test_rounding_level_asymmetry_is_accepted_at_large_scales(self):
        def hamiltonian(lam: float) -> np.ndarray:
            return 1e10 * np.array([[lam, 1.0 + 1e-15], [1.0, -lam]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=33, )

        np.testing.assert_allclose(model.evaluate_hamiltonian(0.5), hamiltonian(0.5))

    def test_non_finite_matrix_reports_finiteness_not_hermiticity(self):
        model = ControlModel(lambda lam: np.array([[lam, np.nan], [np.nan, -lam]]))
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0)

        with pytest.raises(ValidationError, match="finite values"):
            model.evaluate_hamiltonian(0.5)

    def test_affine_matrices_use_relative_check(self):
        H_c = np.diag([1.0, -1.0]) * 1e-12
        non_hermitian_drift = np.array([[0.0, 1.0], [0.0, 0.0]]) * 1e-12

        with pytest.raises(ValidationError, match="H_d must be Hermitian"):
            ControlModel(H_d=non_hermitian_drift, H_c=H_c)

    @pytest.mark.parametrize("offset", [1e4, 1e8, 1e12])
    def test_identity_offset_does_not_hide_non_hermitian_part(self, offset: float):
        # The check was relative to ||A||_F, which a large identity offset dominates, so this matrix was accepted and
        # eigh silently used only one triangle of it (C-08).
        non_hermitian = offset * np.eye(2) + np.array([[0.0, 1.0], [0.5, 0.0]])

        assert not is_hermitian(non_hermitian)
        with pytest.raises(ValidationError, match="H_d must be Hermitian"):
            ControlModel(H_d=non_hermitian, H_c=np.diag([1.0, -1.0]))

        model = ControlModel(lambda lam: non_hermitian + lam * np.diag([1.0, -1.0]))
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0)
        with pytest.raises(ValidationError, match="Hermitian"):
            model.evaluate_hamiltonian(0.5)

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e8, 1e12])
    @pytest.mark.parametrize("dimension", [2, 16])
    def test_rotated_hermitian_matrix_with_large_offset_is_accepted(self, offset: float, dimension: int):
        # U (offset * I + h) U^dagger carries rounding errors of order eps * offset; they must not be rejected.
        rng = np.random.default_rng(1)
        shape = (dimension, dimension)
        x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        h = (x + x.conj().T) / 2
        unitary, _ = np.linalg.qr(rng.normal(size=shape) + 1j * rng.normal(size=shape))

        assert is_hermitian(unitary @ (offset * np.eye(dimension) + h) @ unitary.conj().T)

    def test_multiples_of_identity(self):
        assert is_hermitian(np.zeros((3, 3)))
        assert is_hermitian(5.0 * np.eye(3))
        assert not is_hermitian((5.0 + 1.0j) * np.eye(3))


# ---------------------------------------------------------------------------
# Re-solving keeps the pulse accuracy
# ---------------------------------------------------------------------------
class TestPulseAccuracyIsReused:
    def test_control_sol_resolves_with_previous_accuracy(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=50)

        model.set_control(alpha=3.0)

        assert model.control_sol.shape == (50,)
        assert model.tau.shape == (50,)

    def test_synthesize_pulse_resolves_with_previous_accuracy(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=40)

        model.set_parameters(delta=0.7)

        assert model.synthesize_pulse(duration=2.0).pulse.shape == (40,)

    def test_explicit_accuracy_still_wins(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=40)

        model.solve_problem(pulse_accuracy=60)

        assert model.control_sol.shape == (60,)

    def test_default_accuracy_on_first_solve(self):
        model = configured_model()

        model.solve_problem()

        assert model.control_sol.shape == (ControlModel._DEFAULT_PULSE_ACCURACY,)

    def test_tau_property_is_a_copy_of_the_normalized_grid(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        tau = model.tau
        tau[0] = 5.0

        assert model.tau[0] == 0.0
        assert model.tau[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Setters share the logic of set_control
# ---------------------------------------------------------------------------
class TestSettersDelegateToSetControl:
    def test_setters_and_set_control_produce_the_same_state(self):
        by_setters = ControlModel(lz_hamiltonian, lz_partial)
        by_setters.control_name = "lam"
        by_setters.pulse_initial = -3.0
        by_setters.pulse_final = 3.0
        by_setters.initial_state = 0
        by_setters.alpha = 2.0
        by_setters.beta = 2.0

        by_set_control = ControlModel(lz_hamiltonian, lz_partial)
        by_set_control.set_control(control_name="lam", pulse_initial=-3.0, pulse_final=3.0, initial_state=0,
                                   alpha=2.0, beta=2.0, )

        for name in ("control_name", "pulse_initial", "pulse_final", "initial_state", "final_state", "alpha", "beta",
                     "num_steps"):
            assert getattr(by_setters, name) == getattr(by_set_control, name), name

    def test_setter_change_invalidates_the_solution(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=30)
        first = model.control_sol

        model.alpha = 4.0

        assert not model._flags["metric_computed"]
        assert not np.allclose(model.control_sol, first)

    def test_invalid_setter_value_leaves_state_unchanged(self):
        model = configured_model()

        with pytest.raises(InvalidControlParameterError, match="must be different"):
            model.pulse_final = model.pulse_initial

        assert model.pulse_final == 3.0

    def test_none_keeps_previous_value(self):
        model = configured_model()

        model.alpha = None

        assert model.alpha == 2.0


# ---------------------------------------------------------------------------
# Plotting after reconfiguration
# ---------------------------------------------------------------------------
class TestPlotEigenvaluesAfterReconfiguration:
    def test_first_branch_uses_the_new_grid(self):
        import matplotlib.pyplot as plt

        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        model.set_control(pulse_initial=-1.0)
        fig, ax = model.plot_eigenvalues()

        expected_grid = np.linspace(-1.0, 3.0, 65)
        for line in ax.get_lines():
            np.testing.assert_allclose(np.asarray(line.get_xdata()), expected_grid)
        plt.close(cast(Any, fig))

    def test_changing_num_steps_does_not_break_plot(self):
        import matplotlib.pyplot as plt

        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        model.set_control(num_steps=33)
        fig, ax = model.plot_eigenvalues()

        assert all(np.size(line.get_xdata()) == 33 for line in ax.get_lines())
        plt.close(cast(Any, fig))


# ---------------------------------------------------------------------------
# Numerical derivative on the sampled grid
# ---------------------------------------------------------------------------
class TestGridNumericalDerivative:
    def test_smooth_hamiltonian_needs_no_extra_evaluations(self):
        calls = []

        def hamiltonian(lam: float) -> np.ndarray:
            calls.append(lam)
            return np.array([[np.sin(lam), 0.5], [0.5, -np.sin(lam)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=129, )

        model._solve_eigenproblem()

        assert len(calls) == 129

    def test_smooth_derivative_is_accurate(self, monkeypatch):
        def hamiltonian(lam: float) -> np.ndarray:
            return np.array([[np.sin(3 * lam), 1j * lam ** 3], [-1j * lam ** 3, np.exp(lam)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=129, )
        model._solve_eigenproblem()

        def no_jacobian(*args, **kwargs):
            raise AssertionError("The adaptive fallback should not be needed for a smooth Hamiltonian.")

        monkeypatch.setattr("geodesiq.controlmodel.jacobian", no_jacobian)
        derivative = model._compute_numerical_partial_H()

        # The derivative of the traceless part of H: the trace part only adds a multiple of the identity (C-09).
        x = model.control_pulse
        half_difference = (3 * np.cos(3 * x) - np.exp(x)) / 2
        expected = np.zeros((x.size, 2, 2), dtype=complex)
        expected[:, 0, 0] = half_difference
        expected[:, 0, 1] = 3j * x ** 2
        expected[:, 1, 0] = -3j * x ** 2
        expected[:, 1, 1] = -half_difference
        np.testing.assert_allclose(derivative, expected, atol=1e-9)

    def test_kink_falls_back_to_adaptive_jacobian_only_near_the_kink(self, monkeypatch):
        kink = 0.0123

        def hamiltonian(lam: float) -> np.ndarray:
            return np.array([[abs(lam - kink), 1.0], [1.0, -abs(lam - kink)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=65, )
        model._solve_eigenproblem()

        import geodesiq.controlmodel as controlmodel_module

        fallback_points = []
        original_jacobian = controlmodel_module.jacobian

        def spy_jacobian(func, x, **kwargs):
            fallback_points.extend(np.ravel(x).tolist())
            return original_jacobian(func, x, **kwargs)

        monkeypatch.setattr(controlmodel_module, "jacobian", spy_jacobian)
        derivative = model._compute_numerical_partial_H()

        x = model.control_pulse
        assert 0 < len(fallback_points) < x.size // 2
        assert all(abs(point - kink) < 0.3 for point in fallback_points)
        np.testing.assert_allclose(derivative[:, 0, 0].real, np.sign(x - kink), atol=1e-6)

    @pytest.mark.parametrize("offset", [1e4, 1e6, 1e9])
    def test_large_control_offset_uses_grid_derivative(self, monkeypatch, offset: float):
        # Rounding of x on a linspace with a large offset made the uniformity check fail, and all points went to
        # scipy.differentiate.jacobian, which then raised (C-02).
        sz = np.diag([1.0, -1.0])
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = ControlModel(lambda x: (x - offset) * sz + 0.1 * sx)
        model.set_control(control_name="x", pulse_initial=offset - 0.3, pulse_final=offset + 0.3, initial_state=0,
                          alpha=2.0, beta=2.0, )
        model._solve_eigenproblem()

        def no_jacobian(*args, **kwargs):
            raise AssertionError("A Hamiltonian linear in x should not need the adaptive fallback.")

        monkeypatch.setattr("geodesiq.controlmodel.jacobian", no_jacobian)
        derivative = model._compute_numerical_partial_H()

        # Accuracy is limited by the rounding of x relative to the step: eps * offset / step.
        step = 0.6 / (model.control_pulse.size - 1)
        roundoff = np.finfo(float).eps * offset / step
        np.testing.assert_allclose(derivative, np.broadcast_to(sz, derivative.shape), atol=max(1e-9, 1e2 * roundoff))

    def test_large_control_offset_solves_problem(self):
        offset = 1e6
        sz = np.diag([1.0, -1.0])
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = ControlModel(lambda x: (x - offset) * sz + 0.1 * sx)
        model.set_control(control_name="x", pulse_initial=offset - 0.3, pulse_final=offset + 0.3, initial_state=0,
                          alpha=2.0, beta=2.0, )
        model.solve_problem(pulse_accuracy=100)

        reference = ControlModel(H_d=0.1 * sx, H_c=sz)
        reference.set_control(control_name="x", pulse_initial=-0.3, pulse_final=0.3, initial_state=0, alpha=2.0,
                              beta=2.0, )
        reference.solve_problem(pulse_accuracy=100)

        np.testing.assert_allclose(model.control_sol - offset, reference.control_sol, atol=1e-6)

    def test_kink_with_large_offset_converges_in_adaptive_jacobian(self):
        offset = 1e6
        kink = offset + 0.0123

        def hamiltonian(lam: float) -> np.ndarray:
            return np.array([[abs(lam - kink), 1.0], [1.0, -abs(lam - kink)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=offset - 1.0, pulse_final=offset + 1.0, initial_state=0,
                          alpha=2.0, beta=2.0, num_steps=65, )
        model._solve_eigenproblem()

        with warnings.catch_warnings():
            warnings.simplefilter("error", NumericalStabilityWarning)
            derivative = model._compute_numerical_partial_H()

        x = model.control_pulse
        np.testing.assert_allclose(derivative[:, 0, 0].real, np.sign(x - kink), atol=1e-4)

    @pytest.mark.parametrize("offset", [1e6, 1e8, 1e10])
    def test_large_identity_offset_needs_no_extra_evaluations(self, monkeypatch, offset: float):
        # The rounding error of offset * I in the samples dominated the error estimate of the full H, and most points
        # went to scipy.differentiate.jacobian, which cannot resolve it either (C-09). The step 2.01 / 256 is not a
        # power of two, so the rounding error differs from sample to sample.
        sz = np.diag([1.0, -1.0])
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        calls = []

        def hamiltonian(x: float, offset: float) -> np.ndarray:
            calls.append(x)
            return offset * np.eye(2) + x * sz + 0.1 * sx

        def no_jacobian(*args, **kwargs):
            raise AssertionError("An identity offset should not need the adaptive fallback.")

        def solved(model: ControlModel) -> ControlModel:
            model.set_control(control_name="x", pulse_initial=-1.3, pulse_final=0.71, initial_state=0, alpha=2.0,
                              beta=2.0, num_steps=257, )
            model.set_parameters(offset=offset)
            model.solve_problem(pulse_accuracy=100)
            return model

        monkeypatch.setattr("geodesiq.controlmodel.jacobian", no_jacobian)
        model = solved(ControlModel(hamiltonian))
        # The 257 grid samples plus one evaluation that validates the configuration.
        assert len(calls) <= 257 + 1

        exact = solved(ControlModel(hamiltonian, lambda x, offset: sz))
        # The samples carry an absolute rounding error of about eps * offset, which the grid derivative amplifies by
        # 1 / step; the metric (~ |dH|^2 / gap^2 with gap >= 0.2) inherits it relative to |dH| = 1.
        step = 2.01 / 256
        bound = max(1e-9, 1e3 * np.finfo(float).eps * offset / step)
        assert model._metric_tensor is not None and exact._metric_tensor is not None
        np.testing.assert_allclose(model._metric_tensor, exact._metric_tensor, rtol=bound)

    def test_kink_with_large_identity_offset_uses_jacobian_near_the_kink(self, monkeypatch):
        offset = 1e8
        kink = 0.0123

        def hamiltonian(lam: float) -> np.ndarray:
            return offset * np.eye(2) + np.array([[abs(lam - kink), 1.0], [1.0, -abs(lam - kink)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=65, )
        model._solve_eigenproblem()

        import geodesiq.controlmodel as controlmodel_module

        fallback_points = []
        original_jacobian = controlmodel_module.jacobian

        def spy_jacobian(func, x, **kwargs):
            fallback_points.extend(np.ravel(x).tolist())
            return original_jacobian(func, x, **kwargs)

        monkeypatch.setattr(controlmodel_module, "jacobian", spy_jacobian)
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumericalStabilityWarning)
            derivative = model._compute_numerical_partial_H()

        x = model.control_pulse
        assert 0 < len(fallback_points) < x.size // 2
        assert all(abs(point - kink) < 0.3 for point in fallback_points)
        np.testing.assert_allclose(derivative[:, 0, 0].real, np.sign(x - kink), atol=1e-4)

    def test_failed_jacobian_falls_back_to_grid_derivative_with_warning(self, monkeypatch):
        kink = 0.0123

        def hamiltonian(lam: float) -> np.ndarray:
            return np.array([[abs(lam - kink), 1.0], [1.0, -abs(lam - kink)]])

        model = ControlModel(hamiltonian)
        model.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0,
                          beta=2.0, num_steps=65, )
        model._solve_eigenproblem()

        def failed_jacobian(func, x, **kwargs):
            del func, kwargs
            shape = (8, 1, x.shape[-1])
            return SimpleNamespace(df=np.full(shape, np.nan), success=np.zeros(shape, dtype=bool),
                                   status=np.full(shape, -2), error=np.full(shape, 1e-2), )

        monkeypatch.setattr("geodesiq.controlmodel.jacobian", failed_jacobian)
        with pytest.warns(NumericalStabilityWarning, match="finite-difference derivative"):
            derivative = model._compute_numerical_partial_H()

        x = model.control_pulse
        assert np.all(np.isfinite(derivative))
        # Away from the kink, the grid derivative is exact for the piecewise-linear Hamiltonian.
        far = np.abs(x - kink) > 0.3
        np.testing.assert_allclose(derivative[far, 0, 0].real, np.sign(x[far] - kink), atol=1e-8)


# ---------------------------------------------------------------------------
# Compact diabatic list (C-16)
# ---------------------------------------------------------------------------
def test_vectorized_diabatic_metric_matches_the_sum_over_transitions():
    # A chain of four levels with neighbouring anticrossings; the transfer 1 -> 3 passes two of them diabatically,
    # with blending towards the gap maxima.
    dim = 4
    hopping = 0.3 * (np.eye(dim, k=1) + np.eye(dim, k=-1))
    model = ControlModel(H_d=hopping + np.diag(np.linspace(0.0, -2.0 * dim, dim)), H_c=np.diag(np.arange(dim) * 1.0))
    model.set_control(pulse_initial=-2.0, pulse_final=10.0, initial_state=1, final_state=3, alpha=3.75, beta=2.78,
                      dia_alpha=-7.03, dia_beta=9.79, num_steps=257, )
    model.solve_problem(pulse_accuracy=100)

    assert model._dia_list is not None
    dense = model._dia_list.dense()
    assert 0 < np.count_nonzero((dense >= 0) & (dense < 1))
    energies, elements = model._centered_energies, model._matrix_elements
    assert energies is not None and elements is not None
    expected = np.zeros(energies.shape[0])
    for i in range(energies.shape[0]):
        for m in range(dim):
            for n in range(m + 1, dim):
                fraction = dense[i, m, n]
                if fraction < 0:
                    continue
                alpha = fraction * 3.75 + (1.0 - fraction) * -7.03
                beta = fraction * 2.78 + (1.0 - fraction) * 9.79
                expected[i] += elements[i, m, n] ** beta / abs(energies[i, n] - energies[i, m]) ** alpha

    assert model._metric_tensor is not None
    np.testing.assert_allclose(model._metric_tensor, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Memory of the eigenproblem (C-17)
# ---------------------------------------------------------------------------
class TestEigenproblemMemory:
    N_POINTS, DIM = 1025, 64

    @classmethod
    def _matrices(cls) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(cls.DIM, cls.DIM)) + 1j * rng.normal(size=(cls.DIM, cls.DIM))
        return (a + a.conj().T) / 2, np.diag(np.linspace(-1.0, 1.0, cls.DIM))

    def _model(self, kind: str) -> ControlModel:
        h_d, h_c = self._matrices()
        if kind == "affine":
            model = ControlModel(H_d=h_d, H_c=h_c)
        elif kind == "exact":
            model = ControlModel(lambda x: h_d + x * h_c, lambda x: h_c)
        else:
            model = ControlModel(lambda x: h_d + x * h_c)
        model.set_control(control_name=None if kind == "affine" else "x", pulse_initial=-1.0, pulse_final=1.0,
                          num_steps=self.N_POINTS, )
        return model

    # Peak memory of _solve_eigenproblem in complex (N, d, d) stacks. Before C-17 the three paths took about 5, 6.5
    # and 7.5 stacks; now about 2, 2.2 and 3.5. The bounds leave room for the small chunk temporaries.
    @pytest.mark.parametrize(("kind", "bound"), [("affine", 2.5), ("exact", 2.75), ("numerical", 4.0)])
    def test_peak_memory_stays_bounded(self, kind: str, bound: float):
        model = self._model(kind)
        stack = self.N_POINTS * self.DIM ** 2 * 16

        tracemalloc.start()
        try:
            model._solve_eigenproblem()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert peak < bound * stack

    def test_chunked_matrix_elements_match_the_direct_product(self, monkeypatch):
        model = self._model("exact")
        h_d, h_c = self._matrices()
        # A few grid points per chunk, so the last chunk is incomplete.
        monkeypatch.setattr(ControlModel, "_MATRIX_ELEMENT_CHUNK_BYTES", 7 * 16 * self.DIM ** 2)
        model._solve_eigenproblem()

        x = model.control_pulse
        _, vectors = np.linalg.eigh(h_d[None] + x[:, None, None] * h_c[None])
        expected = np.abs(vectors.conj().transpose(0, 2, 1) @ h_c @ vectors)
        assert model._matrix_elements is not None
        np.testing.assert_allclose(model._matrix_elements, expected, atol=1e-12)
