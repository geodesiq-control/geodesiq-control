from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from geodesiq import ControlModel
from geodesiq.exceptions import (
    InvalidControlParameterError,
    MetricComputationError,
    SolverError,
    ValidationError,
)

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
    return np.array([[lam, 0.0], [0.0, -lam], ], dtype=float, )


def degenerate_partial(lam: float, ) -> np.ndarray:
    del lam

    return np.array([[1.0, 0.0], [0.0, -1.0], ], dtype=float, )


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


def test_degenerate_gap_raises_clear_metric_error():
    def hamiltonian(z: float) -> np.ndarray:
        return np.array([[z, 0.0], [0.0, -z]])

    model = ControlModel(hamiltonian, lambda z: np.array([[1.0, 0.0], [0.0, -1.0]]))
    model.set_control(control_name="z", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, alpha=2.0, beta=2.0,
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


# ---------------------------------------------------------------------------
# Re-solving keeps the pulse accuracy
# ---------------------------------------------------------------------------
class TestPulseAccuracyIsReused:
    def test_control_sol_resolves_with_previous_accuracy(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=50)

        model.set_control(alpha=3.0)

        assert model.control_sol.shape == (50,)
        assert model.s.shape == (50,)

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

    def test_s_property_is_a_copy_of_the_normalized_grid(self):
        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        s = model.s
        s[0] = 5.0

        assert model.s[0] == 0.0
        assert model.s[-1] == pytest.approx(1.0)


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
    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_first_branch_uses_the_new_grid(self):
        import matplotlib.pyplot as plt

        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        model.set_control(pulse_initial=-1.0)
        fig, ax = cast(Any, model.plot_eigenvalues())

        expected_grid = np.linspace(-1.0, 3.0, 65)
        for line in ax.get_lines():
            np.testing.assert_allclose(line.get_xdata(), expected_grid)
        plt.close(fig)

    def test_changing_num_steps_does_not_break_plot(self):
        import matplotlib.pyplot as plt

        model = configured_model()
        model.solve_problem(pulse_accuracy=30)

        model.set_control(num_steps=33)
        fig, ax = cast(Any, model.plot_eigenvalues())

        assert all(len(line.get_xdata()) == 33 for line in ax.get_lines())
        plt.close(fig)


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

        x = model.control_pulse
        expected = np.zeros((x.size, 2, 2), dtype=complex)
        expected[:, 0, 0] = 3 * np.cos(3 * x)
        expected[:, 0, 1] = 3j * x ** 2
        expected[:, 1, 0] = -3j * x ** 2
        expected[:, 1, 1] = np.exp(x)
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
