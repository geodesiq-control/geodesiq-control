from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from geodesiq import ControlModel
from geodesiq.exceptions import (InvalidControlParameterError, MetricComputationError, SolverError, )


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

    with pytest.raises(SolverError) as exc_info:
        model._solve_eigenproblem()

    assert "Numerical differentiation failed to converge" in str(exc_info.value)
    assert str(status_code) in str(exc_info.value)


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
