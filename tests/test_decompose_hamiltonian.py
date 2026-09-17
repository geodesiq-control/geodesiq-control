import numpy as np
import pytest
import qutip as qt

from geodesiq.decompose_hamiltonian import decompose_hamiltonian
from typing import Any, cast


def _reference_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_d = np.array([[0.4, 0.1j], [-0.1j, -0.2]], dtype=complex)
    h_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    h_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return h_d, h_x, h_z


def _model_hamiltonian(t: float) -> np.ndarray:
    h_d, h_x, h_z = _reference_components()
    return h_d + np.sin(0.8 * t) * h_x + (0.3 + np.cos(1.1 * t)) * h_z


def test_decompose_reconstructs_sampled_hamiltonians() -> None:
    times = np.linspace(0.0, 3.0, 21)
    h_d, _, _ = _reference_components()

    decomposition = decompose_hamiltonian(_model_hamiltonian, times=times, drift=h_d, rank=2)

    assert decomposition.rank == 2
    assert len(decomposition.H_controls) == 2
    assert decomposition.coefficients.shape == (2, len(times))

    for index, t in enumerate(times):
        reconstructed = decomposition.reconstruct_sample(index).full()
        np.testing.assert_allclose(reconstructed, _model_hamiltonian(float(t)), atol=1e-10)

    assert decomposition.relative_residual_error <= 1e-12
    assert decomposition.relative_total_error <= 1e-12


def test_qobjevo_matches_reconstruction_at_sample_points() -> None:
    times = np.linspace(0.0, 2.0, 13)
    decomposition = decompose_hamiltonian(_model_hamiltonian, times=times, drift="mean", rank=2)

    qevo = decomposition.qobjevo(order=3)
    assert isinstance(qevo, qt.QobjEvo)

    for index, t in enumerate(times):
        from_qevo = qevo(float(t)).full()
        from_reconstruct = decomposition.reconstruct_sample(index).full()
        np.testing.assert_allclose(from_qevo, from_reconstruct, atol=1e-9)


def test_drift_modes_use_expected_static_term() -> None:
    times = np.linspace(0.0, 1.5, 9)
    samples = np.asarray([_model_hamiltonian(float(t)) for t in times])

    mean_decomposition = decompose_hamiltonian(_model_hamiltonian, times=times, drift="mean", rank=2)
    first_decomposition = decompose_hamiltonian(_model_hamiltonian, times=times, drift="first", rank=2)
    zero_decomposition = decompose_hamiltonian(_model_hamiltonian, times=times, drift="zero", rank=2)

    np.testing.assert_allclose(mean_decomposition.H_d.full(), np.mean(samples, axis=0), atol=1e-12)
    np.testing.assert_allclose(first_decomposition.H_d.full(), samples[0], atol=1e-12)
    np.testing.assert_allclose(zero_decomposition.H_d.full(), np.zeros((2, 2), dtype=complex), atol=1e-12)


def test_auto_rank_uses_singular_value_threshold() -> None:
    times = np.linspace(0.0, 2.0, 15)
    h_d, h_x, _ = _reference_components()

    def one_control_hamiltonian(t: float) -> np.ndarray:
        return h_d + np.sin(t) * h_x

    rank_one = decompose_hamiltonian(one_control_hamiltonian, times=times, drift=h_d, rtol=1e-12, atol=0.0)
    assert rank_one.rank == 1

    rank_zero = decompose_hamiltonian(one_control_hamiltonian, times=times, drift=h_d, rtol=0.0, atol=1e6)
    assert rank_zero.rank == 0
    assert rank_zero.coefficients.shape == (0, len(times))


def test_qobj_input_preserves_dims_in_outputs() -> None:
    times = np.linspace(0.0, 1.0, 7)
    dims = [[2], [2]]

    def qobj_hamiltonian(t: float) -> qt.Qobj:
        matrix = _model_hamiltonian(t)
        return qt.Qobj(matrix, dims=dims)

    decomposition = decompose_hamiltonian(qobj_hamiltonian, times=times, drift="mean", rank=2)

    assert decomposition.H_d.dims == dims
    assert all(control.dims == dims for control in decomposition.H_controls)


def test_invalid_times_raise_value_error() -> None:
    with pytest.raises(ValueError, match="one-dimensional non-empty"):
        decompose_hamiltonian(_model_hamiltonian, times=np.array([]))

    with pytest.raises(ValueError, match="one-dimensional non-empty"):
        decompose_hamiltonian(_model_hamiltonian, times=np.array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="strictly increasing"):
        decompose_hamiltonian(_model_hamiltonian, times=np.array([0.0, 0.5, 0.5]))


def test_non_square_and_non_hermitian_samples_raise_value_error() -> None:
    times = np.array([0.0, 0.5, 1.0])

    def non_square(_: float) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def non_hermitian(t: float) -> np.ndarray:
        return np.array([[0.0, 1.0 + t], [0.0, 0.0]], dtype=complex)

    with pytest.raises(ValueError, match="square matrices"):
        decompose_hamiltonian(non_square, times=times)

    with pytest.raises(ValueError, match="must be Hermitian"):
        decompose_hamiltonian(non_hermitian, times=times)


def test_invalid_drift_configuration_raises_value_error() -> None:
    times = np.linspace(0.0, 1.0, 5)

    with pytest.raises(ValueError, match="Unknown drift option"):
        decompose_hamiltonian(_model_hamiltonian, times=times, drift=cast(Any, "invalid"))

    with pytest.raises(ValueError, match="incompatible dimensions"):
        decompose_hamiltonian(_model_hamiltonian, times=times, drift=np.eye(3))

    with pytest.raises(ValueError, match="H_d must be Hermitian"):
        decompose_hamiltonian(_model_hamiltonian, times=times,
                              drift=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))


def test_invalid_rank_bounds_raise_value_error() -> None:
    times = np.linspace(0.0, 1.0, 7)

    with pytest.raises(ValueError, match="rank must satisfy"):
        decompose_hamiltonian(_model_hamiltonian, times=times, rank=-1)

    with pytest.raises(ValueError, match="rank must satisfy"):
        decompose_hamiltonian(_model_hamiltonian, times=times, rank=10)
