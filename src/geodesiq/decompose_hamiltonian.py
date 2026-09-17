from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import qutip as qt


def _hermitian_to_real_vector(H: np.ndarray) -> np.ndarray:
    """Map a d x d Hermitian matrix to R^(d^2), preserving the Frobenius inner product."""
    d = H.shape[0]
    upper = np.triu_indices(d, 1)
    n_upper = len(upper[0])

    vector = np.empty(d ** 2, dtype=float)
    vector[:d] = np.diag(H).real
    vector[d:d + n_upper] = np.sqrt(2.0) * H[upper].real
    vector[d + n_upper:] = np.sqrt(2.0) * H[upper].imag

    return vector


def _real_vector_to_hermitian(vector: np.ndarray, d: int) -> np.ndarray:
    """Inverse of _hermitian_to_real_vector."""
    H = np.zeros((d, d), dtype=complex)

    np.fill_diagonal(H, vector[:d])

    upper = np.triu_indices(d, 1)
    n_upper = len(upper[0])

    values = (vector[d:d + n_upper] + 1j * vector[d + n_upper:]) / np.sqrt(2.0)

    H[upper] = values
    H[(upper[1], upper[0])] = values.conj()

    return H


@dataclass
class HamiltonianDecomposition:
    times: np.ndarray
    H_d: qt.Qobj
    H_controls: list[qt.Qobj]
    coefficients: np.ndarray
    singular_values: np.ndarray
    rank: int
    relative_residual_error: float
    relative_total_error: float

    def qobjevo(self, order: int = 3) -> qt.QobjEvo:
        """Construct H(t) = H_d + sum_i u_i(t) H_i as a QuTiP QobjEvo."""
        terms = [self.H_d]
        terms.extend([[H, coefficient] for H, coefficient in zip(self.H_controls, self.coefficients, strict=False)])
        return qt.QobjEvo(terms, tlist=self.times, order=order)

    def reconstruct_sample(self, index: int) -> qt.Qobj:
        """Reconstruct H at one of the sampled times."""
        H = self.H_d.copy()

        for H_control, coefficient in zip(self.H_controls, self.coefficients, strict=False):
            H += coefficient[index] * H_control

        return H


def decompose_hamiltonian(H_func: Callable[[float], qt.Qobj | np.ndarray],
                          times: np.ndarray,
                          drift: Literal["mean", "first", "zero"] | qt.Qobj | np.ndarray = "mean",
                          rank: int | None = None,
                          rtol: float = 1e-10,
                          atol: float = 0.0,
                          hermitian_tol: float = 1e-10, ) -> HamiltonianDecomposition:
    """
    Numerically decompose a time-dependent Hamiltonian as

        H(t) ~= H_d + sum_i u_i(t) H_i

    using an SVD of sampled Hamiltonians.

    Parameters
    ----------
    H_func
        Callable returning H(t) as a Qobj or numpy array.
    times
        Times at which H(t) is sampled.
    drift
        Choice of static drift:
            "mean"  -> mean Hamiltonian over sampled times
            "first" -> H(times[0])
            "zero"  -> no drift
            Qobj / ndarray -> explicitly supplied drift
    rank
        Number of SVD components to retain. If None, determine it using
        singular_value > atol + rtol * largest_singular_value.
    rtol
        Relative singular-value cutoff.
    atol
        Absolute singular-value cutoff.
    hermitian_tol
        Tolerance used to verify Hermiticity.

    Returns
    -------
    HamiltonianDecomposition
    """
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must be a one-dimensional non-empty array.")

    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing.")

    raw_samples = [H_func(float(t)) for t in times]

    first = raw_samples[0]
    dims = first.dims if isinstance(first, qt.Qobj) else None

    samples = np.asarray([H.full() if isinstance(H, qt.Qobj) else np.asarray(H, dtype=complex) for H in raw_samples])

    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError("H_func must return square matrices.")

    d = samples.shape[1]

    for H in samples:
        if not np.allclose(H, H.conj().T, atol=hermitian_tol, rtol=0.0):
            raise ValueError("All sampled Hamiltonians must be Hermitian.")

    if isinstance(drift, str):
        if drift == "mean":
            H_d_array = np.mean(samples, axis=0)
        elif drift == "first":
            H_d_array = samples[0].copy()
        elif drift == "zero":
            H_d_array = np.zeros((d, d), dtype=complex)
        else:
            raise ValueError(f"Unknown drift option: {drift}")
    else:
        H_d_array = drift.full() if isinstance(drift, qt.Qobj) else np.asarray(drift, dtype=complex)

    if H_d_array.shape != (d, d):
        raise ValueError("The drift Hamiltonian has incompatible dimensions.")

    if not np.allclose(H_d_array, H_d_array.conj().T, atol=hermitian_tol, rtol=0.0):
        raise ValueError("H_d must be Hermitian.")

    residuals = samples - H_d_array

    matrix = np.column_stack([_hermitian_to_real_vector(H) for H in residuals])

    U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)

    if rank is None:
        threshold = atol + rtol * singular_values[0] if singular_values.size else atol
        rank = int(np.sum(singular_values > threshold))
    elif not 0 <= rank <= len(singular_values):
        raise ValueError(f"rank must satisfy 0 <= rank <= {len(singular_values)}.")

    basis_vectors = U[:, :rank]
    coefficients = singular_values[:rank, None] * Vh[:rank, :]

    H_controls = []

    for i in range(rank):
        H_array = _real_vector_to_hermitian(basis_vectors[:, i], d)
        H_controls.append(qt.Qobj(H_array, dims=dims) if dims is not None else qt.Qobj(H_array))

    H_d = qt.Qobj(H_d_array, dims=dims) if dims is not None else qt.Qobj(H_d_array)

    discarded_norm = np.linalg.norm(singular_values[rank:])
    residual_norm = np.linalg.norm(singular_values)

    full_matrix = np.column_stack([_hermitian_to_real_vector(H) for H in samples])
    full_norm = np.linalg.norm(full_matrix)

    relative_residual_error = float(discarded_norm / residual_norm if residual_norm > 0 else 0.0)
    relative_total_error = float(discarded_norm / full_norm if full_norm > 0 else 0.0)

    return HamiltonianDecomposition(times=times, H_d=H_d, H_controls=H_controls, coefficients=coefficients,
                                    singular_values=singular_values, rank=rank,
                                    relative_residual_error=relative_residual_error,
                                    relative_total_error=relative_total_error, )
