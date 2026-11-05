from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt
from numpy.typing import ArrayLike, NDArray

from ._utils import HERMITIAN_RTOL, is_hermitian
from .exceptions import ValidationError


def _hermitian_to_real_vector(H: np.ndarray) -> np.ndarray:
    """Map a d x d Hermitian matrix to R^(d^2), preserving the Frobenius inner product."""
    d = H.shape[0]
    upper = np.triu_indices(d, 1)
    n_upper = len(upper[0])

    vector = np.empty(d**2, dtype=float)
    vector[:d] = np.diag(H).real
    vector[d : d + n_upper] = np.sqrt(2.0) * H[upper].real
    vector[d + n_upper :] = np.sqrt(2.0) * H[upper].imag

    return vector


def _real_vector_to_hermitian(vector: np.ndarray, d: int) -> np.ndarray:
    """Inverse of _hermitian_to_real_vector."""
    H = np.zeros((d, d), dtype=complex)

    np.fill_diagonal(H, vector[:d])

    upper = np.triu_indices(d, 1)
    n_upper = len(upper[0])

    values = (vector[d : d + n_upper] + 1j * vector[d + n_upper :]) / np.sqrt(2.0)

    H[upper] = values
    H[(upper[1], upper[0])] = values.conj()

    return H


@dataclass
class HamiltonianDecomposition:
    """
    Low-rank decomposition ``H(t) ~= H_d + sum_i u_i(t) H_i`` of a sampled Hamiltonian, as returned by
    ``decompose_hamiltonian``.

    Attributes
    ----------
    times : NDArray[np.float64]
        Strictly increasing sample times, shape ``(N,)``.
    H_d : qt.Qobj
        Static drift Hamiltonian.
    H_controls : list[qt.Qobj]
        The ``rank`` Hermitian control Hamiltonians ``H_i``, orthonormal in the Frobenius inner product.
    coefficients : NDArray[np.float64]
        Real control coefficients ``u_i(t)`` at the sample times, shape ``(rank, N)``.
    singular_values : NDArray[np.float64]
        All singular values of the sampled residuals ``H(t) - H_d``, in decreasing order, including the discarded ones.
    rank : int
        Number of control terms kept.
    relative_residual_error : float
        Frobenius norm of the discarded part relative to that of the residuals ``H(t) - H_d``.
    relative_total_error : float
        Frobenius norm of the discarded part relative to that of the sampled ``H(t)``.
    """

    times: NDArray[np.float64]
    H_d: qt.Qobj
    H_controls: list[qt.Qobj]
    coefficients: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    rank: int
    relative_residual_error: float
    relative_total_error: float

    def qobjevo(self, order: int = 3) -> qt.QobjEvo:
        """Construct H(t) = H_d + sum_i u_i(t) H_i as a QuTiP QobjEvo.

        Parameters
        ----------
        order : int
            Spline interpolation order for the sampled coefficients (default is 3).
            Use 0 for sample-and-hold or 1 for linear interpolation.

        Returns
        -------
        qt.QobjEvo
            Time-dependent Hamiltonian with coefficients interpolated on ``times``.
        """
        terms = [self.H_d]
        terms.extend([[H, coefficient] for H, coefficient in zip(self.H_controls, self.coefficients, strict=False)])
        return qt.QobjEvo(terms, tlist=self.times, order=order)

    def reconstruct_sample(self, index: int) -> qt.Qobj:
        """Reconstruct H at one of the sampled times.

        Parameters
        ----------
        index : int
            Index into ``times`` and the coefficient samples. Negative indices count
            from the end, following NumPy indexing conventions.

        Returns
        -------
        qt.Qobj
            Hamiltonian at ``times[index]``, reconstructed using the retained control terms.
            Discarded terms are not included.
        """
        H = self.H_d.copy()

        for H_control, coefficient in zip(self.H_controls, self.coefficients, strict=False):
            H += coefficient[index] * H_control

        return H


def decompose_hamiltonian(
    H_func: Callable[[float], qt.Qobj | np.ndarray],
    times: ArrayLike,
    drift: Literal["mean", "first", "zero"] | qt.Qobj | np.ndarray = "mean",
    rank: int | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    hermitian_tol: float = HERMITIAN_RTOL,
) -> HamiltonianDecomposition:
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
        Relative tolerance used to verify Hermiticity:
        ``||H - H^dagger||_F <= hermitian_tol * ||H_0||_F + 100 * eps * ||H||_F``, with ``H_0`` the traceless part of
        ``H`` (see ``_utils.is_hermitian``).

    Returns
    -------
    HamiltonianDecomposition
        The drift, control Hamiltonians and coefficients, with the truncation errors.
    """
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 1:
        raise ValidationError("times must be a one-dimensional non-empty array.")

    if np.any(np.diff(times) <= 0):
        raise ValidationError("times must be strictly increasing.")

    raw_samples = [H_func(float(t)) for t in times]

    first = raw_samples[0]
    dims = first.dims if isinstance(first, qt.Qobj) else None

    samples = np.asarray([H.full() if isinstance(H, qt.Qobj) else np.asarray(H, dtype=complex) for H in raw_samples])

    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValidationError("H_func must return square matrices.")

    d = samples.shape[1]

    if not np.all(np.isfinite(samples)):
        raise ValidationError("All sampled Hamiltonians must contain only finite values.")

    for H in samples:
        if not is_hermitian(H, rtol=hermitian_tol):
            raise ValidationError("All sampled Hamiltonians must be Hermitian.")

    if isinstance(drift, str):
        if drift == "mean":
            H_d_array = np.mean(samples, axis=0)
        elif drift == "first":
            H_d_array = samples[0].copy()
        elif drift == "zero":
            H_d_array = np.zeros((d, d), dtype=complex)
        else:
            raise ValidationError(f"Unknown drift option: {drift}")
    else:
        H_d_array = drift.full() if isinstance(drift, qt.Qobj) else np.asarray(drift, dtype=complex)

    if H_d_array.shape != (d, d):
        raise ValidationError("The drift Hamiltonian has incompatible dimensions.")

    if not np.all(np.isfinite(H_d_array)):
        raise ValidationError("H_d must contain only finite values.")

    if not is_hermitian(H_d_array, rtol=hermitian_tol):
        raise ValidationError("H_d must be Hermitian.")

    residuals = samples - H_d_array

    # One column per sample, filled in place to avoid a second d**2 x N copy. The SVD stays on this matrix: the Gram
    # matrix would square its condition number and lose the singular values below about 1e-8 of the largest, which
    # the default rtol=1e-10 cutoff needs (D-19).
    matrix = np.empty((d * d, len(times)), dtype=float)
    for column, H in enumerate(residuals):
        matrix[:, column] = _hermitian_to_real_vector(H)

    U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)

    if rank is None:
        threshold = atol + rtol * singular_values[0] if singular_values.size else atol
        rank = int(np.sum(singular_values > threshold))
    elif not 0 <= rank <= len(singular_values):
        raise ValidationError(f"rank must satisfy 0 <= rank <= {len(singular_values)}.")

    basis_vectors = U[:, :rank]
    coefficients = singular_values[:rank, None] * Vh[:rank, :]

    H_controls = []

    for i in range(rank):
        H_array = _real_vector_to_hermitian(basis_vectors[:, i], d)
        H_controls.append(qt.Qobj(H_array, dims=dims) if dims is not None else qt.Qobj(H_array))

    H_d = qt.Qobj(H_d_array, dims=dims) if dims is not None else qt.Qobj(H_d_array)

    discarded_norm = np.linalg.norm(singular_values[rank:])
    residual_norm = np.linalg.norm(singular_values)

    # The real-vector map preserves the Frobenius norm, so the norm of all samples needs no second matrix.
    full_norm = np.linalg.norm(samples)

    relative_residual_error = float(discarded_norm / residual_norm if residual_norm > 0 else 0.0)
    relative_total_error = float(discarded_norm / full_norm if full_norm > 0 else 0.0)

    return HamiltonianDecomposition(
        times=times,
        H_d=H_d,
        H_controls=H_controls,
        coefficients=coefficients,
        singular_values=singular_values,
        rank=rank,
        relative_residual_error=relative_residual_error,
        relative_total_error=relative_total_error,
    )
