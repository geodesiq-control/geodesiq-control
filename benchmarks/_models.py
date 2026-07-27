"""
Hamiltonian factory helpers for the geodesiq benchmark suite.

All factories return a fully configured :class:`~geodesiq.ControlModel` ready
to be passed to ``solve_problem()``.
"""

from __future__ import annotations

import numpy as np

from geodesiq import ControlModel


# ---------------------------------------------------------------------------
# Internal ControlModel builders
# ---------------------------------------------------------------------------


def _lz_H_dH(coupling: float = 0.5):
    """2×2 Landau-Zener Hamiltonian  H(λ) = λ·σ_z + coupling·σ_x."""
    D = np.array([[1.0, 0.0], [0.0, -1.0]])
    C = np.array([[0.0, coupling], [coupling, 0.0]])

    def H(lam: float) -> np.ndarray:
        return lam * D + C

    def dH(lam: float) -> np.ndarray:
        return D.copy()

    return H, dH


def _chain_H_dH(N: int, coupling: float = 0.5, seed: int = 42):
    """
    N-level chain Hamiltonian with a single λ-dependent drive:

        H(λ) = λ · D  +  coupling · C

    where ``D = diag(linspace(-1, 1, N))`` gives linearly spaced bare
    energies and ``C`` is a symmetric tridiagonal coupling matrix with a
    small random dense perturbation (normalized so the largest off-diagonal
    element equals ``coupling``).

    ∂H/∂λ = D  (constant, returned as ``dH``).
    """
    rng = np.random.default_rng(seed)
    D = np.diag(np.linspace(-1.0, 1.0, N))

    # Tridiagonal skeleton
    C = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i + 1] = 1.0
        C[i + 1, i] = 1.0

    # Small symmetric dense perturbation to lift accidental degeneracies
    noise = rng.standard_normal((N, N)) * 0.1
    noise = (noise + noise.T) / 2.0
    np.fill_diagonal(noise, 0.0)
    C += noise

    # Normalize: largest off-diagonal → coupling
    off_max = np.abs(C - np.diag(np.diag(C))).max()
    if off_max > 0:
        C = C / off_max * coupling

    # Capture in closures (D and C are immutable after this point)
    _D, _C = D, C

    def H(lam: float) -> np.ndarray:
        return lam * _D + _C

    def dH(lam: float) -> np.ndarray:
        return _D.copy()

    return H, dH


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_ham(dim: int = 2, num_steps: int = 2 ** 8 + 1, analytical_partial: bool = True, adiabatic: bool = True,
             pulse_initial: float = -5.0, pulse_final: float = 5.0, alpha: float = 2.0, beta: float = 2.0,
             coupling: float = 0.5, seed: int = 42, ) -> ControlModel:
    """
    Build and configure a :class:`~geodesiq.ControlModel` for benchmarking.

    Parameters
    ----------
    dim
        Hilbert-space dimension.  ``dim=2`` uses the Landau-Zener model;
        larger values use an N-level chain Hamiltonian.
    num_steps
        Number of λ grid points (controls the resolution of the eigenproblem
        sweep and is the main source of scaling for that stage).
    analytical_partial
        If ``True`` the analytical ∂H/∂λ is provided to the ControlModel.
        If ``False`` no partial function is supplied, so the internal
        numerical-differentiation path is exercised instead.
    adiabatic
        If ``True`` the adiabatic metric (initial state == final state) is
        used.  If ``False`` a diabatic transfer from state 0 → state
        *dim − 1* is configured (requires ``dia_alpha`` / ``dia_beta``).
    pulse_initial, pulse_final
        Range of the control parameter λ.
    alpha, beta
        Metric tensor exponents.
    coupling
        Off-diagonal coupling strength passed to the ControlModel model.
    seed
        RNG seed for the chain Hamiltonian's random noise term.
    """
    if dim == 2:
        H, dH = _lz_H_dH(coupling=coupling)
    else:
        H, dH = _chain_H_dH(dim, coupling=coupling, seed=seed)

    ham = ControlModel(H, partial_H_func=(dH if analytical_partial else None))

    ctrl: dict = dict(control_name="lam", pulse_initial=pulse_initial, pulse_final=pulse_final, initial_state=0,
                      alpha=alpha, beta=beta, num_steps=num_steps, )
    if not adiabatic:
        ctrl.update(final_state=dim - 1, dia_alpha=1.0, dia_beta=1.0)

    ham.set_control(**ctrl)
    return ham
