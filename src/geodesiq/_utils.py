from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Any

import numpy as np
from threadpoolctl import ThreadpoolController

from .exceptions import ValidationError


class Flags:
    """
    A class to manage flags with hierarchical dependencies.

    Each flag can have one or more parents. If any parent flag is down (False),
    all descendants reachable from that parent are set down as well.

    Example
    -------
    >>> flags = Flags()
    >>> flags.add("solver")
    >>> flags.add("gradient", parent="solver")
    >>> flags.add("hessian", parents=["gradient", "solver"])
    >>> flags["solver"]    # True
    >>> flags["solver"] = False
    >>> flags["gradient"]  # False  (parent is down)
    >>> flags["hessian"]   # False  (grandparent is down)
    """

    def __init__(self, _verbose: bool = False):
        self._values: dict[str, bool] = {}
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}

        self._verbose = _verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        value: bool = False,
        parent: str | None = None,
        parents: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        """Register a new flag.

        Parameters
        ----------
        name:
            Unique identifier for the flag.
        value:
            Initial value (default ``False``).
        parent:
            Name of an already-registered flag that this one depends on.
            Backward-compatible alias for a single entry in ``parents``.
        parents:
            Names of already-registered flags that this one depends on.

        Raises
        ------
        KeyError
            If *name* is already registered or any parent does not exist.
        ValueError
            If both *parent* and *parents* are given.
        """
        if name in self._values:
            raise KeyError(f"Flag '{name}' is already registered.")

        if parent is not None and parents is not None:
            raise ValueError("Use either 'parent' or 'parents', not both.")

        if parents is None:
            normalized_parents = [parent] if parent is not None else []
        else:
            normalized_parents = list(dict.fromkeys(parents))

        for parent_name in normalized_parents:
            if parent_name not in self._values:
                raise KeyError(f"Parent flag '{parent_name}' does not exist.")

        self._values[name] = value
        self._parents[name] = normalized_parents
        self._children[name] = []

        for parent_name in normalized_parents:
            if name not in self._children[parent_name]:
                self._children[parent_name].append(name)

    def get(self, name: str) -> bool:
        """Return the value of a flag.

        Descendants are explicitly reset when a parent is set to ``False``,
        so this method returns the stored value directly.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        self._check_exists(name)

        if self._verbose:
            print(f"Getting flag '{name}': stored value={self._values[name]}")

        return self._values[name]

    def set(self, name: str, value: bool) -> None:
        """Set the stored value of a flag.

        Parameters
        ----------
        name:
            Flag to update.
        value:
            New boolean value.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        self._check_exists(name)
        if not isinstance(value, bool):
            raise TypeError("Flag values must be booleans.")
        self._values[name] = value

        for child in self._children[name]:
            if not value:
                self.set(child, False)

        if self._verbose:
            print(f"Set flag '{name}' to {value}. Updated children: {self._children[name]}")

    def all(self) -> bool:
        """Return ``True`` if all flags are effectively up."""

        if self._verbose:
            print("Checking if all flags are effectively up:")
            for name in self._values:
                print(f"  {name}: {self.get(name)}")

        return all(self.get(name) for name in self._values)

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> bool:
        return self.get(name)

    def __setitem__(self, name: str, value: bool) -> None:
        self.set(name, value)

    def __repr__(self) -> str:
        lines = []
        for name, raw in self._values.items():
            parents = self._parents[name]
            lines.append(f"  {name}: stored={raw}," + (f" parents={parents!r}" if parents else ""))
        return "Flags(\n" + "\n".join(lines) + "\n)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exists(self, name: str) -> None:
        if name not in self._values:
            raise KeyError(f"Flag '{name}' is not registered.")


# Relative tolerance for Hermiticity checks, measured against the Frobenius norm of the matrix.
HERMITIAN_RTOL = 1e-8


def is_hermitian(matrix: np.ndarray, rtol: float = HERMITIAN_RTOL) -> bool:
    """
    Check Hermiticity relative to the scale of the matrix, so the result does not depend on the energy units.

    The matrix is Hermitian if ``||A - A^dagger||_F <= rtol * ||A||_F``. The zero matrix is Hermitian.
    """
    array = np.asarray(matrix)
    return bool(np.linalg.norm(array - array.conj().T) <= rtol * np.linalg.norm(array))


def build_diab(initial_state: int, final_state: int, dim: int) -> np.ndarray:
    """Build the adiabatic/diabatic transition mask with validated indices."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValidationError("dim must be a positive integer.")
    for label, state in (("initial_state", initial_state), ("final_state", final_state)):
        if not isinstance(state, int) or isinstance(state, bool) or not 0 <= state < dim:
            raise ValidationError(f"{label} must be an integer in [0, {dim - 1}].")
    diad_list = -1 * np.eye(dim, dtype=int)  # Diagonal entries are -1 by default

    min_state = min(initial_state, final_state)
    max_state = max(initial_state, final_state)

    for i in range(dim):
        for j in range(i + 1, dim):
            if min_state <= i <= max_state and min_state <= j <= max_state:
                diad_list[i, j] = 0
                diad_list[j, i] = 0
            else:
                diad_list[i, j] = 1
                diad_list[j, i] = 1

    return diad_list


# -----------------------------------
# Compare values
# -----------------------------------
def values_equal(a: Any, b: Any) -> bool:
    # NumPy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b, equal_nan=True)
        except (TypeError, ValueError):
            return False

    # Dictionaries
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(values_equal(a[key], b[key]) for key in a)

    # Lists / tuples
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))

    # Scalars, including NaN
    try:
        if np.isscalar(a) and np.isscalar(b) and np.isnan(a) and np.isnan(b):
            return True
    except TypeError:
        pass

    return a == b


def validate_state_index(index: int, dimension: int, name: str) -> int:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise ValidationError(f"{name} must be an integer.")

    index = int(index)

    if not 0 <= index < dimension:
        raise ValidationError(f"{name} must be between 0 and {dimension - 1}, got {index}.")

    return index


# -----------------------------------
# Linear algebra threading
# -----------------------------------
# Largest matrix dimension for which batched dense linear algebra runs single-threaded. Many small, independent
# problems parallelize poorly inside BLAS/LAPACK: thread start-up and synchronization dominate, and with many cores the
# multi-threaded call can be more than ten times slower than the single-threaded one.
SMALL_MATRIX_DIMENSION = 256


@lru_cache(maxsize=1)
def _threadpool_controller() -> ThreadpoolController:
    # Inspecting the loaded BLAS libraries is slow, so it is done once and reused.
    return ThreadpoolController()


@contextmanager
def limit_blas_threads(dimension: int) -> Generator[None, None, None]:
    """Run BLAS/LAPACK single-threaded inside the context if ``dimension`` is small, otherwise leave it untouched."""
    if dimension <= SMALL_MATRIX_DIMENSION:
        with _threadpool_controller().limit(limits=1, user_api="blas"):
            yield
    else:
        yield


# -----------------------------------
# Finite differences
# -----------------------------------
@lru_cache(maxsize=None)
def finite_difference_weights(offsets: tuple[int, ...]) -> np.ndarray:
    """
    Weights of the first-derivative finite-difference stencil on integer ``offsets`` (in units of the grid step).

    ``f'(x_0) ~= sum_j w_j f(x_0 + offsets[j] h) / h``, exact for polynomials of degree ``len(offsets) - 1``. The
    Vandermonde system is solved in exact rational arithmetic, so the weights carry no conditioning error.
    """
    m = len(offsets)
    if m < 2 or len(set(offsets)) != m:
        raise ValidationError("A finite-difference stencil needs at least two distinct offsets.")

    # Rows k = 0..m-1: sum_j w_j o_j^k / k! = delta_{k,1}
    matrix = [[Fraction(o) ** k / factorial(k) for o in offsets] + [Fraction(int(k == 1))] for k in range(m)]
    for col in range(m):
        pivot = next(row for row in range(col, m) if matrix[row][col] != 0)
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for row in range(m):
            if row != col and matrix[row][col] != 0:
                factor = matrix[row][col] / matrix[col][col]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col], strict=True)]
    weights = np.array([float(matrix[k][m] / matrix[k][k]) for k in range(m)])
    weights.flags.writeable = False
    return weights


def uniform_grid_derivative(samples: np.ndarray, step: float, n_points: int) -> np.ndarray:
    """
    First derivative along axis 0 of samples taken on a uniform grid with spacing ``step``.

    Each point uses the ``n_points`` nearest samples: a centered stencil in the interior and shifted (one-sided)
    stencils near the boundaries, so the accuracy order ``n_points - 1`` is the same everywhere.
    """
    n = samples.shape[0]
    if not 2 <= n_points <= n:
        raise ValidationError(f"n_points must be between 2 and the number of samples ({n}); got {n_points}.")

    derivative = np.zeros(samples.shape, dtype=np.result_type(samples.dtype, float))
    half = n_points // 2
    # Stencil start for every point; points sharing the same relative stencil form one contiguous range.
    starts = np.clip(np.arange(n) - half, 0, n - n_points)
    relative = starts - np.arange(n)
    boundaries = np.flatnonzero(np.diff(relative)) + 1
    for block in np.split(np.arange(n), boundaries):
        first, last = int(block[0]), int(block[-1]) + 1
        offsets = tuple(int(relative[first]) + j for j in range(n_points))
        weights = finite_difference_weights(offsets)
        for offset, weight in zip(offsets, weights, strict=True):
            derivative[first:last] += weight * samples[first + offset : last + offset]

    return derivative / step
