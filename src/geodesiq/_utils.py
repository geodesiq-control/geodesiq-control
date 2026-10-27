import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import factorial
from typing import Any

import numpy as np
from threadpoolctl import ThreadpoolController

from .exceptions import InvalidControlParameterError, ValidationError


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

    def __init__(self, _verbose: bool = False) -> None:
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
        ValidationError
            If *value* is ``True`` while a parent is ``False``.
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
        if value:
            self._check_parents_up(name, normalized_parents)

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

        A flag can only be set to ``True`` while all its parents are ``True``, so that a flag never claims a result
        whose inputs are missing. Setting it to ``False`` also sets all its descendants to ``False``.

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
        ValidationError
            If *value* is ``True`` while a parent is ``False``.
        """
        self._check_exists(name)
        if not isinstance(value, bool):
            raise TypeError("Flag values must be booleans.")
        if value:
            self._check_parents_up(name, self._parents[name])
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

    def _check_parents_up(self, name: str, parents: list[str]) -> None:
        down = [parent for parent in parents if not self._values[parent]]
        if down:
            raise ValidationError(f"Flag '{name}' cannot be True while its parent flag(s) {down} are False.")


# Relative tolerance for Hermiticity checks, measured against the Frobenius norm of the traceless part of the matrix.
HERMITIAN_RTOL = 1e-8
# Allowance for the rounding of matrices with a large trace, in units of eps * ||A||_F.
_HERMITIAN_ROUNDOFF_FACTOR = 100.0


def is_hermitian(matrix: np.ndarray, rtol: float = HERMITIAN_RTOL) -> bool:
    """
    Check Hermiticity relative to the scale of the matrix, so the result does not depend on the energy units.

    The matrix is Hermitian if ``||A - A^dagger||_F <= rtol * ||A_0||_F + 100 * eps * ||A||_F``, where
    ``A_0 = A - tr(A)/d * I`` is its traceless part. Measuring against ``||A_0||`` keeps a large identity offset (which
    does not affect the physics) from hiding a non-Hermitian part, e.g. ``1e8 * I + [[0, 1], [0.5, 0]]``. The second
    term allows for the rounding errors of such an offset. The zero matrix is Hermitian.
    """
    array = np.asarray(matrix)
    dimension = array.shape[0]
    traceless = array - np.trace(array) / dimension * np.eye(dimension)
    tolerance = rtol * np.linalg.norm(traceless) + _HERMITIAN_ROUNDOFF_FACTOR * np.finfo(float).eps * np.linalg.norm(
        array
    )
    return bool(np.linalg.norm(array - array.conj().T) <= tolerance)


# Relative tolerance for dropping the imaginary part of real-valued samples, measured against the largest real part.
REAL_RTOL = 1e-10


def real_samples(values: np.ndarray, name: str, rtol: float = REAL_RTOL) -> np.ndarray:
    """
    Return the real part of complex samples whose imaginary part is negligible, relative to their scale.

    The samples are accepted if ``max|Im| <= rtol * max(max|Re|, tiny)``, so the check does not depend on the units of
    the samples. Real input is returned unchanged.

    Raises
    ------
    ValidationError
        If the imaginary part is not negligible.
    """
    if not np.iscomplexobj(values):
        return values
    scale = max(float(np.max(np.abs(values.real), initial=0.0)), np.finfo(float).tiny)
    if np.max(np.abs(values.imag), initial=0.0) > rtol * scale:
        raise ValidationError(f"{name} must contain real-valued samples.")
    return values.real


def stack_samples(function: Callable[[float], Any], points: np.ndarray) -> np.ndarray:
    """
    Evaluate ``function`` at every point and stack the results along a new first axis, like ``np.stack``.

    The results are written into one preallocated array, so the list of samples and the stacked copy of
    ``np.stack`` do not both exist. The dtype is at least ``float`` and is widened (e.g. to complex) if a later sample
    requires it.
    """
    first = np.asarray(function(points[0]))
    stacked = np.empty((len(points), *first.shape), dtype=np.result_type(first.dtype, float))
    stacked[0] = first
    for index in range(1, len(points)):
        sample = np.asarray(function(points[index]))
        if sample.shape != first.shape:
            raise ValidationError(
                f"All samples must have the same shape: expected {first.shape}, got {sample.shape} at point {index}."
            )
        if not np.can_cast(sample.dtype, stacked.dtype, casting="safe"):
            stacked = stacked.astype(np.result_type(stacked.dtype, sample.dtype))
        stacked[index] = sample
    return stacked


def traceless(matrices: np.ndarray) -> np.ndarray:
    """
    Remove the trace part ``tr(H)/d * I`` from a Hermitian matrix or a stack of them (last two axes).

    The trace of a Hermitian matrix is real, so its rounding-level imaginary part is dropped.
    """
    array = np.asarray(matrices)
    dimension = array.shape[-1]
    offset = np.trace(array, axis1=-2, axis2=-1).real / dimension
    return array - offset[..., None, None] * np.eye(dimension)


def degenerate_clusters(energies: np.ndarray, rtol: float) -> list[tuple[int, int, int]]:
    """
    Groups of degenerate eigenstates, as ``(grid index, first state, last state)`` with ``last > first``.

    Two neighbouring levels are degenerate if their gap is at most ``rtol`` times the spectral bandwidth at that point.
    ``energies`` are sorted in increasing order along the last axis, with shape (n_points, dim).
    """
    tolerance = rtol * np.ptp(energies, axis=1)
    degenerate = np.diff(energies, axis=1) <= tolerance[:, None]
    clusters = []
    for point in np.flatnonzero(np.any(degenerate, axis=1)):
        edges = np.diff(np.concatenate(([0], degenerate[point].astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        clusters.extend((int(point), int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    return clusters


def rotate_degenerate_subspaces(eigenvectors: np.ndarray, partial_H: np.ndarray, clusters: list) -> None:
    """
    Choose the eigenvectors of each degenerate subspace so that they diagonalize dH/dx within it (in place).

    ``eigh`` returns an arbitrary basis of a degenerate subspace. At a level crossing, the basis that diagonalizes
    dH/dx is the limit of the eigenvectors on either side, so the states continue smoothly through the crossing, and
    the matrix element of dH/dx between two crossing states that are not coupled vanishes.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors as columns, with shape (n_points, dim, dim).
    partial_H : np.ndarray
        dH/dx, one matrix (affine model) or one per grid point.
    clusters : list
        Degenerate subspaces from ``degenerate_clusters``.
    """
    for point, first, last in clusters:
        vectors = eigenvectors[point, :, first : last + 1]
        derivative = partial_H if partial_H.ndim == 2 else partial_H[point]
        _, rotation = np.linalg.eigh(vectors.conj().T @ derivative @ vectors)
        eigenvectors[point, :, first : last + 1] = vectors @ rotation


def _permutation_cycles(permutation: np.ndarray) -> list[np.ndarray]:
    """Cycles of length > 1 of a permutation of ``range(len(permutation))``."""
    visited = np.zeros(permutation.size, dtype=bool)
    cycles = []
    for start in range(permutation.size):
        cycle = []
        index = start
        while not visited[index]:
            visited[index] = True
            cycle.append(index)
            index = int(permutation[index])
        if len(cycle) > 1:
            cycles.append(np.array(cycle))
    return cycles


def _uncoupled(matrix_elements: np.ndarray, states: np.ndarray, rtol: float) -> bool:
    """Whether dH/dx does not couple any two of ``states``, relative to its largest matrix element at that point."""
    block = matrix_elements[np.ix_(states, states)]
    off_diagonal = block[~np.eye(states.size, dtype=bool)]
    return bool(np.all(off_diagonal <= rtol * np.max(matrix_elements)))


def crossing_map(eigenvectors: np.ndarray, matrix_elements: np.ndarray, rtol: float, chunk: int) -> np.ndarray:
    """
    Follow the eigenstates through true (symmetry-protected) level crossings between neighbouring grid points.

    States are indexed in increasing energy at every grid point. At a true crossing, two states that dH/dx does not
    couple exchange their energy order, so the index of each one changes. Between two grid points, the eigenvectors
    are matched by their overlap ``|<m(i-1)|n(i)>|^2``. A change of index is accepted only if dH/dx does not couple the
    states involved (all matrix elements between them at most ``rtol`` times the largest one, at both points);
    otherwise it is an anticrossing that the grid does not resolve, and the states keep their energy order.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors as columns, sorted by increasing energy, with shape (n_points, dim, dim).
    matrix_elements : np.ndarray
        ``|<m|dH/dx|n>|`` at every grid point, with shape (n_points, dim, dim).
    rtol : float
        Relative tolerance below which two states count as uncoupled.
    chunk : int
        Number of grid points whose overlaps are computed together.

    Returns
    -------
    np.ndarray
        Integer array with shape (n_points, dim). Entry ``[i, m]`` is the index at grid point ``i`` of the state with
        index ``m`` at grid point ``i - 1``. Row 0, and every row without a crossing, is ``range(dim)``.
    """
    from scipy.optimize import linear_sum_assignment

    n_points, dimension = eigenvectors.shape[:2]
    state_map = np.tile(np.arange(dimension), (n_points, 1))
    # A state whose overlap with itself at the previous point exceeds 1/2 has the largest overlap with itself, so its
    # index cannot have changed. The full overlap matrix is only needed where this fails for some state.
    candidates = []
    for start in range(1, n_points, chunk):
        block = slice(start, min(start + chunk, n_points))
        previous = eigenvectors[block.start - 1 : block.stop - 1]
        same = np.abs(np.einsum("pij,pij->pj", previous.conj(), eigenvectors[block])) ** 2
        candidates.extend(start + np.flatnonzero(np.any(same <= 0.5, axis=1)))

    for point in candidates:
        overlap = np.abs(eigenvectors[point - 1].conj().T @ eigenvectors[point]) ** 2
        _, permutation = linear_sum_assignment(overlap, maximize=True)
        for cycle in _permutation_cycles(permutation):
            if _uncoupled(matrix_elements[point - 1], cycle, rtol) and _uncoupled(
                matrix_elements[point], permutation[cycle], rtol
            ):
                state_map[point, cycle] = permutation[cycle]
    return state_map


def follow_state(state_map: np.ndarray, state: int) -> np.ndarray:
    """
    Energy index of the state that starts with index ``state``, at every grid point of a ``crossing_map``.

    Without true crossings, it is ``state`` everywhere.
    """
    occupied = np.empty(state_map.shape[0], dtype=int)
    for point in range(state_map.shape[0]):
        if point > 0:
            state = int(state_map[point, state])
        occupied[point] = state
    return occupied


def _gap_minima(gap: np.ndarray) -> np.ndarray:
    """Mask of the local minima of a sampled energy gap, including minima at the ends of the grid."""
    is_min = np.ones(gap.shape, dtype=bool)
    # Strictly below the previous sample and not above the next one, so a flat minimum is counted once.
    is_min[1:] &= gap[1:] < gap[:-1]
    is_min[:-1] &= gap[:-1] <= gap[1:]
    return is_min


def _gap_basin(gap: np.ndarray, index: int) -> tuple[int, int]:
    """Grid range around a minimum of ``gap`` that is bounded by the nearest local maxima on each side."""
    left = index
    while left > 0 and gap[left - 1] >= gap[left]:
        left -= 1
    right = index
    while right < gap.size - 1 and gap[right + 1] >= gap[right]:
        right += 1
    return left, right


# A diabatic passage is fully diabatic over this fraction of the way (along the control grid) from the anticrossing
# minimum to the bounding gap maximum. From there to the maximum the exponents are blended smoothly into the adiabatic
# ones.
DIABATIC_BLEND_START = 0.5


def _diabatic_weight(gap: np.ndarray, index: int, left: int, right: int) -> np.ndarray:
    """
    Diabatic weight on ``gap[left:right + 1]`` around the anticrossing at ``index``.

    The weight is 1 near the anticrossing and decreases smoothly (C1 smoothstep along the grid) to 0 at the bounding
    gap maxima. Edges at the ends of the grid are not maxima of the gap, so the weight is not reduced towards them.
    """
    weight = np.ones(right - left + 1)
    for edge in (left, right):
        if edge in (0, gap.size - 1) or edge == index:
            continue
        lo, hi = min(edge, index), max(edge, index)
        distance = np.abs(np.arange(lo, hi + 1) - index) / abs(edge - index)
        t = np.clip((distance - DIABATIC_BLEND_START) / (1.0 - DIABATIC_BLEND_START), 0.0, 1.0)
        side = weight[lo - left : hi - left + 1]
        np.minimum(side, 1.0 - t * t * (3.0 - 2.0 * t), out=side)
    return weight


@dataclass(frozen=True)
class DiabaticList:
    """
    Adiabatic fraction of the transitions out of the occupied eigenstate along the control grid.

    Only these transitions enter the metric, so the list stores one row per grid point instead of a full
    (n_points, dim, dim) matrix.

    Attributes
    ----------
    occupied : np.ndarray
        Index of the occupied instantaneous eigenstate at each grid point, with shape (n_points,).
    adiabatic_fraction : np.ndarray
        Shape (n_points, dim). Entry ``[i, n]`` is ``1`` if the transition from ``occupied[i]`` to state ``n`` is
        adiabatic at grid point ``i``, ``0`` if it is diabatic, and a value in between where the two are blended. The
        entry of the occupied state itself is ``-1``.
    """

    occupied: np.ndarray
    adiabatic_fraction: np.ndarray

    def dense(self) -> np.ndarray:
        """
        Expand to the full transition matrix with shape (n_points, dim, dim), e.g. for plotting.

        Entry ``[i, m, n]`` is the adiabatic fraction of the transition between ``m`` and ``n`` if one of them is the
        occupied state, and ``-1`` otherwise (including the diagonal).
        """
        n_points, dim = self.adiabatic_fraction.shape
        points = np.arange(n_points)
        matrix = -np.ones((n_points, dim, dim), dtype=float)
        matrix[points, self.occupied, :] = self.adiabatic_fraction
        matrix[points, :, self.occupied] = self.adiabatic_fraction
        return matrix


def build_dia_list(
    energies: np.ndarray, initial_state: int, final_state: int, state_map: np.ndarray | None = None
) -> DiabaticList:
    """
    Build the adiabatic/diabatic transition mask along the control grid.

    The occupied instantaneous eigenstate starts in ``initial_state`` and is followed along the grid, also through true
    level crossings given by ``state_map``, where it changes its energy index without a diabatic passage. At every
    anticrossing (local minimum of the gap) between the occupied state ``k`` and its neighbour ``k + d``, with
    ``d = sign(final_state - initial_state)``, the passage is diabatic and the occupied state becomes ``k + d``. All
    other anticrossings, including those back away from ``final_state`` and all anticrossings after reaching it, are
    passed adiabatically. A diabatic passage only acts inside its anticrossing: between the closest maxima of the
    corresponding gap, and the exponents are blended smoothly into the adiabatic ones towards those maxima (see
    ``DIABATIC_BLEND_START``). Overlapping anticrossings, as in a fan of levels crossing at one point, are passed
    together: if the next gap has no minimum ahead but the grid point of a passage is still inside its last
    anticrossing, that anticrossing is passed at the same point.

    Parameters
    ----------
    energies : np.ndarray
        Eigenenergies sorted in increasing order, with shape (n_points, dim), sampled in the sweep direction.
    initial_state, final_state : int
        Indices of the instantaneous eigenstates at the start and at the end of the sweep.
    state_map : np.ndarray | None
        True level crossings, as returned by ``crossing_map``. The gap minima of a true crossing are not anticrossings.
        ``None`` means that there are no true crossings.

    Returns
    -------
    DiabaticList
        The occupied state at each grid point and the adiabatic fraction of the transitions out of it, with shapes
        (n_points,) and (n_points, dim). ``DiabaticList.dense()`` expands it to a (n_points, dim, dim) matrix.

    Raises
    ------
    InvalidControlParameterError
        If ``final_state`` cannot be reached through anticrossings that lead towards it.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or energies.shape[0] < 2:
        raise ValidationError("energies must have shape (n_points, dim) with at least two points.")
    n_points, dim = energies.shape
    for label, state in (("initial_state", initial_state), ("final_state", final_state)):
        if not isinstance(state, (int, np.integer)) or isinstance(state, bool) or not 0 <= state < dim:
            raise ValidationError(f"{label} must be an integer in [0, {dim - 1}].")

    if state_map is None:
        state_map = np.tile(np.arange(dim), (n_points, 1))
    elif state_map.shape != energies.shape:
        raise ValidationError("state_map must have the same shape as energies.")

    gaps = np.diff(energies, axis=1)  # gaps[:, j] = E_{j+1} - E_j
    gap_minima = np.stack([_gap_minima(gaps[:, j]) for j in range(dim - 1)], axis=1) if dim > 1 else None
    step = 1 if final_state > initial_state else -1

    # The gap between two truly crossing states has its minimum at one of the two grid points around the crossing.
    if gap_minima is not None:
        for point in np.flatnonzero(np.any(state_map != np.arange(dim), axis=1)):
            moved = state_map[point] != np.arange(dim)
            crossing_gaps = moved[:-1] & moved[1:]
            gap_minima[point - 1 : point + 1, crossing_gaps] = False

    def overlaps(gap_index: int, index: int) -> bool:
        # The anticrossing of this gap overlaps with a passage at ``index``: its last minimum is behind ``index``,
        # no minimum lies ahead and ``index`` is still inside the anticrossing (no maximum in between).
        assert gap_minima is not None
        minima = np.flatnonzero(gap_minima[:, gap_index])
        if minima.size == 0 or minima[-1] >= index:
            return False
        return _gap_basin(gaps[:, gap_index], int(minima[-1]))[1] >= index

    # Follow the occupied state along the grid. It changes at the minimum of each diabatic anticrossing.
    occupied = np.empty(n_points, dtype=int)
    passages: list[tuple[int, int, int]] = []  # (grid index, state before, state after)
    state = initial_state
    for i in range(n_points):
        if i > 0:
            state = int(state_map[i, state])
        passed = False
        while state != final_state:
            assert gap_minima is not None
            gap_index = min(state, state + step)
            if not (gap_minima[i, gap_index] or (passed and overlaps(gap_index, i))):
                break
            passages.append((i, state, state + step))
            state += step
            passed = True
        occupied[i] = state

    if state != final_state:
        raise InvalidControlParameterError(
            f"final_state={final_state} cannot be reached from initial_state={initial_state}: following the "
            f"anticrossings towards final_state, the sweep ends in state {state}. No anticrossing between states "
            f"{state} and {state + step} follows."
        )

    # Every transition from the occupied state is adiabatic unless it belongs to a diabatic anticrossing.
    fraction = np.ones((n_points, dim), dtype=float)
    fraction[np.arange(n_points), occupied] = -1

    for index, before, after in passages:
        gap = gaps[:, min(before, after)]
        left, right = _gap_basin(gap, index)
        window = np.arange(left, right + 1)
        adiabatic_fraction = 1.0 - _diabatic_weight(gap, index, left, right)
        # Inside the anticrossing the occupied state is one of the pair; the transition is to the other one.
        for source, target in ((before, after), (after, before)):
            on_source = occupied[window] == source
            fraction[window[on_source], target] = adiabatic_fraction[on_source]

    return DiabaticList(occupied=occupied, adiabatic_fraction=fraction)


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


# The controller and the number of imported modules when it was created.
_controller_cache: dict[str, Any] = {"controller": None, "modules": -1}


def _threadpool_controller() -> ThreadpoolController:
    # Inspecting the loaded BLAS libraries takes about 10 ms, so the controller is reused. It is re-created when modules
    # have been imported since, because an import (e.g. torch) can load another BLAS library, which a controller created
    # earlier would not limit. Counting sys.modules costs about a microsecond.
    modules = len(sys.modules)
    if _controller_cache["controller"] is None or _controller_cache["modules"] != modules:
        _controller_cache["controller"] = ThreadpoolController()
        _controller_cache["modules"] = modules
    return _controller_cache["controller"]


@contextmanager
def limit_blas_threads(dimension: int) -> Generator[None, None, None]:
    """
    Run BLAS/LAPACK single-threaded inside the context if ``dimension`` is small, otherwise leave it untouched.

    For small matrices, the threading overhead of BLAS outweighs its gain. The limit is set through ``threadpoolctl``,
    which changes the thread count of the BLAS libraries themselves: while the context is active it applies to the
    whole process, so BLAS calls made from other threads at the same time also run single-threaded. BLAS libraries
    loaded by a later import are included; a library loaded without an import (e.g. through ``ctypes``) is not.
    """
    if dimension <= SMALL_MATRIX_DIMENSION:
        with _threadpool_controller().limit(limits=1, user_api="blas"):
            yield
    else:
        yield


# -----------------------------------
# Finite differences
# -----------------------------------
@cache
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


# Size of the pieces in which uniform_grid_derivative forms its weighted sums.
_DERIVATIVE_CHUNK_BYTES = 2**22


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
    # Rows per piece, so that the temporary ``weight * samples[...]`` stays small for large matrices.
    rows = max(1, _DERIVATIVE_CHUNK_BYTES // max(1, derivative[0].nbytes))
    for block in np.split(np.arange(n), boundaries):
        first, last = int(block[0]), int(block[-1]) + 1
        offsets = tuple(int(relative[first]) + j for j in range(n_points))
        weights = finite_difference_weights(offsets)
        for lo in range(first, last, rows):
            hi = min(lo + rows, last)
            for offset, weight in zip(offsets, weights, strict=True):
                derivative[lo:hi] += weight * samples[lo + offset : hi + offset]

    derivative /= step
    return derivative
