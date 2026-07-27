from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

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

    def add(self, name: str, value: bool = False, parent: str | None = None,
            parents: list[str] | tuple[str, ...] | set[str] | None = None, ) -> None:
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


def build_diab(initial_state: int, final_state: int, dim: int) -> np.ndarray:
    """Build the adiabatic/diabatic transition mask with validated indices."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError("dim must be a positive integer.")
    for label, state in (("initial_state", initial_state), ("final_state", final_state)):
        if not isinstance(state, int) or isinstance(state, bool) or not 0 <= state < dim:
            raise ValueError(f"{label} must be an integer in [0, {dim - 1}].")
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
