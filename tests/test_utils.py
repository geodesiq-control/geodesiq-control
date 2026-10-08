import numpy as np
import pytest
from threadpoolctl import threadpool_info

from geodesiq._utils import (
    SMALL_MATRIX_DIMENSION,
    build_dia_list,
    finite_difference_weights,
    limit_blas_threads,
    uniform_grid_derivative,
)
from geodesiq.exceptions import ValidationError

# ---------------------------------------------------------------------------
# build_dia_list()
# ---------------------------------------------------------------------------


def three_level_energies(x: np.ndarray, gap_01: np.ndarray) -> np.ndarray:
    """Sorted energies whose gap between states 1 and 2 has two anticrossings, at x = -1 and x = 1."""
    gap_12 = 0.1 + (x ** 2 - 1) ** 2
    return np.stack([np.zeros_like(x), gap_01, gap_01 + gap_12], axis=1)


def occupied_states(dia_list: np.ndarray) -> np.ndarray:
    """The occupied state is the only one with a considered transition to every other state."""
    considered = (dia_list >= 0).sum(axis=2)
    return np.argmax(considered == dia_list.shape[1] - 1, axis=1)


class TestBuildDiaList:
    x = np.linspace(-2.0, 2.0, 401)

    def test_same_pair_is_only_crossed_diabatically_towards_the_final_state(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=2)

        assert dia_list.shape == (self.x.size, 3, 3)
        occupied = occupied_states(dia_list)
        np.testing.assert_array_equal(occupied, np.where(self.x < -1.0 - 1e-9, 1, 2))
        # Diabatic up to the gap maximum at x = 0, adiabatic at the second anticrossing at x = 1.
        assert np.all(dia_list[self.x <= 0.0, 1, 2] == 0)
        assert np.all(dia_list[self.x > 0.0, 1, 2] == 1)
        np.testing.assert_array_equal(dia_list[:, 1, 2], dia_list[:, 2, 1])

    def test_transitions_between_unoccupied_states_are_not_considered(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=2)

        after = self.x > -1.0 - 1e-9
        assert np.all(dia_list[after, 0, 1] == -1)
        assert np.all(dia_list[after, 0, 2] == 1)
        assert np.all(dia_list[~after, 0, 2] == -1)
        assert all(np.all(dia_list[:, i, i] == -1) for i in range(3))

    def test_downward_transfer_crosses_the_first_anticrossing(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=2, final_state=1)

        np.testing.assert_array_equal(occupied_states(dia_list), np.where(self.x < -1.0 - 1e-9, 2, 1))
        assert np.all(dia_list[self.x > 0.0, 1, 2] == 1)

    def test_multi_step_transfer_follows_the_anticrossings_in_order(self):
        energies = three_level_energies(self.x, gap_01=1.0 + (self.x + 1.5) ** 2)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2)

        occupied = occupied_states(dia_list)
        assert occupied[0] == 0
        assert occupied[-1] == 2
        assert np.all(np.diff(occupied) >= 0)
        np.testing.assert_allclose(self.x[np.flatnonzero(np.diff(occupied)) + 1], [-1.5, -1.0], atol=1e-9)

    def test_overlapping_anticrossings_are_passed_together(self):
        # The minimum of the 1-2 gap (x = -0.1) lies just before the one of the 0-1 gap (x = 0.1), inside its flank.
        energies = np.stack(
            [np.zeros_like(self.x), 1.0 + (self.x - 0.1) ** 2, 2.0 + (self.x - 0.1) ** 2 + (self.x + 0.1) ** 2],
            axis=1,
        )

        dia_list = build_dia_list(energies, initial_state=0, final_state=2)

        occupied = occupied_states(dia_list)
        np.testing.assert_array_equal(occupied, np.where(self.x < 0.1 - 1e-9, 0, 2))
        assert np.all(dia_list[self.x < 0.1 - 1e-9, 0, 1] == 0)
        assert np.all(dia_list[self.x > 0.1 - 1e-9, 1, 2] == 0)

    def test_a_later_anticrossing_is_preferred_over_an_overlapping_one(self):
        energies = three_level_energies(self.x, gap_01=1.0 + (self.x + 0.5) ** 2)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2)

        occupied = occupied_states(dia_list)
        np.testing.assert_allclose(self.x[np.flatnonzero(np.diff(occupied)) + 1], [-0.5, 1.0], atol=1e-9)

    def test_equal_states_give_purely_adiabatic_transitions_of_that_state(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=1)

        assert np.all(dia_list[:, 1, [0, 2]] == 1)
        assert np.all(dia_list[:, 0, 2] == -1)

    @pytest.mark.parametrize(("initial_state", "final_state"), [(-1, 0), (0, 3), (True, 1)])
    def test_invalid_state_indices_raise(self, initial_state, final_state):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        with pytest.raises(ValidationError, match="must be an integer"):
            build_dia_list(energies, initial_state=initial_state, final_state=final_state)


# -----------------------------------
# finite differences
# -----------------------------------
class TestFiniteDifferences:
    @pytest.mark.parametrize(
        ("offsets", "expected"),
        [((-1, 0, 1), [-0.5, 0.0, 0.5]), ((0, 1, 2), [-1.5, 2.0, -0.5]), ((-2, -1, 0), [0.5, -2.0, 1.5]),
         ((-2, -1, 0, 1, 2), [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12])],
    )
    def test_known_stencils(self, offsets, expected):
        np.testing.assert_allclose(finite_difference_weights(offsets), expected, atol=1e-15)

    @pytest.mark.parametrize("n_points", [2, 3, 5, 9])
    def test_exact_for_polynomials_of_stencil_degree(self, n_points: int):
        x = np.linspace(-1.0, 2.0, 21)
        coefficients = np.arange(1, n_points + 1, dtype=float)
        values = np.polynomial.polynomial.polyval(x, coefficients)
        expected = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients))

        derivative = uniform_grid_derivative(values[:, None, None], float(x[1] - x[0]), n_points)

        np.testing.assert_allclose(derivative[:, 0, 0], expected, rtol=1e-9, atol=1e-9)

    def test_high_order_is_accurate_for_smooth_functions(self):
        x = np.linspace(-1.0, 2.0, 257)
        derivative = uniform_grid_derivative(np.sin(3 * x)[:, None, None], float(x[1] - x[0]), 9)

        np.testing.assert_allclose(derivative[:, 0, 0], 3 * np.cos(3 * x), atol=1e-12)

    def test_weights_are_cached_and_read_only(self):
        weights = finite_difference_weights((-1, 0, 1))

        assert finite_difference_weights((-1, 0, 1)) is weights
        with pytest.raises(ValueError):
            weights[0] = 1.0

    @pytest.mark.parametrize("offsets", [(0,), (1, 1)])
    def test_invalid_stencils_raise(self, offsets):
        with pytest.raises(ValidationError, match="at least two distinct offsets"):
            finite_difference_weights(offsets)

    @pytest.mark.parametrize("n_points", [1, 6])
    def test_invalid_number_of_points_raises(self, n_points: int):
        with pytest.raises(ValidationError, match="n_points must be between 2"):
            uniform_grid_derivative(np.zeros((5, 2, 2)), 0.1, n_points)


# -----------------------------------
# BLAS threads
# -----------------------------------
def _blas_threads() -> list[int]:
    return [info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"]


class TestLimitBlasThreads:
    def test_small_dimension_runs_single_threaded_and_restores(self):
        before = _blas_threads()

        with limit_blas_threads(SMALL_MATRIX_DIMENSION):
            assert all(n == 1 for n in _blas_threads())

        assert _blas_threads() == before

    def test_large_dimension_is_left_untouched(self):
        before = _blas_threads()

        with limit_blas_threads(SMALL_MATRIX_DIMENSION + 1):
            assert _blas_threads() == before
