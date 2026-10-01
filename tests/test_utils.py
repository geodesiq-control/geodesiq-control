import numpy as np
import pytest
from threadpoolctl import threadpool_info

from geodesiq._utils import (
    SMALL_MATRIX_DIMENSION,
    build_diab,
    finite_difference_weights,
    limit_blas_threads,
    uniform_grid_derivative,
)
from geodesiq.exceptions import ValidationError

# ---------------------------------------------------------------------------
# build_diab()
# ---------------------------------------------------------------------------


class TestBuildDiab:
    def test_build_diab_sets_zero_inside_transition_window(self):
        diad = build_diab(initial_state=1, final_state=3, dim=5)

        assert diad.shape == (5, 5)
        assert diad[1, 2] == 0
        assert diad[2, 3] == 0
        assert diad[1, 3] == 0

    def test_build_diab_sets_one_outside_transition_window_and_minus_one_diagonal(self):
        diad = build_diab(initial_state=1, final_state=3, dim=5)

        assert diad[0, 4] == 1
        assert diad[0, 1] == 1
        assert diad[4, 3] == 1
        assert all(diad[i, i] == -1 for i in range(5))


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
