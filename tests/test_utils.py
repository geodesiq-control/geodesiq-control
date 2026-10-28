import sys
import tracemalloc
import types

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from geodesiq._utils import (
    DIABATIC_BLEND_START,
    SMALL_MATRIX_DIMENSION,
    _threadpool_controller,
    build_dia_list,
    crossing_map,
    degenerate_clusters,
    finite_difference_weights,
    follow_state,
    limit_blas_threads,
    rotate_degenerate_subspaces,
    stack_samples,
    traceless,
    uniform_grid_derivative,
)
from geodesiq.exceptions import InvalidControlParameterError, ValidationError

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

        dia_list = build_dia_list(energies, initial_state=1, final_state=2).dense()

        assert dia_list.shape == (self.x.size, 3, 3)
        occupied = occupied_states(dia_list)
        np.testing.assert_array_equal(occupied, np.where(self.x < -1.0 - 1e-9, 1, 2))
        # Diabatic around the first anticrossing, blended up to the gap maximum at x = 0, adiabatic at the second one.
        assert np.all(dia_list[self.x <= -0.5, 1, 2] == 0)
        assert np.all(dia_list[self.x >= 0.0, 1, 2] == 1)
        np.testing.assert_array_equal(dia_list[:, 1, 2], dia_list[:, 2, 1])

    def test_transitions_between_unoccupied_states_are_not_considered(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=2).dense()

        after = self.x > -1.0 - 1e-9
        assert np.all(dia_list[after, 0, 1] == -1)
        assert np.all(dia_list[after, 0, 2] == 1)
        assert np.all(dia_list[~after, 0, 2] == -1)
        assert all(np.all(dia_list[:, i, i] == -1) for i in range(3))

    def test_diabatic_and_adiabatic_exponents_are_blended_smoothly_towards_the_gap_maximum(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=2).dense()

        blend = dia_list[(self.x >= -1.0) & (self.x <= 0.0), 1, 2]
        assert blend[0] == 0 and blend[-1] == 1
        assert np.all(np.diff(blend) >= 0)
        assert np.max(np.diff(blend)) < 0.05
        # The blend starts DIABATIC_BLEND_START of the way from the anticrossing (x = -1) to the gap maximum (x = 0).
        blend_start = -1.0 + DIABATIC_BLEND_START
        assert np.all(dia_list[(self.x > -1.0) & (self.x < blend_start - 1e-9), 1, 2] == 0)
        assert np.all(dia_list[(self.x > blend_start + 1e-9) & (self.x < 0.0), 1, 2] > 0)

    def test_no_blending_towards_the_ends_of_the_grid(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=2).dense()

        assert np.all(dia_list[self.x < -1.0, 1, 2] == 0)

    @pytest.mark.parametrize("x", [np.linspace(1.0, 10.0, 101), np.linspace(-10.0, -1.0, 101)])
    def test_passage_at_an_end_of_the_grid_needs_the_override(self, x):
        # Two-level sweep that does not contain the anticrossing at x = 0: the gap minimum lies at the start (x >= 1)
        # or at the end (x <= -1) of the grid. Only with allow_endpoint_passages is the transfer made there, fully
        # diabatic.
        gap = 2.0 * np.sqrt(1.0 + x**2)
        energies = np.stack([-gap / 2, gap / 2], axis=1)

        with pytest.raises(InvalidControlParameterError, match="allow_endpoint_passages=True"):
            build_dia_list(energies, initial_state=0, final_state=1)

        dia_list = build_dia_list(energies, initial_state=0, final_state=1, allow_endpoint_passages=True)

        assert np.all(dia_list.adiabatic_fraction[dia_list.adiabatic_fraction >= 0] == 0)

    def test_gap_rising_at_the_start_does_not_move_the_passage_there(self):
        # Review C-01 (R1): the gap rises up to x = 0.5 and the anticrossing is at x = (1 + sqrt(5)) / 2. The passage
        # used to be made at x = 0, and the real anticrossing was then passed adiabatically.
        x = np.linspace(0.0, 2.5, 1025)
        gap = 2.0 * np.sqrt((1.0 + x - x**2) ** 2 + 0.05**2)
        energies = np.stack([-gap / 2, gap / 2], axis=1)

        dia_list = build_dia_list(energies, initial_state=0, final_state=1)

        passage = x[np.argmax(dia_list.occupied == 1)]
        assert passage == pytest.approx((1.0 + np.sqrt(5.0)) / 2.0, abs=2 * (x[1] - x[0]))
        assert dia_list.adiabatic_fraction[0, 1] == 1
        # With the override, the start of the grid is again the (first) passage.
        assert build_dia_list(energies, 0, 1, allow_endpoint_passages=True).occupied[0] == 1

    @pytest.mark.parametrize("reverse", [False, True])
    def test_first_anticrossing_is_passed_not_the_deepest(self, reverse):
        # The same pair anticrosses at x = -1 (gap 0.24) and x = 1 (gap 0.11). The first one along the sweep is passed
        # diabatically, the second one adiabatically, in either sweep direction (C-01: not the deepest minimum).
        x = self.x[::-1] if reverse else self.x
        gap = (x**2 - 1) ** 2 + 0.3 - 0.25 * (x + 2) / 4
        energies = np.stack([-gap / 2, gap / 2], axis=1)

        dia_list = build_dia_list(energies, initial_state=0, final_state=1)

        first, second = (1.0, -1.0) if reverse else (-1.0, 1.0)
        passage = x[np.flatnonzero(np.diff(dia_list.occupied)) + 1]
        np.testing.assert_allclose(passage, [first], atol=0.02)
        transition = dia_list.adiabatic_fraction[np.arange(x.size), 1 - dia_list.occupied]
        assert np.all(transition[np.abs(x - first) < 0.1] == 0)
        assert np.all(transition[np.abs(x - second) < 0.1] == 1)

    def test_first_anticrossing_keeps_the_later_ones_available(self):
        # Gap 0-1 anticrosses at x = -1.5 (shallow) and x = 0.5 (deep), gap 1-2 at x = -0.5 only. Passing the first 0-1
        # anticrossing reaches the 1-2 one; passing the deepest 0-1 anticrossing at x = 0.5 would leave none for 1-2.
        gap_01 = ((self.x + 1.5) * (self.x - 0.5)) ** 2 + 0.2 - 0.05 * (self.x + 1.5)
        gap_12 = 0.5 + (self.x + 0.5) ** 2
        energies = np.stack([np.zeros_like(self.x), gap_01, gap_01 + gap_12], axis=1)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2)

        passages = self.x[np.flatnonzero(np.diff(dia_list.occupied)) + 1]
        np.testing.assert_allclose(passages, [-1.5, -0.5], atol=0.02)

    def test_rounding_dips_do_not_count_but_resolved_ones_do(self):
        x = np.linspace(0.0, 3.0, 301)
        noise = np.random.default_rng(1).normal(scale=4e-16, size=x.size)
        gap = np.where(x < 1.0, 2.0, 2.0 - 1.9 * np.sin(np.pi * np.clip(x - 1.0, 0.0, 2.0) / 2.0) ** 2) + noise
        energies = np.stack([-gap / 2, gap / 2], axis=1)

        np.testing.assert_allclose(x[np.flatnonzero(np.diff(build_dia_list(energies, 0, 1).occupied)) + 1], [2.0])
        # Without the tolerance, the first rounding dip is taken as the anticrossing.
        assert x[np.argmax(build_dia_list(energies, 0, 1, gap_rtol=0.0).occupied == 1)] < 1.0

    def test_downward_transfer_crosses_the_first_anticrossing(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=2, final_state=1).dense()

        np.testing.assert_array_equal(occupied_states(dia_list), np.where(self.x < -1.0 - 1e-9, 2, 1))
        assert np.all(dia_list[self.x > 0.0, 1, 2] == 1)

    def test_multi_step_transfer_follows_the_anticrossings_in_order(self):
        energies = three_level_energies(self.x, gap_01=1.0 + (self.x + 1.5) ** 2)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2).dense()

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

        dia_list = build_dia_list(energies, initial_state=0, final_state=2).dense()

        occupied = occupied_states(dia_list)
        np.testing.assert_array_equal(occupied, np.where(self.x < 0.1 - 1e-9, 0, 2))
        assert np.all(dia_list[self.x < 0.1 - 1e-9, 0, 1] == 0)
        assert np.all(dia_list[self.x > 0.1 - 1e-9, 1, 2] == 0)

    def test_a_later_anticrossing_is_preferred_over_an_overlapping_one(self):
        energies = three_level_energies(self.x, gap_01=1.0 + (self.x + 0.5) ** 2)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2).dense()

        occupied = occupied_states(dia_list)
        np.testing.assert_allclose(self.x[np.flatnonzero(np.diff(occupied)) + 1], [-0.5, 1.0], atol=1e-9)

    def test_equal_states_give_purely_adiabatic_transitions_of_that_state(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        dia_list = build_dia_list(energies, initial_state=1, final_state=1).dense()

        assert np.all(dia_list[:, 1, [0, 2]] == 1)
        assert np.all(dia_list[:, 0, 2] == -1)

    @pytest.mark.parametrize(("initial_state", "final_state"), [(-1, 0), (0, 3), (True, 1)])
    def test_invalid_state_indices_raise(self, initial_state, final_state):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))

        with pytest.raises(ValidationError, match="must be an integer"):
            build_dia_list(energies, initial_state=initial_state, final_state=final_state)

    def test_compact_list_stores_one_row_per_point(self):
        energies = three_level_energies(self.x, gap_01=1.0 + (self.x + 1.5) ** 2)

        dia_list = build_dia_list(energies, initial_state=0, final_state=2)
        dense = dia_list.dense()

        assert dia_list.occupied.shape == (self.x.size,)
        assert dia_list.adiabatic_fraction.shape == (self.x.size, 3)
        np.testing.assert_array_equal(dia_list.occupied, occupied_states(dense))
        points = np.arange(self.x.size)
        assert np.all(dia_list.adiabatic_fraction[points, dia_list.occupied] == -1)
        np.testing.assert_array_equal(dense[points, dia_list.occupied, :], dia_list.adiabatic_fraction)
        np.testing.assert_array_equal(dense, dense.transpose(0, 2, 1))

    def test_large_dimension_does_not_allocate_a_dense_matrix(self):
        # The dense (n_points, dim, dim) list took 328 MB for dim = 200 and n_points = 1025 (C-16).
        n_points, dim = 1025, 200
        x = np.linspace(-1.0, 1.0, n_points)
        energies = np.sort(np.add.outer(x ** 2, np.arange(dim, dtype=float)), axis=1)

        tracemalloc.start()
        try:
            dia_list = build_dia_list(energies, initial_state=0, final_state=0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert dia_list.adiabatic_fraction.shape == (n_points, dim)
        assert peak < 10 * n_points * dim * 8

    def test_true_crossing_is_followed_and_not_passed_diabatically(self):
        # State a = x crosses the flat level b = 0 at x = 0 (state_map) and then anticrosses c = 1 + (x - 1)^2 / 10.
        x = self.x
        a, b, c = x, np.zeros_like(x), 1.0 + (x - 1.0) ** 2 / 10
        energies = np.sort(np.stack([a, b, c], axis=1), axis=1)
        state_map = np.tile(np.arange(3), (x.size, 1))
        crossing = np.flatnonzero(x > 0)[0]
        state_map[crossing, :2] = [1, 0]

        dia_list = build_dia_list(energies, initial_state=0, final_state=2, state_map=state_map)

        assert dia_list.occupied[crossing - 1] == 0 and dia_list.occupied[crossing] == 1
        # Without the state map, the gap minimum of the crossing would be a diabatic passage.
        assert np.all(dia_list.dense()[np.abs(x) < 0.2, 0, 1] == 1)
        assert dia_list.occupied[-1] == 2

    def test_state_map_must_match_the_energies(self):
        energies = three_level_energies(self.x, gap_01=np.full_like(self.x, 5.0))
        with pytest.raises(ValidationError, match="state_map"):
            build_dia_list(energies, initial_state=1, final_state=2, state_map=np.zeros((3, 3), dtype=int))


# ---------------------------------------------------------------------------
# Level crossings: degenerate_clusters(), rotate_degenerate_subspaces(), crossing_map(), follow_state()
# ---------------------------------------------------------------------------
def two_level_vectors(x: np.ndarray, coupling: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues, eigenvectors and |<m|sz|n>| of x sz + coupling sx."""
    sz = np.diag([1.0, -1.0])
    energies, vectors = np.linalg.eigh(x[:, None, None] * sz + coupling * np.array([[0.0, 1.0], [1.0, 0.0]]))
    return energies, vectors, np.abs(vectors.transpose(0, 2, 1) @ sz @ vectors)


class TestLevelCrossings:
    def test_degenerate_clusters_are_found_relative_to_the_bandwidth(self):
        energies = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [1e-14, 2e-14, 3e-14]]) * 1e6

        assert degenerate_clusters(energies, rtol=1e-12) == [(0, 0, 1)]
        assert degenerate_clusters(np.zeros((1, 3)), rtol=1e-12) == [(0, 0, 2)]

    def test_rotation_diagonalizes_dh_in_a_degenerate_subspace(self):
        vectors = np.eye(2)[None] @ (np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2))
        partial_H = np.diag([1.0, -1.0])

        rotate_degenerate_subspaces(vectors, partial_H, [(0, 0, 1)])

        elements = vectors[0].T @ partial_H @ vectors[0]
        assert abs(elements[0, 1]) < 1e-15

    def test_uncoupled_crossing_is_followed(self):
        x = np.linspace(-1.0, 1.0, 64)  # 0 is not on the grid
        _, vectors, elements = two_level_vectors(x, 0.0)

        state_map = crossing_map(vectors, elements, rtol=1e-8, chunk=7)

        changed = np.flatnonzero(np.any(state_map != np.arange(2), axis=1))
        np.testing.assert_array_equal(changed, [32])
        np.testing.assert_array_equal(state_map[32], [1, 0])
        np.testing.assert_array_equal(follow_state(state_map, 0), np.where(x < 0, 0, 1))

    def test_coupled_swap_keeps_the_energy_order(self):
        x = np.linspace(-1.0, 1.0, 64)
        _, vectors, elements = two_level_vectors(x, 1e-6)

        state_map = crossing_map(vectors, elements, rtol=1e-8, chunk=7)

        np.testing.assert_array_equal(state_map, np.tile(np.arange(2), (64, 1)))


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

    def test_result_does_not_depend_on_the_chunk_size(self, monkeypatch):
        x = np.linspace(-1.0, 2.0, 65)
        samples = np.exp(1j * np.multiply.outer(x, np.arange(9.0).reshape(3, 3)))
        reference = uniform_grid_derivative(samples, float(x[1] - x[0]), 9)

        # One row per piece (C-17): the weighted sums are formed in pieces of the interior block.
        monkeypatch.setattr("geodesiq._utils._DERIVATIVE_CHUNK_BYTES", 1)
        chunked = uniform_grid_derivative(samples, float(x[1] - x[0]), 9)

        np.testing.assert_array_equal(chunked, reference)


class TestStackSamples:
    def test_matches_np_stack(self):
        points = np.linspace(0.0, 1.0, 5)
        stacked = stack_samples(lambda x: np.array([[x, 1.0], [1.0, -x]]), points)

        np.testing.assert_array_equal(stacked, np.stack([np.array([[x, 1.0], [1.0, -x]]) for x in points]))

    def test_integer_samples_become_float(self):
        assert stack_samples(lambda x: np.eye(2, dtype=int), np.arange(3.0)).dtype == np.float64

    def test_dtype_is_widened_by_a_later_complex_sample(self):
        stacked = stack_samples(lambda x: np.array([[x]]) if x < 1 else np.array([[1j * x]]), np.arange(3.0))

        assert np.iscomplexobj(stacked)
        np.testing.assert_array_equal(stacked[:, 0, 0], [0.0, 1j, 2j])

    def test_inconsistent_shapes_raise(self):
        with pytest.raises(ValidationError, match="same shape"):
            stack_samples(lambda x: np.eye(2 if x < 1 else 3), np.arange(3.0))


class TestTraceless:
    def test_single_matrix(self):
        matrix = np.array([[3.0, 1j, 0.0], [-1j, 1.0, 2.0], [0.0, 2.0, 5.0]])

        result = traceless(matrix)

        np.testing.assert_allclose(result, matrix - 3.0 * np.eye(3))
        assert abs(np.trace(result)) < 1e-14

    def test_stack_removes_each_trace_and_keeps_off_diagonals(self):
        offsets = np.array([0.0, 1e8, -2.5])
        stack = np.stack([offset * np.eye(2) + np.array([[0.3, 1j], [-1j, -0.1]]) for offset in offsets])

        result = traceless(stack)

        np.testing.assert_allclose(result, np.broadcast_to([[0.2, 1j], [-1j, -0.2]], stack.shape), atol=1e-8)
        np.testing.assert_array_equal(result[:, 0, 1], stack[:, 0, 1])

    def test_real_input_stays_real(self):
        assert traceless(np.diag([1.0, 2.0])).dtype == np.float64


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

    def test_controller_is_reused_until_a_module_is_imported(self, monkeypatch):
        # The controller is a snapshot of the loaded BLAS libraries, so it was cached forever and missed libraries that
        # a later import (e.g. torch) loads (C-20).
        first = _threadpool_controller()
        assert _threadpool_controller() is first

        monkeypatch.setitem(sys.modules, "geodesiq_test_new_module", types.ModuleType("geodesiq_test_new_module"))

        refreshed = _threadpool_controller()
        assert refreshed is not first
        assert _threadpool_controller() is refreshed
        with limit_blas_threads(SMALL_MATRIX_DIMENSION):
            assert all(n == 1 for n in _blas_threads())
