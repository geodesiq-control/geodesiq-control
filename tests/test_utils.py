from geodesiq._utils import build_diab


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
