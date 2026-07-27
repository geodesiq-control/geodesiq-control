import pytest

from geodesiq._utils import Flags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_flags():
    """A Flags instance with three flags: A → B → C."""
    f = Flags()
    f.add("A", value=True)
    f.add("B", value=True, parent="A")
    f.add("C", value=True, parent="B")
    return f


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_single_flag_default_value(self):
        f = Flags()
        f.add("A")
        assert f["A"] is False

    def test_add_single_flag_custom_value(self):
        f = Flags()
        f.add("A", value=True)
        assert f["A"] is True

    def test_add_flag_with_valid_parent(self):
        f = Flags()
        f.add("A", value=True)
        f.add("B", value=True, parent="A")
        assert f["B"] is True

    def test_add_duplicate_raises(self):
        f = Flags()
        f.add("A")
        with pytest.raises(KeyError, match="already registered"):
            f.add("A")

    def test_add_unknown_parent_raises(self):
        f = Flags()
        with pytest.raises(KeyError, match="does not exist"):
            f.add("B", parent="A")

    def test_add_flag_with_multiple_parents(self):
        f = Flags()
        f.add("A", value=True)
        f.add("B", value=True)
        f.add("C", value=True, parents=["A", "B"])
        assert f["C"] is True

    def test_add_with_parent_and_parents_raises(self):
        f = Flags()
        f.add("A", value=True)
        with pytest.raises(ValueError, match="either 'parent' or 'parents'"):
            f.add("B", value=True, parent="A", parents=["A"])


# ---------------------------------------------------------------------------
# get() / __getitem__
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing_flag(self, simple_flags):
        assert simple_flags.get("A") is True

    def test_getitem_existing_flag(self, simple_flags):
        assert simple_flags["A"] is True

    def test_get_nonexistent_raises(self):
        f = Flags()
        with pytest.raises(KeyError, match="not registered"):
            f.get("missing")

    def test_getitem_nonexistent_raises(self):
        f = Flags()
        with pytest.raises(KeyError):
            _ = f["missing"]


# ---------------------------------------------------------------------------
# set() / __setitem__
# ---------------------------------------------------------------------------


class TestSet:
    def test_set_value(self):
        f = Flags()
        f.add("A", value=True)
        f.set("A", False)
        assert f["A"] is False

    def test_setitem_value(self):
        f = Flags()
        f.add("A", value=True)
        f["A"] = False
        assert f["A"] is False

    def test_set_nonexistent_raises(self):
        f = Flags()
        with pytest.raises(KeyError, match="not registered"):
            f.set("missing", True)

    def test_setitem_nonexistent_raises(self):
        f = Flags()
        with pytest.raises(KeyError):
            f["missing"] = True


# ---------------------------------------------------------------------------
# Hierarchical propagation
# ---------------------------------------------------------------------------


class TestPropagation:
    def test_parent_false_propagates_to_child(self, simple_flags):
        simple_flags["A"] = False
        assert simple_flags["B"] is False

    def test_parent_false_propagates_to_grandchild(self, simple_flags):
        simple_flags["A"] = False
        assert simple_flags["C"] is False

    def test_intermediate_false_propagates_to_grandchild(self, simple_flags):
        simple_flags["B"] = False
        assert simple_flags["C"] is False

    def test_intermediate_false_does_not_affect_parent(self, simple_flags):
        simple_flags["B"] = False
        assert simple_flags["A"] is True

    def test_restoring_parent_does_not_auto_restore_child(self, simple_flags):
        """Setting parent back to True should NOT automatically restore children."""
        simple_flags["A"] = False  # propagates False down
        simple_flags["A"] = True  # restore parent only
        assert simple_flags["B"] is False
        assert simple_flags["C"] is False

    def test_child_can_be_false_independently(self, simple_flags):
        """A child can be False even when its parent is True."""
        simple_flags["B"] = False
        assert simple_flags["A"] is True
        assert simple_flags["B"] is False

    def test_multiple_children_all_receive_propagation(self):
        f = Flags()
        f.add("root", value=True)
        f.add("child_a", value=True, parent="root")
        f.add("child_b", value=True, parent="root")
        f["root"] = False
        assert f["child_a"] is False
        assert f["child_b"] is False

    def test_any_parent_false_propagates_to_multi_parent_child(self):
        f = Flags()
        f.add("A", value=True)
        f.add("B", value=True)
        f.add("C", value=True, parents=["A", "B"])

        f["A"] = False
        assert f["C"] is False

    def test_multi_parent_propagates_to_descendants(self):
        f = Flags()
        f.add("A", value=True)
        f.add("B", value=True)
        f.add("C", value=True, parents=["A", "B"])
        f.add("D", value=True, parent="C")

        f["B"] = False
        assert f["C"] is False
        assert f["D"] is False


# ---------------------------------------------------------------------------
# all()
# ---------------------------------------------------------------------------


class TestAll:
    def test_all_true_when_all_flags_up(self, simple_flags):
        assert simple_flags.all() is True

    def test_all_false_when_one_flag_down(self, simple_flags):
        simple_flags["B"] = False
        assert simple_flags.all() is False

    def test_all_false_when_all_flags_down(self, simple_flags):
        simple_flags["A"] = False
        assert simple_flags.all() is False

    def test_all_true_on_empty_flags(self):
        f = Flags()
        assert f.all() is True  # vacuously true


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------


class TestVerbose:
    def test_get_prints_when_verbose(self, capsys):
        f = Flags(_verbose=True)
        f.add("A", value=True)

        assert f.get("A") is True
        captured = capsys.readouterr()
        assert "Getting flag 'A': stored value=True" in captured.out

    def test_set_prints_when_verbose_with_propagation(self, capsys):
        f = Flags(_verbose=True)
        f.add("A", value=True)
        f.add("B", value=True, parent="A")

        f.set("A", False)
        captured = capsys.readouterr()

        assert "Set flag 'A' to False." in captured.out
        assert "Set flag 'B' to False." in captured.out

    def test_all_prints_status_when_verbose(self, capsys):
        f = Flags(_verbose=True)
        f.add("A", value=True)
        f.add("B", value=False)

        assert f.all() is False
        captured = capsys.readouterr()

        assert "Checking if all flags are effectively up:" in captured.out
        assert "Getting flag 'A': stored value=True" in captured.out
        assert "Getting flag 'B': stored value=False" in captured.out
        assert "A: True" in captured.out
        assert "B: False" in captured.out


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_includes_flag_names_and_parent_info(self):
        f = Flags()
        f.add("A", value=True)
        f.add("B", value=False, parent="A")

        rendered = repr(f)

        assert "A: stored=True" in rendered
        assert "B: stored=False" in rendered
        assert "parents=['A']" in rendered
