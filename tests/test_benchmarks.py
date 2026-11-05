"""Smoke tests for the stage benchmarks, which drive private ControlModel stages and flags (P-35)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# benchmarks/ is not shipped in the sdist, which includes the tests, and it needs the optional `bench` group;
# importorskip skips the module in both cases.
for module in ("pandas", "filelock", "tqdm"):
    pytest.importorskip(module)
run_benchmarks = pytest.importorskip("benchmarks.run_benchmarks")
make_ham = pytest.importorskip("benchmarks._models").make_ham


@pytest.fixture
def one_call_per_stage(monkeypatch):
    """Skip the calibration so that every stage is timed with a single call."""
    monkeypatch.setattr(run_benchmarks, "_calibrate_n", lambda fn, target_s=None: 1)


@pytest.mark.parametrize("adiabatic", [True, False])
def test_benchmark_ham_times_every_stage_on_a_small_model(one_call_per_stage, adiabatic):
    """Fails if a private stage method or flag used by benchmark_ham is renamed."""
    ham = make_ham(dim=4, num_steps=33, adiabatic=adiabatic)

    results = run_benchmarks.benchmark_ham(ham, pulse_accuracy=50, n_repeat=1)

    assert list(results) == ["eigenproblem", "metric", "ode", "total"]
    for stats in results.values():
        assert stats["n_inner"] == 1
        assert stats["mean_s"] > 0


def test_metric_stage_rebuilds_the_diabatic_list_on_every_call(one_call_per_stage, monkeypatch):
    """The metric stage includes the diabatic passage list, not only in its first call."""
    ham = make_ham(dim=4, num_steps=33, adiabatic=False)
    calls = []
    solve_dia_list = ham._solve_dia_list

    def counting_solve_dia_list(config):
        if not ham._flags["dia_list_computed"]:
            calls.append(1)
        solve_dia_list(config)

    monkeypatch.setattr(ham, "_solve_dia_list", counting_solve_dia_list)
    time_fn = run_benchmarks._time_fn
    metric_calls = {}

    def recording_time_fn(fn, n_inner, n_repeat):
        before = len(calls)
        times = time_fn(fn, n_inner, n_repeat)
        metric_calls[fn.__name__] = len(calls) - before
        return times

    monkeypatch.setattr(run_benchmarks, "_time_fn", recording_time_fn)

    run_benchmarks.benchmark_ham(ham, pulse_accuracy=50, n_repeat=3)

    # Three timed repetitions of one call each: the list is rebuilt three times in the metric stage.
    assert metric_calls["_bench_metric"] == 3
    assert metric_calls["_bench_ode"] == 0
