"""
geodesiq CPU benchmark runner
==============================

Measures wall-clock time for the three internal computational stages of
:meth:`~geodesiq.ControlModel.solve_problem` across five physically motivated
sweep variables, then appends the results to a versioned Parquet history file
so multiple runs (across package versions or machines) can be compared later.

Usage
-----
Run from the project root with::

    uv run python -m benchmarks.run_benchmarks
    uv run python -m benchmarks.run_benchmarks --scenarios dim_scaling num_steps_scaling
    uv run python -m benchmarks.run_benchmarks --repeats 5

Results are appended to ``benchmarks/results/benchmark_history.parquet``.
"""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
import timeit
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from filelock import FileLock
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – allows  `python benchmarks/run_benchmarks.py`  (direct) as
# well as  `python -m benchmarks.run_benchmarks`  (module).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geodesiq  # noqa: E402 (must be after sys.path setup)
from geodesiq import ControlModel  # noqa: E402

# relative import works when run as module; absolute when run as script
try:
    from ._models import make_ham
except ImportError:
    from _models import make_ham  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = _HERE / "results"
HISTORY_FILE = RESULTS_DIR / "benchmark_history.parquet"

_DEFAULT_N_REPEAT = 7
_CALIBRATION_TARGET_S = 0.15  # aim for ≈ 0.15 s of total work per inner-loop batch
_CALIBRATION_MAX_N = 100  # never run more than 100 inner loops

# Sweep grids
DIMS = [2, 4, 8, 16, 32, 64]
NUM_STEPS_LIST = [2 ** k + 1 for k in (5, 7, 9, 11)]  # 33, 129, 513, 2049
PULSE_ACCURACIES = [50, 100, 250, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _calibrate_n(fn: Callable, target_s: float = _CALIBRATION_TARGET_S) -> int:
    """Return ``n`` so that ``n × cost_per_call ≈ target_s`` (clamped to [1, 100])."""
    t0 = time.perf_counter()
    fn()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return max(1, min(_CALIBRATION_MAX_N, int(target_s / elapsed)))


def _time_fn(fn: Callable, n_inner: int, n_repeat: int = _DEFAULT_N_REPEAT) -> list[float]:
    """Return a list of *per-call* wall-clock times (seconds)."""
    raw = timeit.repeat(fn, number=n_inner, repeat=n_repeat, timer=time.perf_counter)
    return [t / n_inner for t in raw]


def _stats(times: list[float]) -> dict:
    return {"mean_s": statistics.mean(times), "median_s": statistics.median(times), "std_s": statistics.pstdev(times),
            "min_s": min(times), "max_s": max(times), }


# ---------------------------------------------------------------------------
# Stage-level benchmarking
# ---------------------------------------------------------------------------


def benchmark_ham(ham: ControlModel, pulse_accuracy: int = 1000, n_repeat: int = _DEFAULT_N_REPEAT, ) -> dict[
    str, dict]:
    """
    Benchmark the three internal stages of :meth:`solve_problem` plus the
    full pipeline, in isolation.

    The model is solved once as a warm-up so all caches are populated.
    Each stage is then timed through the public cache-invalidation and staged
    solve APIs, keeping the benchmark independent of private implementation details.

    Parameters
    ----------
    ham
        A *fully configured* :class:`~geodesiq.ControlModel` instance.
    pulse_accuracy
        Number of ODE evaluation points; forwarded to ``_solve_ode``.
    n_repeat
        Number of outer timing repetitions (``timeit.repeat`` ``repeat``
        argument).

    Returns
    -------
    dict
        Maps stage name → dict with keys ``mean_s``, ``median_s``,
        ``std_s``, ``min_s``, ``max_s``, ``n_inner``.
    """
    # Warm up once. Warnings are intentionally not suppressed: numerical
    # invalidity must fail or remain visible in benchmark runs.
    ham.solve_problem(pulse_accuracy=pulse_accuracy)

    results: dict[str, dict] = {}

    # ── eigenproblem stage ──────────────────────────────────────────────────
    # Resetting eigenproblem_solved cascades metric_computed → ode_solved.
    def _bench_eigen() -> None:
        ham._flags["eigenproblem_solved"] = False
        ham._solve_eigenproblem()

    n = _calibrate_n(_bench_eigen)
    results["eigenproblem"] = {"n_inner": n, **_stats(_time_fn(_bench_eigen, n, n_repeat))}

    # ── metric tensor stage ─────────────────────────────────────────────────
    # After the last _bench_eigen call, eigenproblem_solved=True.
    # Resetting metric_computed cascades only ode_solved.
    def _bench_metric() -> None:
        ham._flags["metric_computed"] = False
        ham._compute_metric_tensor()

    n = _calibrate_n(_bench_metric)
    results["metric"] = {"n_inner": n, **_stats(_time_fn(_bench_metric, n, n_repeat))}

    # ── ODE stage ───────────────────────────────────────────────────────────
    # After the last _bench_metric call, metric_computed=True.
    def _bench_ode() -> None:
        ham._flags["ode_solved"] = False
        ham._solve_ode(pulse_accuracy)

    n = _calibrate_n(_bench_ode)
    results["ode"] = {"n_inner": n, **_stats(_time_fn(_bench_ode, n, n_repeat))}

    # ── full pipeline ───────────────────────────────────────────────────────
    def _bench_total() -> None:
        ham._flags["eigenproblem_solved"] = False  # cascades everything
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ham.solve_problem(pulse_accuracy=pulse_accuracy)

    n = _calibrate_n(_bench_total)
    results["total"] = {"n_inner": n, **_stats(_time_fn(_bench_total, n, n_repeat))}

    return results


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _make_rows(scenario: str, sweep_var: str, sweep_val: float | int | str, variant: str,
               stage_results: dict[str, dict], meta: dict, ) -> list[dict]:
    rows: list[dict] = []
    for stage, stats in stage_results.items():
        rows.append({"scenario": scenario, "sweep_var": sweep_var, "sweep_val": str(sweep_val),
                     "sweep_val_num": float(sweep_val) if isinstance(sweep_val, (int, float)) else float("nan"),
                     "variant": variant, "stage": stage, **stats, **meta, })
    return rows


# ---------------------------------------------------------------------------
# Machine / run metadata
# ---------------------------------------------------------------------------


def _machine_info() -> dict:
    return {"geodesiq_version": geodesiq.__version__, "python_version": sys.version.split()[0],
            "platform": platform.platform(), "cpu_count": os.cpu_count() or 1, "processor": platform.processor(),
            "run_timestamp": datetime.now(timezone.utc).isoformat(), }


# ---------------------------------------------------------------------------
# Individual scenarios
# ---------------------------------------------------------------------------


def run_dim_scaling(meta: dict, n_repeat: int) -> list[dict]:
    """
    **Scenario: dim_scaling**

    Times all four stages (eigenproblem / metric / ODE / total) as the
    Hilbert-space dimension grows from 2 to 64, using the analytical
    partial derivative.  ``num_steps`` is fixed at 2⁸ + 1 = 257.
    """
    rows: list[dict] = []
    for dim in tqdm(DIMS, desc="  dim_scaling", leave=False):
        ham = make_ham(dim=dim, num_steps=2 ** 8 + 1, analytical_partial=True)
        res = benchmark_ham(ham, n_repeat=n_repeat)
        rows += _make_rows("dim_scaling", "dim", dim, "analytical", res, meta)
    return rows


def run_num_steps_scaling(meta: dict, n_repeat: int) -> list[dict]:
    """
    **Scenario: num_steps_scaling**

    Times all stages as ``num_steps`` grows from 33 to 2049 (powers of 2
    plus one, to keep the Romberg integrator happy), using the 2-level
    Landau-Zener model with analytical partial derivative.
    """
    rows: list[dict] = []
    for ns in tqdm(NUM_STEPS_LIST, desc="  num_steps_scaling", leave=False):
        ham = make_ham(dim=2, num_steps=ns, analytical_partial=True)
        res = benchmark_ham(ham, n_repeat=n_repeat)
        rows += _make_rows("num_steps_scaling", "num_steps", ns, "analytical", res, meta)
    return rows


def run_analytical_vs_numerical(meta: dict, n_repeat: int) -> list[dict]:
    """
    **Scenario: analytical_vs_numerical**

    Compares CPU time when an analytical ∂H/∂λ is provided versus when the
    internal numerical-differentiation path is used.  Sweeps over ``dim``;
    times all stages so the overhead can be attributed to the right stage.
    """
    rows: list[dict] = []
    for dim in tqdm(DIMS, desc="  analytical_vs_numerical", leave=False):
        for variant, analytical in (("analytical", True), ("numerical", False)):
            ham = make_ham(dim=dim, num_steps=2 ** 8 + 1, analytical_partial=analytical)
            res = benchmark_ham(ham, n_repeat=n_repeat)
            rows += _make_rows("analytical_vs_numerical", "dim", dim, variant, res, meta)
    return rows


def run_adiabatic_vs_diabatic(meta: dict, n_repeat: int) -> list[dict]:
    """
    **Scenario: adiabatic_vs_diabatic**

    Compares the adiabatic metric (initial_state == final_state) against the
    diabatic metric (state-to-state transfer, dim − 1 anticrossings).
    Sweeps over ``dim`` ≥ 4; times all stages.
    """
    rows: list[dict] = []
    for dim in tqdm(DIMS[1:], desc="  adiabatic_vs_diabatic", leave=False):  # dim ≥ 4
        for variant, adiabatic in (("adiabatic", True), ("diabatic", False)):
            ham = make_ham(dim=dim, num_steps=2 ** 8 + 1, analytical_partial=True, adiabatic=adiabatic)
            res = benchmark_ham(ham, n_repeat=n_repeat)
            rows += _make_rows("adiabatic_vs_diabatic", "dim", dim, variant, res, meta)
    return rows


def run_pulse_accuracy_scaling(meta: dict, n_repeat: int) -> list[dict]:
    """
    **Scenario: pulse_accuracy_scaling**

    Times all stages as the ODE evaluation-point count (``pulse_accuracy``)
    grows from 50 to 2000, using the 2-level LZ model.  Only the ODE stage
    is expected to grow significantly; the other stages serve as baselines.
    """
    rows: list[dict] = []
    for pa in tqdm(PULSE_ACCURACIES, desc="  pulse_accuracy_scaling", leave=False):
        ham = make_ham(dim=2, num_steps=2 ** 8 + 1, analytical_partial=True)
        res = benchmark_ham(ham, pulse_accuracy=pa, n_repeat=n_repeat)
        rows += _make_rows("pulse_accuracy_scaling", "pulse_accuracy", pa, "default", res, meta)
    return rows


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Callable[..., list[dict]]] = {"dim_scaling": run_dim_scaling,
                                                   "num_steps_scaling": run_num_steps_scaling,
                                                   "analytical_vs_numerical": run_analytical_vs_numerical,
                                                   "adiabatic_vs_diabatic": run_adiabatic_vs_diabatic,
                                                   "pulse_accuracy_scaling": run_pulse_accuracy_scaling, }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(scenarios: list[str] | None = None, n_repeat: int = _DEFAULT_N_REPEAT, ) -> pd.DataFrame:
    """
    Run the selected (or all) scenarios and return a tidy
    :class:`~pandas.DataFrame` with one row per (scenario, sweep_val, stage)
    measurement.

    Parameters
    ----------
    scenarios
        Names of scenarios to run.  Defaults to all registered scenarios.
    n_repeat
        Number of outer timing repetitions per measurement point.
    """
    meta = _machine_info()
    names = scenarios or list(SCENARIOS.keys())

    # Validate up-front
    invalid = [n for n in names if n not in SCENARIOS]
    if invalid:
        raise ValueError(f"Unknown scenario(s): {invalid}.  Available: {list(SCENARIOS.keys())}")

    all_rows: list[dict] = []
    for name in names:
        tqdm.write(f"\n▶  {name}")
        rows = SCENARIOS[name](meta, n_repeat)
        all_rows.extend(rows)
        tqdm.write(f"   ✓  {len(rows)} rows")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def save_results(df: pd.DataFrame) -> Path:
    """
    Append *df* to ``benchmark_history.parquet``.

    Because Parquet files are immutable, the existing file is read,
    concatenated with *df*, and written back.  The ``run_timestamp`` column
    allows filtering by session later.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = HISTORY_FILE.with_suffix(HISTORY_FILE.suffix + ".lock")
    temporary = HISTORY_FILE.with_suffix(HISTORY_FILE.suffix + f".{os.getpid()}.tmp")

    with FileLock(lock_path):
        combined = df
        if HISTORY_FILE.exists():
            existing = pd.read_parquet(HISTORY_FILE)
            combined = pd.concat([existing, df], ignore_index=True)
        try:
            combined.to_parquet(temporary, index=False)
            os.replace(temporary, HISTORY_FILE)
        finally:
            temporary.unlink(missing_ok=True)
    return HISTORY_FILE


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m benchmarks.run_benchmarks", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument("--scenarios", nargs="*", choices=list(SCENARIOS.keys()), metavar="SCENARIO",
                        help=f"Scenarios to run (default: all).  Choices: {list(SCENARIOS.keys())}", )
    parser.add_argument("--repeats", type=int, default=_DEFAULT_N_REPEAT, metavar="N",
                        help=f"Outer timing repetitions per measurement (default: {_DEFAULT_N_REPEAT}).", )
    parser.add_argument("--list", action="store_true", dest="list_only", help="List available scenarios and exit.", )
    args = parser.parse_args()

    if args.list_only:
        print("Available scenarios:")
        for name, fn in SCENARIOS.items():
            first_line = (fn.__doc__ or "").strip().splitlines()[0].strip("* ")
            print(f"  {name:<30}  {first_line}")
        return

    # ── print header ─────────────────────────────────────────────────────────
    sep = "─" * 62
    print(sep)
    print(f"  geodesiq benchmark suite  (v{geodesiq.__version__})")
    print(f"  Python {sys.version.split()[0]}  │  {platform.processor() or platform.machine()}")
    print(f"  {platform.platform()}")
    print(f"  Repeats per point: {args.repeats}  │  Scenarios: {args.scenarios or 'all'}")
    print(sep)

    df = run_all(scenarios=args.scenarios, n_repeat=args.repeats)

    if df.empty:
        print("No results collected.")
        return

    path = save_results(df)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  ✅  {len(df)} rows appended → {path}")
    print(f"{'─' * 62}")

    summary = (df.groupby(["scenario", "stage"])["mean_s"].mean().unstack("stage").reindex(
        columns=["eigenproblem", "metric", "ode", "total"], fill_value=float("nan")))
    # Format as ms
    print("\nMean CPU time (ms) by scenario / stage:")
    print((summary * 1e3).round(3).to_string())


if __name__ == "__main__":
    main()
