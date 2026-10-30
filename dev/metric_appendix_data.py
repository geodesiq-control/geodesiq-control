"""
Data of the appendix that compares the metrics (paper: "Comparison of the metrics"), saved to
data/metric_appendix.npz for the figure cells of manuscript_figures.ipynb.

- profiles: spectrum, local slowdown (x_f - x_i) dtau/dx and pulse x(tau) of five metrics for four systems
  (metric_comparison.py), at one gap per system (Landau-Zener with Delta = 1, as in the paper).
- gap scan: peak slowdown (computed here) and shortest duration with 1 - F <= 1e-2 (cached duration scan of
  metric_comparison.py, data/metric_comparison_scan.npz) against the minimum gap.
- crowded qubits (metric_pnorm_example.py): slowdown and 1 - F at t_f = 300 for N = 1, 2, 4, 6, 8 crowded qubits and
  p = 1, 2, inf (computed here, one model at a time), and from the fine scan (metric_pnorm_fine.py) the infidelity
  against t_f for (alpha, beta) = (2, 2) and the shortest duration with 1 - F <= 1e-3 over (n_+, 1/p).

Run from dev/ (single-threaded, so that it can share the machine with timing runs):
    python metric_appendix_data.py
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import geodesiq as gq  # noqa: E402
import metric_comparison as mc  # noqa: E402
import metric_pnorm_example as ex  # noqa: E402
import metric_pnorm_fine as fine  # noqa: E402

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
OUT = HERE / "data" / "metric_appendix.npz"
PROFILE_GAPS = [1.0, 0.12, 0.03, 0.8]  # Landau-Zener Delta, band coupling g, DQD Delta E_X, Ising h_z
LEVELS_SHOWN = 9  # lowest levels stored for the spectra
CLUSTER_SIZES = [1, 2, 4, 6, 8]
QUBIT_DURATION = 300.0
EPS_QUBITS = 1e-3


def profiles(out):
    for s, (system, value) in enumerate(zip(mc.SYSTEMS, PROFILE_GAPS, strict=True)):
        for m, metric in enumerate(mc.METRICS):
            model = mc.make_model(system, value, metric)
            if m == 0:
                out[f"profile{s}_x"] = model.control_pulse
                out[f"profile{s}_E"] = model.eigenenergies[:, :LEVELS_SHOWN]
                out[f"profile{s}_gap"] = np.array(mc.min_gap(model, system["state"]))
            out[f"profile{s}_slowdown{m}"] = mc.slowdown(model, system)
            out[f"profile{s}_tau{m}"] = model.tau
            out[f"profile{s}_pulse{m}"] = model.control_sol


def gap_scan(out):
    table = mc.load_scan()
    for s, system in enumerate(mc.SYSTEMS):
        gaps, peaks, durations = [], [], []
        for v in system["values"]:
            row_peaks = []
            for metric in mc.METRICS:
                model = mc.make_model(system, v, metric)
                row_peaks.append(mc.slowdown(model, system).max())
            gaps.append(mc.min_gap(model, system["state"])[0])
            peaks.append(row_peaks)
            durations.append([table[f"{s}|{v}|{metric}"] for metric in mc.METRICS])
        out[f"scan{s}_values"] = np.array(system["values"])
        out[f"scan{s}_gap"] = np.array(gaps)
        out[f"scan{s}_peak"] = np.array(peaks)
        out[f"scan{s}_duration"] = np.array(durations)


def crowded_qubits(out):
    norms = list(ex.NORMS)
    scan = np.full((len(norms), len(CLUSTER_SIZES)), np.nan)
    for j, n in enumerate(CLUSTER_SIZES):
        for i, p in enumerate(norms):
            x, v, (infidelity,) = ex.run(n, p, [QUBIT_DURATION])
            scan[i, j] = infidelity
            if n == ex.N_CLUSTER:
                out["qubits_x"] = x
                out[f"qubits_slowdown{i}"] = v
            print(f"N={n} {p}: 1 - F = {infidelity:.2e}", flush=True)
    out["qubits_cluster_sizes"] = np.array(CLUSTER_SIZES)
    out["qubits_scan"] = scan
    out["qubits_frequencies"] = np.array(ex.frequencies(ex.N_CLUSTER))

    # Fine scan: infidelity against t_f at n_+ = 4, and t* over (n_+, 1/p)
    n_plus, inv_p, durations, data = fine.grid(fine.load_results())
    out["fine_n_plus"], out["fine_inv_p"], out["fine_durations"] = n_plus, inv_p, durations
    i4 = int(np.argmin(np.abs(n_plus - 4)))
    out["fine_qmt"] = np.array([data[i4, int(np.argmin(np.abs(inv_p - q)))] for q in (1.0, 0.5, 0.0)])
    log_eps = np.log(EPS_QUBITS)
    t_star = np.full(data.shape[:2], np.nan)
    for i, j in np.ndindex(*t_star.shape):
        log_curve = np.log(np.maximum(data[i, j], 1e-16))
        below = np.nonzero(log_curve <= log_eps)[0]
        if below.size == 0:
            continue
        k = below[0]
        if k == 0:
            t_star[i, j] = durations[0]
            continue
        f = (log_eps - log_curve[k - 1]) / (log_curve[k] - log_curve[k - 1])
        t_star[i, j] = np.exp(np.log(durations[k - 1]) + f * np.log(durations[k] / durations[k - 1]))
    out["fine_t_star"] = t_star


if __name__ == "__main__":
    out = {}
    profiles(out)
    print("profiles done", flush=True)
    gap_scan(out)
    print("gap scan done", flush=True)
    crowded_qubits(out)
    np.savez_compressed(OUT, **out)
    print(f"saved {OUT}", flush=True)
