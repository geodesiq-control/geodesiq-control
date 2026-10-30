"""
Heatmaps for the p-norm example (metric_pnorm_example.py): infidelity of the register inversion at a fixed duration
over (alpha, beta) for p = 1, 2, inf, and the ratio between p = 1 and p = 2.

For this register every transition is a two-level crossing with a constant Rabi coupling, |<k|H_c|0>| = Omega / (2 dE_k),
so the metric depends on alpha + beta only and the maps are constant along the anti-diagonals.

Run from dev/:  python metric_pnorm_heatmap.py [--rescan]   (cached in data/metric_pnorm_heatmap.npz)
"""

import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, TwoSlopeNorm

import geodesiq as gq
import metric_pnorm_example as ex

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
CACHE = HERE / "data" / "metric_pnorm_heatmap.npz"
ALPHAS = np.arange(0, 6.01, 0.5)
BETAS = np.arange(0, 4.01, 0.5)
DURATIONS = [100.0, 300.0, 1000.0]
SHOWN_DURATION = 300.0


def pulses():
    """All pulses from one register model: the eigenproblem is solved once, only the metric and the ODE change."""
    H_d, H_c = ex.register(ex.frequencies(ex.N_CLUSTER))
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(control_name="x", pulse_initial=ex.X0, pulse_final=ex.XF, initial_state=0, alpha=1, beta=1)
    result = {}
    for (i, a), (j, b), p in itertools.product(enumerate(ALPHAS), enumerate(BETAS), ex.NORMS):
        model.set_control(alpha=float(a), beta=float(b), norm=ex.NORMS[p])
        model.solve_problem(pulse_accuracy=ex.PULSE_ACCURACY)
        result[i, j, p] = (model.tau, model.control_sol)
    return result


def infidelities(tau, x):
    omegas = ex.frequencies(ex.N_CLUSTER)
    qubits = [ex.make_model([w], alpha=0, beta=0) for w in omegas]
    return [
        1 - np.prod([gq.Dynamics(q, times=tau * T, pulse=x, options=ex.OPTIONS).state_fidelity() for q in qubits])
        for T in DURATIONS
    ]


def scan():
    table = pulses()
    keys = list(table)
    values = Parallel(n_jobs=-1, verbose=2)(delayed(infidelities)(*table[k]) for k in keys)
    data = np.full((len(ex.NORMS), len(ALPHAS), len(BETAS), len(DURATIONS)), np.nan)
    norms = list(ex.NORMS)
    for (i, j, p), v in zip(keys, values, strict=True):
        data[norms.index(p), i, j] = v
    np.savez(CACHE, data=data, alphas=ALPHAS, betas=BETAS, durations=DURATIONS)
    return data


def figure(data):
    k = DURATIONS.index(SHOWN_DURATION)
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.7), layout="constrained")
    extent = [ALPHAS[0] - 0.25, ALPHAS[-1] + 0.25, BETAS[0] - 0.25, BETAS[-1] + 0.25]
    norm = LogNorm(vmin=1e-6, vmax=1)
    for ax, (n, p) in zip(axes[:3], enumerate(ex.NORMS), strict=True):
        im = ax.imshow(data[n, :, :, k].T, origin="lower", extent=extent, aspect="auto", norm=norm, cmap="PuBu_r")
        ax.plot([2], [2], "o", mfc="none", mec="k", ms=5)  # QMT
        ax.plot([4], [2], "s", mfc="none", mec="k", ms=5)  # FAQUAD
        ax.set_title(f"QMT-family, {p}", fontsize=8)
    fig.colorbar(im, ax=axes[:3], label=rf"$1-\mathcal{{F}}$ at $t_f={SHOWN_DURATION:g}$", shrink=0.9)
    ratio = np.log10(data[0, :, :, k] / data[1, :, :, k])
    im = axes[3].imshow(
        ratio.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -3, 3)
    )
    axes[3].set_title(r"$\log_{10}$ of $(1-\mathcal{F})_{p=1}/(1-\mathcal{F})_{p=2}$", fontsize=8)
    fig.colorbar(im, ax=axes[3], shrink=0.9)
    for ax in axes:
        ax.set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\beta$")
    return fig


if __name__ == "__main__":
    data = scan() if "--rescan" in sys.argv or not CACHE.exists() else np.load(CACHE)["data"]
    figure(data).savefig(HERE / "figures" / "metric_pnorm_heatmap.png", dpi=200)
