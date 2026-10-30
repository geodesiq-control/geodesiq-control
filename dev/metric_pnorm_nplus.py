"""
Fine scan of the p-norm example (metric_pnorm_example.py) over n_+ = alpha + beta and the norm order p.

For the register every transition has |<k|H_c|0>| = Omega / (2 dE_k), so the metric depends on alpha + beta only; here
alpha = beta = n_+ / 2 (beta > 0, so that uncoupled transitions never contribute). The norm axis is 1/p in [0, 1]:
p = inf at 0, p = 2 at 0.5, p = 1 at 1. The cluster of N qubits is weighted by N^(1/p) against the isolated qubit.

Panels: 1 - F at fixed durations, and the shortest duration t* with 1 - F <= EPS.

Run from dev/:  python metric_pnorm_nplus.py [--rescan]   (cached in data/metric_pnorm_nplus.npz)
"""

import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm

import geodesiq as gq
import metric_pnorm_example as ex

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
CACHE = HERE / "data" / "metric_pnorm_nplus.npz"
N_PLUS = np.round(np.arange(0.2, 8.01, 0.2), 2)
INV_P = np.round(np.arange(0, 1.01, 0.1), 2)
DURATIONS = np.geomspace(30, 3000, 21)
SHOWN = [100.0, 300.0, 1000.0]
EPS = 1e-3


def pulses():
    """All pulses from one register model: the eigenproblem is solved once, only the metric and the ODE change."""
    H_d, H_c = ex.register(ex.frequencies(ex.N_CLUSTER))
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(control_name="x", pulse_initial=ex.X0, pulse_final=ex.XF, initial_state=0, alpha=1, beta=1)
    result = {}
    for (i, n), (j, q) in itertools.product(enumerate(N_PLUS), enumerate(INV_P)):
        model.set_control(alpha=n / 2, beta=n / 2, norm=np.inf if q == 0 else 1 / q)
        model.solve_problem(pulse_accuracy=ex.PULSE_ACCURACY)
        result[i, j] = (model.tau, model.control_sol)
    return result


def infidelities(tau, x):
    qubits = [ex.make_model([w], alpha=0, beta=0) for w in ex.frequencies(ex.N_CLUSTER)]
    return [
        1 - np.prod([gq.Dynamics(q, times=tau * T, pulse=x, options=ex.OPTIONS).state_fidelity() for q in qubits])
        for T in DURATIONS
    ]


def scan():
    table = pulses()
    values = Parallel(n_jobs=-1, verbose=2)(delayed(infidelities)(*table[k]) for k in table)
    data = np.full((len(N_PLUS), len(INV_P), len(DURATIONS)), np.nan)
    for (i, j), v in zip(table, values, strict=True):
        data[i, j] = v
    np.savez(CACHE, data=data, n_plus=N_PLUS, inv_p=INV_P, durations=DURATIONS)
    return data


def shortest_duration(curve):
    """First duration with 1 - F <= EPS, interpolated in log-log between the grid points; nan if never reached."""
    log_curve, log_eps = np.log(np.maximum(curve, 1e-16)), np.log(EPS)
    below = np.nonzero(log_curve <= log_eps)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return DURATIONS[0]
    f = (log_eps - log_curve[i - 1]) / (log_curve[i] - log_curve[i - 1])
    return float(np.exp(np.log(DURATIONS[i - 1]) + f * np.log(DURATIONS[i] / DURATIONS[i - 1])))


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.6,
        "mathtext.fontset": "stix",
    }
)


def figure(data):
    t_star = np.apply_along_axis(shortest_duration, 2, data)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), layout="constrained")
    dn, dq = N_PLUS[1] - N_PLUS[0], INV_P[1] - INV_P[0]
    extent = [N_PLUS[0] - dn / 2, N_PLUS[-1] + dn / 2, INV_P[0] - dq / 2, INV_P[-1] + dq / 2]
    infidelity_norm = LogNorm(vmin=1e-7, vmax=1)
    for ax, T in zip(axes.flat[:3], SHOWN, strict=True):
        k = int(np.argmin(np.abs(DURATIONS - T)))
        im = ax.imshow(
            data[:, :, k].T, origin="lower", extent=extent, aspect="auto", norm=infidelity_norm, cmap="PuBu_r"
        )
        ax.set_title(rf"$1-\mathcal{{F}}$ at $t_f = {DURATIONS[k]:.0f}$", fontsize=8)
    fig.colorbar(im, ax=axes[0, :], label=r"$1-\mathcal{F}$", shrink=0.9)
    im = axes[1, 1].imshow(
        t_star.T, origin="lower", extent=extent, aspect="auto", norm=LogNorm(vmin=DURATIONS[0], vmax=DURATIONS[-1]),
        cmap="inferno_r",
    )
    axes[1, 1].set_title(rf"shortest $t_f$ with $1-\mathcal{{F}}\leq 10^{{{np.log10(EPS):.0f}}}$ (blank: $>{DURATIONS[-1]:.0f}$)",
                         fontsize=8)
    fig.colorbar(im, ax=axes[1, :], label=r"$t^*$", shrink=0.9)
    for ax in axes.flat:
        ax.axhline(0.5, color="k", lw=0.6, ls=":")  # p = 2
        for n, marker in ((4, "o"), (6, "s")):  # QMT, FAQUAD
            ax.plot([n], [0.5], marker, mfc="none", mec="k", ms=5)
        ax.set_xlabel(r"$n_+ = \alpha + \beta$")
        ax.set_ylabel(r"$1/p$")
        secondary = ax.secondary_yaxis("right", functions=(lambda q: q, lambda q: q))
        secondary.set_yticks([0, 0.25, 0.5, 1], [r"$\infty$", "4", "2", "1"])
        secondary.set_ylabel("$p$")
    return fig


if __name__ == "__main__":
    data = scan() if "--rescan" in sys.argv or not CACHE.exists() else np.load(CACHE)["data"]
    figure(data).savefig(HERE / "figures" / "metric_pnorm_nplus.png", dpi=200)
