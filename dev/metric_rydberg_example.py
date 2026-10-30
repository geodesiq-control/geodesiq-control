"""
An example with a huge difference between metrics: adiabatic preparation of the Z2 crystal in a Rydberg chain.

PXP model, L = 9 atoms in the blockade subspace (no neighbouring excitations, 89 states):
    H = Omega/2 sum_i X_i - Delta sum_i n_i,   Delta: -Delta_0 -> +Delta_0,   Omega = 1,
from the all-ground state to the Z2 state |rgrgrgrgr> (Bernien et al., Nature 551, 579 (2017)).

Why the metrics differ so much: once the minimum gap (~0.39 Omega) is passed adiabatically, the infidelity is set by
the endpoint terms of adiabatic perturbation theory, c_k = x' |<k|H_c|n>| / dE_k^2 at t = 0 and t_f, with
x' = a_tilde / (t_f sqrt(G)). Hence 1 - F ~ C / t_f^2 with C = sum over ends and k of (a_tilde |dH_kn| /
(sqrt(G) dE_k^2))^2 (dashed lines in panel c). FAQUAD keeps |dH_kn| / dE^2 / sqrt(G) constant, so its endpoint
terms are as large as at the minimum gap; QMT has sqrt(G) ~ |dH| / dE and gains a factor 1 / dE_end ~ 1 / Delta_0.
Its advantage grows as Delta_0^2 (panel d). The linear ramp has the smallest endpoint terms but loses at the gap.

Run from dev/:  python metric_rydberg_example.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import NullFormatter

import geodesiq as gq
from metric_comparison import METRICS, STYLES, slowdown

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
L = 9
DELTA_0 = 20.0
DURATIONS = np.geomspace(3, 3e3, 25)
SWEEP_RANGES = [5.0, 10.0, 20.0, 40.0, 80.0]
FIXED_DURATION = 300.0
OPTIONS = dict(atol=1e-12, rtol=1e-10, nsteps=10**4)  # infidelities down to ~1e-10


def pxp(L, omega=1.0):
    """H_d = Omega/2 sum X_i and H_c = -sum n_i in the blockade subspace of an open chain of L atoms."""
    states = [s for s in range(2**L) if not (s & (s >> 1))]
    index = {s: i for i, s in enumerate(states)}
    H_d = np.zeros((len(states), len(states)))
    for s in states:
        for j in range(L):
            if (t := s ^ (1 << j)) in index:
                H_d[index[s], index[t]] = omega / 2
    H_c = -np.diag([bin(s).count("1") for s in states]).astype(float)
    return H_d, H_c


def make_model(delta_0, metric):
    H_d, H_c = pxp(L)
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(
        control_name="delta", pulse_initial=-delta_0, pulse_final=delta_0, initial_state=0, **METRICS[metric]
    )
    model.solve_problem(pulse_accuracy=2001)
    return model


def endpoint_coefficient(model):
    """C in 1 - F ~ C / t_f^2: the endpoint terms of adiabatic perturbation theory, summed incoherently."""
    H_d, H_c = model.H_d, model.H_c
    total = 0.0
    for i in (0, -1):
        E, V = np.linalg.eigh(H_d + model.control_pulse[i] * H_c)
        dH = np.abs(V.T @ H_c @ V[:, 0])[1:]
        dE = E[1:] - E[0]
        total += np.sum((model._a_tilde / np.sqrt(model._metric_tensor[i]) * dH / dE**2) ** 2)
    return total


def infidelities(delta_0, metric, durations):
    model = make_model(delta_0, metric)
    return [1 - gq.Dynamics(model, T, options=OPTIONS).state_fidelity() for T in durations]


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "mathtext.fontset": "stix",
    }
)


def figure():
    names = list(METRICS)
    curves = Parallel(n_jobs=-1)(delayed(infidelities)(DELTA_0, m, DURATIONS) for m in names)
    scan = Parallel(n_jobs=-1)(delayed(infidelities)(d, m, [FIXED_DURATION]) for m in names for d in SWEEP_RANGES)
    scan = np.array(scan).reshape(len(names), len(SWEEP_RANGES))

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    (ax_E, ax_v), (ax_F, ax_D) = axes
    models = {m: make_model(DELTA_0, m) for m in names}
    x, E = models["QMT (2, 2)"].control_pulse, models["QMT (2, 2)"].eigenenergies

    ax_E.plot(x, E[:, 1:8] - E[:, [0]], color="0.6", lw=0.8)
    ax_E.set_ylim(0, 6)
    ax_E.set_xlabel(r"$\Delta/\Omega$")
    ax_E.set_ylabel(r"$(E_k - E_0)/\Omega$")
    ax_E.set_title(f"Rydberg chain, $L={L}$: gaps to the ground state", fontsize=8)

    for m, model in models.items():
        ax_v.plot(x, slowdown(model, dict(x0=-DELTA_0, xf=DELTA_0)), label=m, **STYLES[m])
        ax_F.plot(DURATIONS, curves[names.index(m)], label=m, **STYLES[m])
        ax_D.plot(SWEEP_RANGES, scan[names.index(m)], marker="o", ms=3, label=m, **STYLES[m])
    for m in ("QMT (2, 2)", "FAQUAD (4, 2)"):
        ax_F.plot(DURATIONS, endpoint_coefficient(models[m]) / DURATIONS**2, color=STYLES[m]["color"], lw=0.8,
                  ls=(0, (4, 3)), alpha=0.7)
    ax_v.set_yscale("log")
    ax_v.set_xlabel(r"$\Delta/\Omega$")
    ax_v.set_ylabel(r"$(x_f-x_0)\,\mathrm{d}\tau/\mathrm{d}x$")
    ax_v.set_title(rf"Local slowdown, $\Delta_0 = {DELTA_0:g}\,\Omega$", fontsize=8)

    ax_F.set(xscale="log", yscale="log", ylim=(1e-9, 2), xlabel=r"$\Omega\,t_f$", ylabel=r"$1-\mathcal{F}$")
    ax_F.set_title(rf"$\Delta_0 = {DELTA_0:g}\,\Omega$; dashed: endpoint terms $C/t_f^2$", fontsize=8)
    ax_D.set(xscale="log", yscale="log", xlabel=r"sweep range $\Delta_0/\Omega$", ylabel=r"$1-\mathcal{F}$")
    ax_D.set_xticks(SWEEP_RANGES, [f"{d:g}" for d in SWEEP_RANGES])
    ax_D.xaxis.set_minor_formatter(NullFormatter())
    ax_D.set_title(rf"$\Omega\,t_f = {FIXED_DURATION:g}$", fontsize=8)
    for ax in axes.flat:
        ax.grid(True, which="major")
    handles, labels = ax_v.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METRICS), frameon=False)
    return fig


if __name__ == "__main__":
    figure().savefig(HERE / "figures" / "metric_rydberg_example.png", dpi=200)
