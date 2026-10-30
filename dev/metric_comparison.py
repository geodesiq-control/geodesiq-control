"""
Compare the metrics of the geodesic approach for several band structures and gaps.

Systems (built from the examples):
- Landau–Zener: two levels, one isolated anticrossing (gap 2 Delta).
- Level through a band: one level swept across 8 equally spaced levels, starting in the middle of the band, so the
  occupied state meets two close neighbours at once (gap set by the coupling g).
- DQD (examples/dqd_model.ipynb): five levels, the charge anticrossing (tc) followed by the S–T- anticrossing (dEx).
- Ising chain (examples/many_body.ipynb): L = 6, reflection-even sector, many levels. The gap at the transition is
  ~0.69 for any small hz; hz -> 1 closes the gap at the end of the sweep (2 - 2 hz at lambda = 1).

Metrics: linear ramp (0, 0), quantum metric (2, 2), FAQUAD (4, 2), and FAQUAD with the L^1 (multilevel FAQUAD) and
L^inf norms over the transitions.

Figure 1: spectrum, local slowdown (x_f - x_0) dtau/dx = (x_f - x_0) sqrt(G) / a_tilde, and pulse x(tau), for one gap.
Figure 2: against the minimum gap of the occupied state, the slowdown at the minimum gap and the shortest duration with
1 - F <= EPS.

Run from dev/:  python metric_comparison.py [--rescan]
The duration scan is cached in data/metric_comparison_scan.npz.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import NullFormatter

import geodesiq as gq

HERE = Path(__file__).parent
CACHE = HERE / "data" / "metric_comparison_scan.npz"
EPS = 1e-2  # target infidelity
T_MIN, T_MAX, T_FACTOR = 0.1, 2e4, 1.25
PULSE_ACCURACY = 2001

METRICS = {
    "linear (0, 0)": dict(alpha=0, beta=0, norm=2),
    "QMT (2, 2)": dict(alpha=2, beta=2, norm=2),
    "FAQUAD (4, 2)": dict(alpha=4, beta=2, norm=2),
    r"FAQUAD $p=1$": dict(alpha=4, beta=2, norm=1),
    r"FAQUAD $p=\infty$": dict(alpha=4, beta=2, norm=np.inf),
}
STYLES = {
    "linear (0, 0)": dict(color="0.55", ls=":"),
    "QMT (2, 2)": dict(color="C0", ls="-"),
    "FAQUAD (4, 2)": dict(color="C3", ls="-"),
    r"FAQUAD $p=1$": dict(color="C1", ls="--"),
    r"FAQUAD $p=\infty$": dict(color="C2", ls="-."),
}


# ----- Systems --------------------------------------------------------------------------------------------------------
def landau_zener(delta):
    sx, sz = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    return delta * sx, sz


def level_through_band(g, n_band=8, width=4.0):
    energies = np.linspace(-width / 2, width / 2, n_band)
    H_d = np.diag(np.r_[0.0, energies])
    H_d[0, 1:] = H_d[1:, 0] = g
    H_c = np.zeros_like(H_d)
    H_c[0, 0] = 1.0
    return H_d, H_c


def dqd_hamiltonian(eps, U, tc, Ez, dEz, dEx):
    return np.array(
        [
            [U - eps, 0, -tc, tc, 0],
            [0, Ez, dEx, -dEx, 0],
            [-tc, dEx, dEz, 0, dEx],
            [tc, -dEx, 0, -dEz, -dEx],
            [0, 0, dEx, -dEx, -Ez],
        ],
        dtype=float,
    )


def dqd(dEx):
    return dqd_hamiltonian(0, 10, 1, 0.9, 0.1, dEx), dqd_hamiltonian(1, 0, 0, 0, 0, 0)


def ising_numpy(lam, L, hx, hz):
    # Same as ising_model_numpy in examples/many_body.ipynb (reflection-even sector)
    states = np.arange(2**L)
    bits = (states[:, None] >> np.arange(L - 1, -1, -1)) & 1
    sz = 2 * bits - 1
    H = np.diag(lam * (np.sum(sz[:, :-1] * sz[:, 1:], axis=1) + hz * np.sum(sz, axis=1)).astype(float))
    for j in range(L):
        H[states, states ^ (1 << (L - 1 - j))] += (1 - lam) * hx
    mirrored = bits[:, ::-1] @ (1 << np.arange(L - 1, -1, -1))
    reps = np.unique(np.maximum(states, mirrored))[::-1]
    V = np.zeros((2**L, len(reps)))
    V[reps, np.arange(len(reps))] = 1
    V[mirrored[reps], np.arange(len(reps))] = 1
    V /= np.linalg.norm(V, axis=0)
    return V.T @ H @ V


def ising(hz, L=6, hx=1.0):
    H_d = ising_numpy(0, L, hx, hz)
    return H_d, ising_numpy(1, L, hx, hz) - H_d


SYSTEMS = [
    dict(
        name="Landau–Zener",
        build=landau_zener,
        x0=-10,
        xf=10,
        state=0,
        knob=r"$\Delta$",
        values=[0.1, 0.2, 0.5, 1.0, 2.0],
        ref=0.5,
        xlabel="$z$",
    ),
    dict(
        name="Level through a band",
        build=level_through_band,
        x0=-6,
        xf=6,
        state=4,
        knob="$g$",
        values=[0.03, 0.06, 0.12, 0.25, 0.5],
        ref=0.12,
        xlabel="$x$",
    ),
    dict(
        name="DQD (5 levels)",
        build=dqd,
        x0=15,
        xf=0,
        state=0,
        knob=r"$\delta E_x$",
        values=[0.01, 0.03, 0.1, 0.3],
        ref=0.03,
        xlabel=r"$\varepsilon$",
    ),
    dict(
        name="Ising chain ($L=6$)",
        build=ising,
        x0=0,
        xf=1,
        state=0,
        knob="$h_z$",
        values=[0.2, 0.5, 0.8, 0.9, 0.95],
        ref=0.8,
        xlabel=r"$\lambda$",
    ),
]


def make_model(system, value, metric):
    H_d, H_c = system["build"](value)
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(
        control_name="x",
        pulse_initial=system["x0"],
        pulse_final=system["xf"],
        initial_state=system["state"],
        **METRICS[metric],
    )
    model.solve_problem(pulse_accuracy=PULSE_ACCURACY)
    return model


def min_gap(model, state):
    """Smallest gap from the occupied (adiabatic) level to its neighbours, and where it occurs."""
    E = model.eigenenergies
    gaps = np.full(E.shape[0], np.inf)
    if state > 0:
        gaps = np.minimum(gaps, E[:, state] - E[:, state - 1])
    if state < E.shape[1] - 1:
        gaps = np.minimum(gaps, E[:, state + 1] - E[:, state])
    i = int(np.argmin(gaps))
    return gaps[i], model.control_pulse[i]


def slowdown(model, system):
    """(x_f - x_0) dtau/dx on the control grid: 1 for the linear ramp."""
    return abs(system["xf"] - system["x0"]) * np.sqrt(model._metric_tensor) / model._a_tilde


def shortest_duration(system, value, metric):
    """Shortest duration (within a factor T_FACTOR, then bisected) with 1 - F <= EPS; nan above T_MAX."""
    model = make_model(system, value, metric)

    def infidelity(T):
        return 1 - gq.Dynamics(model, T).state_fidelity()

    lower, T = None, T_MIN
    while T <= T_MAX:
        if infidelity(T) <= EPS:
            break
        lower, T = T, T * T_FACTOR
    else:
        return np.nan
    if lower is None:
        return T
    for _ in range(6):  # bisection in log T
        mid = np.sqrt(lower * T)
        lower, T = (lower, mid) if infidelity(mid) <= EPS else (mid, T)
    return T


def scan_durations():
    jobs = [(s, v, m) for s, system in enumerate(SYSTEMS) for v in system["values"] for m in METRICS]
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(shortest_duration)(SYSTEMS[s], v, m) for s, v, m in jobs
    )
    table = {}
    for (s, v, m), T in zip(jobs, results, strict=True):
        table[f"{s}|{v}|{m}"] = T
    np.savez(CACHE, keys=np.array(list(table)), values=np.array(list(table.values())))
    return table


def load_scan():
    data = np.load(CACHE)
    return dict(zip(data["keys"].tolist(), data["values"], strict=True))


# ----- Figures --------------------------------------------------------------------------------------------------------
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


def figure_profiles():
    fig, axes = plt.subplots(len(SYSTEMS), 3, figsize=(7.5, 2.0 * len(SYSTEMS)), layout="constrained")
    for row, system in zip(axes, SYSTEMS, strict=True):
        models = {m: make_model(system, system["ref"], m) for m in METRICS}
        reference = models["QMT (2, 2)"]
        x, E = reference.control_pulse, reference.eigenenergies
        gap, x_gap = min_gap(reference, system["state"])

        ax = row[0]
        ax.plot(x, E, color="0.7", lw=0.8)
        ax.plot(x, E[:, system["state"]], color="k", lw=1.25)
        if E.shape[1] > 12:  # Ising: show the low-lying part only
            lo, hi = E[:, 0].min(), E[:, min(8, E.shape[1] - 1)].max()
            ax.set_ylim(lo - 0.05 * (hi - lo), hi)
        ax.set_xlabel(system["xlabel"])
        ax.set_ylabel("$E$")
        ax.set_title(f"{system['name']}, {system['knob']}$={system['ref']}$\nmin gap {gap:.3g}", fontsize=8)

        for m, model in models.items():
            row[1].plot(x, slowdown(model, system), label=m, **STYLES[m])
            row[2].plot(model.tau, model.control_sol, label=m, **STYLES[m])
        for ax in row[1:2]:
            ax.axvline(x_gap, color="0.8", lw=0.8, zorder=0)
        row[1].set_yscale("log")
        row[1].set_xlabel(system["xlabel"])
        row[1].set_ylabel(r"$(x_f-x_0)\,\mathrm{d}\tau/\mathrm{d}x$")
        row[2].set_xlabel(r"$\tau = t/t_f$")
        row[2].set_ylabel(system["xlabel"])
        for ax in row:
            ax.grid(True)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METRICS), frameon=False)
    return fig


def figure_gap_scan(table):
    fig, axes = plt.subplots(2, len(SYSTEMS), figsize=(7.5, 4.2), layout="constrained")
    for col, (s, system) in enumerate(enumerate(SYSTEMS)):
        gaps, peaks = [], {m: [] for m in METRICS}
        for v in system["values"]:
            for m in METRICS:
                model = make_model(system, v, m)
                peaks[m].append(slowdown(model, system).max())
            gaps.append(min_gap(model, system["state"])[0])
        gaps = np.array(gaps)
        for m in METRICS:
            axes[0, col].plot(gaps, peaks[m], marker="o", ms=3, label=m, **STYLES[m])
            T = np.array([table[f"{s}|{v}|{m}"] for v in system["values"]])
            axes[1, col].plot(gaps, T, marker="o", ms=3, label=m, **STYLES[m])
        axes[0, col].set_title(system["name"], fontsize=8)
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both")
        axes[1, col].set_xlabel(r"min gap $\Delta E_\mathrm{min}$")
    axes[0, 0].set_ylabel("max slowdown")
    axes[1, 0].set_ylabel(rf"$t_f$ for $1-\mathcal{{F}}\leq {EPS:g}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METRICS), frameon=False)
    return fig


if __name__ == "__main__":
    out = HERE / "figures"
    figure_profiles().savefig(out / "metric_comparison_profiles.png", dpi=200)
    table = scan_durations() if "--rescan" in sys.argv or not CACHE.exists() else load_scan()
    figure_gap_scan(table).savefig(out / "metric_comparison_gap_scan.png", dpi=200)
