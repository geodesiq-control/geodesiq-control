"""
An example where the norm order p of the metric makes a huge difference: global adiabatic inversion of a
frequency-crowded qubit register with one chirped drive.

N qubits with nearly equal frequencies (a cluster, omega_k in [-0.05, 0.05]) and one isolated qubit (omega_B = 10),
all driven with the same Rabi frequency Omega = 0.2. In the rotating frame of the drive at frequency x,
    H(x) = sum_k [(omega_k - x)/2 sigma_z^(k) + Omega/2 sigma_x^(k)],   x: -10 -> 20,
which is affine in x. The register ground state is the product of the qubit ground states and only couples to the N + 1
single flips, so at the cluster N transitions contribute at once and at omega_B only one does.

The norm weights the cluster as N (p = 1), sqrt(N) (p = 2) or 1 (p = inf) relative to the isolated qubit. The qubits
are independent, so 1 - F = 1 - prod_k F_k is the sum of their leakage probabilities, which is what p = 2 sums.
p = 1 gives the isolated qubit only 1/(N + 1) of the time instead of 1/(sqrt(N) + 1), and since the leakage at an
anticrossing falls exponentially with the time spent there, its infidelity is orders of magnitude larger. p = inf
gives every crossing the same time and pays a factor ~N in the cluster.

The fidelity is computed exactly as the product of single-qubit fidelities under the same pulse; the metric uses the
full register (2^(N+1) levels).

Run from dev/:  python metric_pnorm_example.py
"""

import warnings
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import NullFormatter

import geodesiq as gq

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
RABI, OMEGA_B, SPREAD = 0.2, 10.0, 0.05
X0, XF = -10.0, 20.0
N_CLUSTER = 8
CLUSTER_SIZES = [1, 2, 4, 6, 8]
DURATIONS = np.geomspace(30, 3e3, 17)
FIXED_DURATION = 300.0
PULSE_ACCURACY = 20001
OPTIONS = dict(atol=1e-12, rtol=1e-10, nsteps=10**4)
NORMS = {"$p=1$": 1, "$p=2$": 2, r"$p=\infty$": np.inf}
COLORS = {"$p=1$": "C1", "$p=2$": "C0", r"$p=\infty$": "C2"}
LINESTYLES = {"$p=1$": "--", "$p=2$": "-", r"$p=\infty$": "-."}

SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SZ = np.diag([1.0, -1.0])


def frequencies(n_cluster):
    return [*SPREAD * np.linspace(-1, 1, n_cluster), OMEGA_B]


def register(omegas):
    """H_d and H_c of independent qubits driven at frequency x (rotating frame)."""
    n = len(omegas)

    def local(op, k):
        return reduce(np.kron, [op if j == k else np.eye(2) for j in range(n)])

    H_d = sum(w / 2 * local(SZ, k) + RABI / 2 * local(SX, k) for k, w in enumerate(omegas))
    H_c = sum(-0.5 * local(SZ, k) for k in range(n))
    return H_d, H_c


def make_model(omegas, **metric):
    H_d, H_c = register(omegas)
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(control_name="x", pulse_initial=X0, pulse_final=XF, initial_state=0, **metric)
    model.solve_problem(pulse_accuracy=PULSE_ACCURACY)
    return model


def infidelities(omegas, model, durations):
    """1 - prod_k F_k: every qubit evolves independently under the register's pulse x(t)."""
    qubits = [make_model([w], alpha=0, beta=0) for w in omegas]
    result = []
    for T in durations:
        times, pulse = model.tau * T, model.control_sol
        fidelity = np.prod([gq.Dynamics(q, times=times, pulse=pulse, options=OPTIONS).state_fidelity() for q in qubits])
        result.append(1 - fidelity)
    return result


def slowdown(model):
    return (XF - X0) * np.sqrt(model._metric_tensor) / model._a_tilde


def run(n_cluster, p, durations):
    omegas = frequencies(n_cluster)
    model = make_model(omegas, alpha=2, beta=2, norm=NORMS[p])
    return model.control_pulse, slowdown(model), infidelities(omegas, model, durations)


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
    # The 2^(N+1)-level models keep ~2 GB of matrix elements for N = 8: few workers at a time
    main = dict(zip(NORMS, Parallel(n_jobs=3)(delayed(run)(N_CLUSTER, p, DURATIONS) for p in NORMS), strict=True))
    scan = Parallel(n_jobs=3)(delayed(run)(n, p, [FIXED_DURATION]) for p in NORMS for n in CLUSTER_SIZES)
    scan = np.array([s[2][0] for s in scan]).reshape(len(NORMS), len(CLUSTER_SIZES))

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    (ax_E, ax_v), (ax_F, ax_N) = axes

    # Single-flip excitation energies of the register ground state: one gap per qubit
    x = main["$p=2$"][0]
    for w in frequencies(N_CLUSTER):
        ax_E.plot(x, np.sqrt((w - x) ** 2 + RABI**2), color="C3" if w == OMEGA_B else "0.5", lw=0.8)
    ax_E.set(ylim=(0, 3), xlabel="drive frequency $x$", ylabel=r"$E_k - E_0$")
    ax_E.set_title(f"{N_CLUSTER} crowded qubits (grey) + 1 isolated qubit (red)", fontsize=8)

    for p, (x, v, F) in main.items():
        ax_v.plot(x, v, color=COLORS[p], ls=LINESTYLES[p], label=f"QMT (2, 2), {p}")
        ax_F.plot(DURATIONS, F, color=COLORS[p], ls=LINESTYLES[p])
    for i, p in enumerate(NORMS):
        ax_N.plot(CLUSTER_SIZES, scan[i], marker="o", ms=3, color=COLORS[p], ls=LINESTYLES[p])
    ax_v.set(yscale="log", xlabel="drive frequency $x$", ylabel=r"$(x_f-x_0)\,\mathrm{d}\tau/\mathrm{d}x$")
    ax_v.set_title(f"Local slowdown, $N={N_CLUSTER}$", fontsize=8)
    ax_F.set(xscale="log", yscale="log", xlabel=r"$t_f$", ylabel=r"$1-\mathcal{F}$")
    ax_F.set_title(f"$N={N_CLUSTER}$", fontsize=8)
    ax_N.set(yscale="log", xlabel="crowded qubits $N$", ylabel=r"$1-\mathcal{F}$")
    ax_N.set_xticks(CLUSTER_SIZES)
    ax_N.xaxis.set_minor_formatter(NullFormatter())
    ax_N.set_title(f"$t_f = {FIXED_DURATION:g}$", fontsize=8)
    for ax in axes.flat:
        ax.grid(True)
    handles, labels = ax_v.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(NORMS), frameon=False)
    return fig


if __name__ == "__main__":
    figure().savefig(HERE / "figures" / "metric_pnorm_example.png", dpi=200)
