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

Used by metric_appendix_data.py, which takes the models, the local slowdown (x_f - x_0) dtau/dx =
(x_f - x_0) sqrt(G) / a_tilde and the cached duration scan from here.

Run from dev/:  python metric_comparison.py
scans the shortest duration with 1 - F <= EPS for every system, gap and metric into data/metric_comparison_scan.npz.
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

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
    dict(build=landau_zener, x0=-10, xf=10, state=0, values=[0.1, 0.2, 0.5, 1.0, 2.0]),  # Delta
    dict(build=level_through_band, x0=-6, xf=6, state=4, values=[0.03, 0.06, 0.12, 0.25, 0.5]),  # g
    dict(build=dqd, x0=15, xf=0, state=0, values=[0.01, 0.03, 0.1, 0.3]),  # delta E_x
    dict(build=ising, x0=0, xf=1, state=0, values=[0.2, 0.5, 0.8, 0.9, 0.95]),  # h_z
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


if __name__ == "__main__":
    scan_durations()
