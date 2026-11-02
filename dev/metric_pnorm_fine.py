"""
Fine scan of the p-norm example (metric_pnorm_example.py) over n_+ = alpha + beta, 1/p and the total time t_f, read
by metric_appendix_data.py.

alpha = beta = n_+ / 2 (the register's metric depends on alpha + beta only). The norm axis is 1/p in [0, 1]: p = inf at
0, p = 2 at 0.5, p = 1 at 1.

Data (dev/data/metric_pnorm_fine/, not tracked by git):
- config.json     system parameters; the cache is refused if they change (delete the folder to start over).
- pulses/*.npy    control_sol(tau) of every (n_+, 1/p), tau = linspace(0, 1, PULSE_ACCURACY).
- results.npz     long table: n_plus, inv_p, duration, infidelity (one row per simulation).
Changing N_PLUS, INV_P or DURATIONS and running `compute` again only simulates the missing rows.

Run from dev/ (pulses + dynamics, saved every SAVE_EVERY pulses, with progress and ETA):
    python metric_pnorm_fine.py
"""

import json
import time
import warnings
from pathlib import Path

import metric_pnorm_example as ex
import numpy as np
from joblib import Parallel, delayed

import geodesiq as gq

warnings.simplefilter("ignore", gq.GeodesiQWarning)

HERE = Path(__file__).parent
STORE = HERE / "data" / "metric_pnorm_fine"
PULSES = STORE / "pulses"
RESULTS = STORE / "results.npz"

N_PLUS = np.round(np.arange(0.1, 8.001, 0.1), 3)
INV_P = np.round(np.arange(0, 1.001, 0.05), 3)
DURATIONS = np.round(np.geomspace(30, 3000, 41), 6)
SAVE_EVERY = 48  # pulses per batch between saves

CONFIG = dict(
    n_cluster=ex.N_CLUSTER,
    rabi=ex.RABI,
    omega_b=ex.OMEGA_B,
    spread=ex.SPREAD,
    x0=ex.X0,
    xf=ex.XF,
    pulse_accuracy=ex.PULSE_ACCURACY,
    options=ex.OPTIONS,
    exponents="alpha = beta = n_plus / 2",
)


# ----- Storage --------------------------------------------------------------------------------------------------------
def open_store():
    PULSES.mkdir(parents=True, exist_ok=True)
    (STORE / ".gitignore").write_text("*\n")
    config_file = STORE / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != json.loads(json.dumps(CONFIG)):
            raise SystemExit(f"{config_file} differs from CONFIG: delete {STORE} to start a new scan.")
    else:
        config_file.write_text(json.dumps(CONFIG, indent=2))


def pulse_file(n_plus, inv_p):
    return PULSES / f"nplus_{n_plus:.3f}_invp_{inv_p:.3f}.npy"


def load_results():
    if not RESULTS.exists():
        return {}
    data = np.load(RESULTS)
    keys = zip(data["n_plus"], data["inv_p"], data["duration"], strict=True)
    return {(round(float(n), 3), round(float(q), 3), round(float(T), 6)): v for (n, q, T), v in
            zip(keys, data["infidelity"], strict=True)}


def save_results(results):
    keys = np.array(list(results), dtype=float).reshape(-1, 3)
    tmp = RESULTS.with_suffix(".tmp.npz")
    np.savez(tmp, n_plus=keys[:, 0], inv_p=keys[:, 1], duration=keys[:, 2], infidelity=np.array(list(results.values())))
    tmp.replace(RESULTS)


# ----- Computation ----------------------------------------------------------------------------------------------------
def compute_pulses(cells):
    """Missing pulses, from one register model: the eigenproblem is solved once, only the metric and the ODE change."""
    missing = [c for c in cells if not pulse_file(*c).exists()]
    if not missing:
        return
    H_d, H_c = ex.register(ex.frequencies(ex.N_CLUSTER))
    model = gq.ControlModel(H_d=H_d, H_c=H_c)
    model.set_control(control_name="x", pulse_initial=ex.X0, pulse_final=ex.XF, initial_state=0, alpha=1, beta=1)
    start = time.perf_counter()
    for i, (n, q) in enumerate(missing, 1):
        model.set_control(alpha=n / 2, beta=n / 2, norm=np.inf if q == 0 else 1 / q)
        model.solve_problem(pulse_accuracy=ex.PULSE_ACCURACY)
        np.save(pulse_file(n, q), model.control_sol)
        if i % 50 == 0 or i == len(missing):
            elapsed = time.perf_counter() - start
            print(f"pulses {i}/{len(missing)}  elapsed {elapsed / 60:.1f} min  "
                  f"ETA {elapsed / i * (len(missing) - i) / 60:.1f} min", flush=True)


def simulate(n, q, durations):
    x = np.load(pulse_file(n, q))
    tau = np.linspace(0, 1, x.size)
    qubits = [ex.make_model([w], alpha=0, beta=0) for w in ex.frequencies(ex.N_CLUSTER)]
    out = {}
    for T in durations:
        fidelity = np.prod([gq.Dynamics(qb, times=tau * T, pulse=x, options=ex.OPTIONS).state_fidelity() for qb in qubits])
        out[n, q, T] = 1 - fidelity
    return out


def compute():
    open_store()
    cells = [(float(n), float(q)) for n in N_PLUS for q in INV_P]
    compute_pulses(cells)
    results = load_results()
    todo = {}
    for n, q in cells:
        missing = [float(T) for T in DURATIONS if (n, q, float(T)) not in results]
        if missing:
            todo[n, q] = missing
    total = sum(len(v) for v in todo.values())
    print(f"{len(results)} simulations cached, {total} to run for {len(todo)} pulses", flush=True)
    jobs, done, start = list(todo.items()), 0, time.perf_counter()
    with Parallel(n_jobs=-1, return_as="generator_unordered") as parallel:
        for b in range(0, len(jobs), SAVE_EVERY):
            for out in parallel(delayed(simulate)(n, q, Ts) for (n, q), Ts in jobs[b : b + SAVE_EVERY]):
                results.update(out)
                done += len(out)
            save_results(results)
            elapsed = time.perf_counter() - start
            print(f"simulations {done}/{total}  elapsed {elapsed / 60:.1f} min  "
                  f"ETA {elapsed / done * (total - done) / 60:.1f} min", flush=True)


def grid(results):
    """(n_plus, inv_p, duration) array of the stored results on their own sorted axes."""
    keys = np.array(list(results))
    axes = [np.unique(keys[:, i]) for i in range(3)]
    data = np.full([len(a) for a in axes], np.nan)
    for (n, q, T), v in results.items():
        data[np.searchsorted(axes[0], n), np.searchsorted(axes[1], q), np.searchsorted(axes[2], T)] = v
    return (*axes, data)


if __name__ == "__main__":
    compute()
