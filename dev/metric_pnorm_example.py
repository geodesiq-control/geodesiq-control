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

Used by metric_appendix_data.py and metric_pnorm_fine.py.
"""

import warnings
from functools import reduce

import numpy as np

import geodesiq as gq

warnings.simplefilter("ignore", gq.GeodesiQWarning)

RABI, OMEGA_B, SPREAD = 0.2, 10.0, 0.05
X0, XF = -10.0, 20.0
N_CLUSTER = 8
PULSE_ACCURACY = 20001
OPTIONS = dict(atol=1e-12, rtol=1e-10, nsteps=10**4)
NORMS = {"$p=1$": 1, "$p=2$": 2, r"$p=\infty$": np.inf}

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
