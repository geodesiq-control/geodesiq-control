import numpy as np

from tqdm.auto import tqdm
import itertools

from joblib import Parallel, delayed

import geodesiq as gq


def dqd_hamiltonian(eps, U, tc, Ez, dEz, dEx):
    ham = np.array([[U - eps, 0, -tc, tc, 0], [0, Ez, dEx, -dEx, 0], [-tc, dEx, dEz, 0, dEx], [tc, -dEx, 0, -dEz, -dEx],
                    [0, 0, dEx, -dEx, -Ez]])
    return ham


def fidelity_vs_time(durations, model, alpha=2, beta=2):
    try:
        model.set_control(alpha=alpha, beta=beta)
        model.solve_problem(pulse_accuracy=int(100))  # pulse_accuracy=int(1000) for recovering the paper figure

        fidelities = []
        for duration in tqdm(durations, disable=True):
            dynamics = gq.Dynamics(duration=duration, model=model)
            fidelities.append(dynamics.state_fidelity())

        return np.array(fidelities)
    except gq.NumericalStabilityWarning as e:
        print(e)
        print("NumericalStabilityWarning: Fidelity computation failed for alpha={}, beta={}".format(alpha, beta))


def optimal_fidelity_time_map(durations, model, alphas, betas, n_jobs=-1):
    """
    Function that computes the fidelity map for a range of alpha and beta values over a range of durations.
    The function relies on the definition of fidelity_vs_time() from above.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(fidelity_vs_time)(durations, model, alpha=alpha, beta=beta) for alpha, beta in
        tqdm(list(itertools.product(alphas, betas)), desc="Running 2d sweep..."))

    fidelities = np.array(results).reshape(len(alphas), len(betas), len(durations))

    max_fidelities = np.max(fidelities, axis=2)
    best_times = durations[np.argmax(fidelities, axis=2)]

    return max_fidelities, best_times


dqd_model = gq.ControlModel(dqd_hamiltonian)

# ----- Set system and control parameters -----

U, tc, Ez, dEz, dEx = 10, 1, .9, .1, .01
eps0, epsf = 15, 0

dqd_model.set_parameters(U=U, tc=tc, Ez=Ez, dEz=dEz, dEx=dEx)
dqd_model.set_control(control_name='eps', pulse_initial=eps0, pulse_final=epsf, initial_state=0)

# ----- Solve for optimal pulse -----
durations_2d = np.linspace(0.01, 1000, 2)

alphas = np.linspace(0, 4, 7)
betas = np.linspace(0, 4, 7)

# fidelity_vs_time(durations_2d, dqd_model, alpha=3 + 1 / 3, beta=4)
optimal_fidelities, best_times = optimal_fidelity_time_map(durations_2d, dqd_model, alphas, betas, n_jobs=1)
