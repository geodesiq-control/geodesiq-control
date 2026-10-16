"""Model and worker helpers for FourLevelSystem.ipynb.

Windows can only ``spawn`` worker processes, which start a fresh interpreter and
cannot see functions defined inside a notebook. Everything the multiprocessing
pool needs therefore lives in this importable module.
"""

import warnings

import numpy as np

import geodesiq as gq


def bb_hamiltonian(eps, DeltaL, DeltaR, tee, teg):
    tgg = np.conjugate(tee)
    tge = -np.conjugate(teg)

    H = np.array(
        [
            [eps / 2 + np.abs(DeltaL), 0, tee, teg],
            [0, eps / 2 - np.abs(DeltaL), tge, tgg],
            [np.conjugate(tee), np.conjugate(tge), -eps / 2 + np.abs(DeltaR), 0],
            [np.conjugate(teg), np.conjugate(tgg), 0, -eps / 2 - np.abs(DeltaR)],
        ]
    )

    return H


def build_model(DeltaL, DeltaR, tee, teg, eps0, epsf, alpha, beta, dia_alpha, dia_beta):
    bb_model = gq.ControlModel(H_c=bb_hamiltonian(1, 0, 0, 0, 0), H_d=bb_hamiltonian(0, DeltaL, DeltaR, tee, teg))
    bb_model.set_control(
        control_name="eps",
        pulse_initial=eps0,
        pulse_final=epsf,
        initial_state=1,
        final_state=2,
        alpha=alpha,
        beta=beta,
        dia_alpha=dia_alpha,
        dia_beta=dia_beta,
    )
    return bb_model


def silence_warnings():
    warnings.filterwarnings("ignore", category=gq.GeodesiQWarning)
    # optional: numpy overflow/invalid-value warnings from extreme alpha/beta
    warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------- optimisation objective ---
def infidelity(x, model_params, param_names, duration=None):
    """1 - F, with ``x`` mapped onto ``param_names`` (``set_control`` kwargs).

    If ``"duration"`` is among ``param_names`` the total time is optimised too and ``duration`` is ignored;
    otherwise ``duration`` sets the fixed total time.

    Bind the fixed arguments with ``functools.partial``: the result is picklable, so scipy's ``workers=``
    can run it in spawned processes. The model is rebuilt per call, which is negligible next to solving.
    Failed or non-finite evaluations count as the worst value, 1.
    """
    params = dict(zip(param_names, x, strict=True))
    duration = params.pop("duration", duration)
    if duration is None:
        raise ValueError('Pass `duration` or include "duration" in `param_names`.')

    with warnings.catch_warnings():
        silence_warnings()
        try:
            model = build_model(**model_params)
            model.set_control(**params)
            model.solve_problem()
            fidelity = float(gq.Dynamics(model, duration).state_fidelity())
        except Exception:
            return 1.0
    return 1.0 - fidelity if np.isfinite(fidelity) else 1.0
