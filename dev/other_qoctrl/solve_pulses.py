import numpy as np
import qutip as qt
import qutip_qtrl.pulseoptim as cpo

import geodesiq as gq


def compute_degradation(H_c, H_d, psi0, psit, pulse, time, cutoff=None):
    values = np.asarray(pulse, dtype=float).copy()

    if cutoff is not None:
        pulse_instance = gq.PulseControl(values, time[-1])
        values = pulse_instance.filtered_pulse(cutoff_freq=cutoff)[1]

    dt = float(np.min(np.diff(time)))
    coeff = qt.coefficient(values, tlist=time, order=0)
    H = qt.QobjEvo([H_d, [H_c, coeff]])

    options = {"atol": 1e-11, "rtol": 1e-11, "max_step": dt / 2, "store_states": False, "store_final_state": True}
    result = qt.sesolve(H, psi0, time, options=options)

    return qt.fidelity(result.final_state, psit) ** 2


def solve_GRAPE(H_c,
                H_d,
                psi0,
                psit,
                tf,
                n_ts,
                amp_lbound,
                amp_ubound,
                fid_err_targ,
                init_amps=None,
                samples_per_slot=40,
                verbose=False,
                return_goal_achieved=False
                ):
    if init_amps is None:
        init_amps = np.zeros(n_ts)

    if init_amps.ndim == 1:
        init_amps = init_amps[:, None]

    optim_GRAPE = cpo.create_pulse_optimizer(H_d, [H_c], psi0, psit, num_tslots=n_ts, evo_time=tf,
                                             amp_lbound=amp_lbound, amp_ubound=amp_ubound, fid_err_targ=fid_err_targ,
                                             max_iter=2000, alg="GRAPE", dyn_type="UNIT", )

    dyn = optim_GRAPE.dynamics
    dyn.init_timeslots()

    init_amps[0] = amp_lbound
    init_amps[-1] = amp_ubound
    dyn.initialize_controls(init_amps)

    # Bounds for every optimization variable
    optim_GRAPE.bounds = [(amp_lbound, amp_ubound)] * n_ts

    # Fix the first and last control amplitudes exactly
    optim_GRAPE.bounds[0] = (amp_lbound, amp_lbound)
    optim_GRAPE.bounds[-1] = (amp_ubound, amp_ubound)

    result_GRAPE = optim_GRAPE.run_optimization()

    amps = result_GRAPE.final_amps[:, 0]

    grape_pulse = np.append(np.repeat(amps, samples_per_slot), amps[-1])
    grape_time = np.linspace(result_GRAPE.time[0], result_GRAPE.time[-1], len(grape_pulse))

    if verbose:
        print("goal:", result_GRAPE.goal_achieved)
        print("error:", result_GRAPE.fid_err)
        print("reason:", result_GRAPE.termination_reason)
        print("initial:", result_GRAPE.final_amps[0, 0])
        print("final:", result_GRAPE.final_amps[-1, 0])

    if return_goal_achieved:
        return grape_time, grape_pulse, result_GRAPE.goal_achieved
    else:
        return grape_time, grape_pulse


def solve_CRAB(H_c,
               H_d,
               psi0,
               psit,
               tf,
               amp_lbound,
               amp_ubound,
               fid_err_targ,
               num_tslots=1000,
               n_super_iterations=10,
               num_coeffs=10,
               correction_scale=0.1,
               print_every=40,
               verbose=False,
               return_goal_achieved=False, ):
    error_history = []
    super_iteration_history = []

    best_error = np.inf
    best_pulse = None
    best_result = None

    linear_pulse = np.linspace(amp_lbound, amp_ubound, num_tslots)

    def inverse_bounds(pulse, lbound, ubound, eps=1e-8):
        mean = 0.5 * (ubound + lbound)
        scale = 0.5 * (ubound - lbound)

        x = (pulse - mean) / scale
        x = np.clip(x, -1 + eps, 1 - eps)

        return np.arctanh(x)

    for super_iter in range(n_super_iterations):
        optim_CRAB = cpo.create_pulse_optimizer(H_d, [H_c], psi0, psit, num_tslots=num_tslots, evo_time=tf,
                                                amp_lbound=amp_lbound, amp_ubound=amp_ubound, fid_err_targ=fid_err_targ,
                                                max_iter=2000, alg="CRAB", dyn_type="UNIT",
                                                alg_params={"num_coeffs": num_coeffs}, )

        dyn = optim_CRAB.dynamics
        dyn.init_timeslots()

        pgen = optim_CRAB.pulse_generator[0]

        # First super-iteration: start around a linear pulse.
        # Later super-iterations: start around the current best pulse.
        if best_pulse is None:
            base_pulse = linear_pulse
        else:
            base_pulse = best_pulse

        pgen.guess_pulse = inverse_bounds(base_pulse, amp_lbound, amp_ubound)
        pgen.guess_pulse_action = "ADD"
        pgen.scaling = correction_scale
        pgen.init_pulse()

        init_amps = pgen.gen_pulse()[:, None]
        dyn.initialize_controls(init_amps)

        original_fid_func = optim_CRAB.fid_err_func_wrapper

        run_state = {"evaluations": 0, "best_error": np.inf, }

        def fid_err_func(x):
            error = original_fid_func(x)

            run_state["evaluations"] += 1
            run_state["best_error"] = min(run_state["best_error"], error)
            error_history.append(error)

            if verbose and run_state["evaluations"] % print_every == 0:
                print(f"\r\033[K"
                      f"Super-iteration {super_iter + 1}/{n_super_iterations} | "
                      f"Evaluation {run_state['evaluations']:5d} | "
                      f"error = {error:.6e} | "
                      f"best = {run_state['best_error']:.6e}", end="", flush=True, )

            return error

        optim_CRAB.fid_err_func_wrapper = fid_err_func

        result = optim_CRAB.run_optimization()

        current_pulse = np.squeeze(result.final_amps)
        current_error = result.fid_err

        if current_error < best_error:
            best_error = current_error
            best_pulse = current_pulse.copy()
            best_result = result

        super_iteration_history.append(best_error)

        if verbose:
            print(f"\r\033[K"
                  f"Super-iteration {super_iter + 1}/{n_super_iterations} finished | "
                  f"error = {current_error:.6e} | "
                  f"global best = {best_error:.6e}", end="", flush=True, )

        if best_error <= fid_err_targ:
            break

    if verbose:
        print("\r\033[K", end="", flush=True)

    result_CRAB = best_result
    crab_time = np.linspace(0, tf, len(best_pulse) + 1)
    crab_pulse = np.append(best_pulse, best_pulse[-1])
    goal = best_error <= fid_err_targ

    if verbose:
        print("goal:", goal)
        print("error:", best_error)
        print("reason:", result_CRAB.termination_reason)

    if return_goal_achieved:
        return crab_time, crab_pulse, goal
    else:
        return crab_time, crab_pulse


def solve_gq(H_c,
             H_d,
             tf,
             index_0,
             alpha,
             beta,
             amp_lbound,
             amp_ubound,
             fid_err_targ,
             n_ts=2 ** 10 + 1,
             pulse_accuracy=1000,
             verbose=False,
             return_goal_achieved=False):
    model = gq.ControlModel(H_c=H_c[:], H_d=H_d[:])
    model.set_control(control_name="z", pulse_initial=amp_lbound, pulse_final=amp_ubound, initial_state=index_0,
                      alpha=alpha, beta=beta, num_steps=n_ts, )

    # ----- Solve for optimal pulse -----
    model.solve_problem(pulse_accuracy=pulse_accuracy)

    dynamics = gq.Dynamics(tf, model)
    gq_error = 1 - np.sqrt(dynamics.state_fidelity())

    gq_pulse = model.synthesize_pulse(tf).pulse
    gq_time = np.linspace(0, tf, len(gq_pulse))
    goal = gq_error <= fid_err_targ

    if verbose:
        print("goal:", goal)
        print("error:", gq_error)

    if return_goal_achieved:
        return gq_time, gq_pulse, goal
    else:
        return gq_time, gq_pulse
