# API overview

The supported top-level API is exported from `geodesiq`:

- `ControlModel`: validate a parameter-dependent Hermitian Hamiltonian, compute its eigensystem and geodesic metric, and
  solve the optimal dimensionless trajectory.
- `PulseControl`: represent a solved trajectory at a physical duration, resample it, inspect its spectrum, filter it,
  plot it, and export it without pickle.
- `Dynamics`: evaluate closed- or open-system dynamics of a `ControlModel`, driven either by its solved optimal pulse or
  by a custom pulse (e.g. a filtered `PulseControl` pulse).
- Typed exception and warning hierarchies rooted at `GeodesiQError` and `GeodesiQWarning`.

## Model lifecycle

```python
import numpy as np
from geodesiq import ControlModel, Dynamics


def hamiltonian(z: float, coupling: float) -> np.ndarray:
    return np.array([[z, 1j * coupling], [-1j * coupling, -z]], dtype=complex)


model = ControlModel(hamiltonian)
model.set_parameters(coupling=0.5)
model.set_control(control_name="z", pulse_initial=-3.0, pulse_final=3.0, initial_state=0, final_state=0, alpha=2.0,
                  beta=2.0, num_steps=129, )
model.solve_problem(pulse_accuracy=500)
pulse = model.synthesize_pulse(duration=10.0)
dynamics = Dynamics(model, duration=10.0)
```

Arrays returned by public properties are defensive copies. The `parameters` property is read-only. A synthesized pulse
is available only after `synthesize_pulse(duration=...)`; the package never guesses a physical duration.

`model.tau` is the normalized time grid, `tau = t / t_f`, for `model.control_sol`, matching the manuscript's
notation. Both properties solve the problem lazily when needed and return defensive copies. `PulseControl.times`
and `Dynamics.times` contain physical times.

## Dynamics with optimal or custom pulses

`Dynamics(model, ...)` accepts exactly one pulse source:

- `duration`: the solved optimal pulse of `model` is rescaled to the physical duration.
- `times` and `pulse` (keyword-only, given together): an arbitrary pulse sampled at strictly increasing physical times,
  for example the output of `PulseControl.filtered_pulse()` or `PulseControl.discretized_pulse()`. The model does not
  need to be solved in this case.

```python
optimal = Dynamics(model, duration=10.0)

times, filtered = pulse.filtered_pulse(cutoff_freq=2.0)
filtered_dynamics = Dynamics(model, times=times, pulse=filtered)
```

In both cases the default initial and target states of `state_fidelity()` are the eigenstates of the Hamiltonian at
`pulse_initial` and `pulse_final` of the model, so fidelities of different pulses are directly comparable. If the model
has no control endpoints configured, the endpoints of the custom pulse are used instead.

Between its samples, the pulse is interpolated with a cubic spline for the optimal pulse and linearly for a custom pulse,
so that the piecewise-linear `discretized_pulse()` is simulated as the waveform it describes (a cubic spline through a
few samples can deviate strongly between them and overshoot the sampled range). Pass `order` to choose explicitly:
`0` holds each sample until the next one, as an arbitrary waveform generator does, `1` is linear and `3` is cubic.

```python
times, ramps = pulse.discretized_pulse(linear_steps=5)
awg = Dynamics(model, times=times, pulse=ramps, order=0)  # sample-and-hold
```

## Dynamics solver accuracy

Pass `options` to the constructor to configure QuTiP propagation for state fidelity, populations and evolution
operators, including the operators computed by `average_gate_fidelity()`:

```python
accurate = Dynamics(model, duration=10.0,
                    options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10000, "max_step": 0.01, "method": "adams"})
final_operator = accurate.time_evolution_operator(final_only=True)[0]
```

`options=None` keeps QuTiP's numerical defaults; tighter settings are opt-in. The dictionary is deep-copied at
construction and for each solver call. The methods override `store_states` and, when required, `store_final_state`
to preserve their documented outputs. QuTiP checks option names and method-specific support when evolution is
requested; rejected configurations raise `ValidationError`. Integration failures raise `SolverError`, chained from
QuTiP's original exception; increasing `nsteps` can resolve an internal step-count limit.

Check convergence by tightening `atol` and `rtol` and reducing `max_step`. Choose `max_step` below half the width of
the narrowest pulse feature; output sample times alone do not bound internal integration steps. Supported settings
depend on the method; see the [QuTiP solver options](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-options.html).

QuTiP normalizes evolved states by default, which can hide norm drift. Set `normalize_output=False` to inspect
probability conservation. Propagated operators are not normalized; check their unitarity separately. These controls
address time-integration error: refine pulse sampling, interpolation, the eigensystem/metric grid and SVD truncation
separately when assessing overall accuracy.

## State populations

`Dynamics.populations()` returns an array of shape `(len(times), dimension)` with the population of every state during
the evolution. With `basis="adiabatic"` (default) populations are measured in the instantaneous eigenstates of the
Hamiltonian, sorted by increasing energy; with `basis="diabatic"` they are measured in the basis of the Hamiltonian
matrix. `Dynamics.plot_populations()` plots them over time:

```python
populations = dynamics.populations(initial_state=0)  # populations[-1, 1] == dynamics.state_fidelity(0, 1)

fig, ax = dynamics.plot_populations(basis="diabatic", states=[0, 1])
```

## Numerical contracts

Hamiltonians must be finite, non-empty, square, Hermitian, and dimensionally stable over the sweep. Analytical
derivatives must match the Hamiltonian shape. Singular metrics, exact degeneracies relevant to a requested transition,
non-finite solver output, incomplete solver time grids, and invalid endpoint completion raise package-specific
exceptions before downstream interpolation or propagation.

State indices refer to the ascending energy spectrum: `initial_state` at `pulse_initial` and `final_state` at
`pulse_final`. Along the grid, the occupied state is followed by eigenvector overlap. At a true (symmetry-protected)
level crossing, where `dH/dx` does not couple the two states, it keeps its identity and changes its energy index, also
when the crossing falls on a grid point. An anticrossing that the grid does not resolve keeps the energy order. As long
as `final_state` has not been set, it is the index in which the followed state ends.

The numerical derivative accepts complex-valued Hamiltonians and preserves their complex dtype. Supplying an analytical
derivative remains preferable for performance and controlled accuracy.

## Threading

For Hamiltonians of dimension up to 256, the diagonalization and matrix products in `ControlModel` and `Dynamics`
run single-threaded, because for small matrices the threading overhead of BLAS outweighs its gain. The limit is set
with `threadpoolctl` on the BLAS libraries themselves, so while it is active it applies to the whole process: BLAS
calls made from other threads at the same time also run on one thread. To solve several models in parallel, use
processes (e.g. `multiprocessing`) rather than threads; larger Hamiltonians are not limited.
