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

The numerical derivative accepts complex-valued Hamiltonians and preserves their complex dtype. Supplying an analytical
derivative remains preferable for performance and controlled accuracy.
