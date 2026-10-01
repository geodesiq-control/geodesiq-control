# geodesiq: geometric optimal control

`geodesiq` computes optimal control pulses for the parameters of generic quantum Hamiltonians. The pulse follows the
geodesic of a metric built from the energy gaps and matrix elements of the Hamiltonian, so the evolution slows down
where transitions are likely and speeds up elsewhere.

## Installation

```bash
pip install geodesiq
```

## Example

The two-level Landau-Zener problem $H[z(t)] = z(t)\,\sigma_z + x\,\sigma_x$ with control parameter $z(t)$:

```python
import numpy as np
from geodesiq import ControlModel, Dynamics


def H_fun(x, z):
    return np.array([[z, x], [x, -z]])


def H_partial(x, z):
    return np.array([[1, 0], [0, -1]])


model = ControlModel(H_fun, H_partial)
model.set_parameters(x=1)
model.set_control(control_name="z", pulse_initial=-10, pulse_final=10, initial_state=0, alpha=2, beta=2)
model.solve_problem()

duration = 10
pulse = model.synthesize_pulse(duration=duration)

dynamics = Dynamics(model, duration=duration)
print(dynamics.state_fidelity())  # 0.999791

times, filtered = pulse.filtered_pulse(cutoff_freq=2.0)
print(Dynamics(model, times=times, pulse=filtered).state_fidelity())  # 0.999960
```

If the Hamiltonian is affine in the control, `H(z) = H_d + z H_c`, pass the matrices instead of a function,
`ControlModel(H_d=..., H_c=...)`, which is considerably faster.

## Where to go next

- The [user guide](api.md) describes the model lifecycle, the two ways of driving `Dynamics`, and the numerical
  contracts.
- The API reference documents every public class and function.
- The [`examples/`](https://github.com/geodesiq-control/geodesiq-control/tree/main/examples) notebooks cover the
  Landau-Zener, double-quantum-dot and many-body problems.

## Citing geodesiq

If you use `geodesiq` in your research, please cite it. The citation metadata is in
[`CITATION.cff`](https://github.com/geodesiq-control/geodesiq-control/blob/main/CITATION.cff).
