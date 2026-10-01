<p align="center">
  <img
    src="https://raw.githubusercontent.com/geodesiq-control/geodesiq-control/main/images/geodesiq_logo.png"
    alt="geodesiq logo"
  />
</p>

# `geodesiq`: Geometric optimal control

[D. Fernández Fernández](https://github.com/Davtax),
[C. Ventura Meinersen](https://github.com/cventurameinersen)

[![Build Status](https://github.com/geodesiq-control/geodesiq-control/actions/workflows/ci.yml/badge.svg)](https://github.com/geodesiq-control/geodesiq-control/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/geodesiq/badge/?version=latest)](https://geodesiq.readthedocs.io/en/latest/)
[![Maintainability](https://qlty.sh/gh/geodesiq-control/projects/geodesiq-control/maintainability.svg)](https://qlty.sh/gh/geodesiq-control/projects/geodesiq-control)
[![Coverage Status](https://coveralls.io/repos/github/geodesiq-control/geodesiq-control/badge.svg?branch=dev)](https://coveralls.io/github/geodesiq-control/geodesiq-control?branch=dev)
[![PyPI - License](https://img.shields.io/pypi/l/geodesiq)](https://opensource.org/license/lgpl-2-1)
[![PyPI Downloads](https://api.pepy.tech/badge/geodesiq/month?left_text=downloads%20%7C%20pip)](https://pypi.org/project/geodesiq/)

[//]: # ([![Coverage Status]&#40;https://img.shields.io/coveralls/qutip/qutip.svg?logo=Coveralls&#41;]&#40;https://coveralls.io/r/qutip/qutip&#41;)
[//]: # ([![Maintainability]&#40;https://api.codeclimate.com/v1/badges/df502674f1dfa1f1b67a/maintainability&#41;]&#40;https://codeclimate.com/github/qutip/qutip/maintainability&#41;)


---
[Installation](#installation) | [Example code](#example-code) | [Documentation](https://geodesiq.readthedocs.io) |
[Citing geodesiq](#citing-geodesiq)

`geodesiq` is a Python package for optimal pulse control of Hamiltonian parameters for generic quantum systems.



# Installation

[![Pip Package](https://img.shields.io/pypi/v/geodesiq?logo=PyPI)](https://pypi.org/project/geodesiq)
[![Python](https://img.shields.io/pypi/pyversions/geodesiq?logo=python)](https://pypi.org/project/geodesiq/)

To install `geodesiq`, you can use the standard Python package installer:

```bash
pip install geodesiq
```

# Example code

Here is an example code based on the two-level Landau-Zener problem $H[z (t)]=z (t)\sigma_z+x \sigma_x$ with control
parameter $z (t)$. To compute the optimal pulse, you define the Hamiltonian and, optionally, its partial derivative with
respect to the control parameter.

```python
import numpy as np
from geodesiq import ControlModel, Dynamics


# ----- Define the Hamiltonian and its derivative -----
def H_fun(x, z):
    return np.array([[z, x], [x, -z]])


def H_partial(x, z):
    return np.array([[1, 0], [0, -1]])


model = ControlModel(H_fun, H_partial)

# ----- Set system and control parameters -----
alpha = 2
beta = 2
x = 1
z0 = -10
zf = -z0

model.set_parameters(x=x)
model.set_control(control_name='z', pulse_initial=z0, pulse_final=zf, initial_state=0, alpha=alpha, beta=beta)

# ----- Solve for optimal pulse -----
model.solve_problem()

# ----- Synthesize the pulse for a physical duration -----
duration = 10
pulse = model.synthesize_pulse(duration=duration)

# ----- Simulate the dynamics driven by the optimal pulse -----
dynamics = Dynamics(model, duration=duration)
print(f"Optimal pulse fidelity:  {dynamics.state_fidelity():.6f}")  # 0.999791

# ----- ... or by a low-pass filtered version of it -----
times, filtered = pulse.filtered_pulse(cutoff_freq=2.0)
filtered_dynamics = Dynamics(model, times=times, pulse=filtered)
print(f"Filtered pulse fidelity: {filtered_dynamics.state_fidelity():.6f}")  # 0.999960

# ----- Compare with a linear ramp of the same duration -----
ramp = Dynamics(model, times=times, pulse=np.linspace(z0, zf, times.size))
print(f"Linear ramp fidelity:    {ramp.state_fidelity():.6f}")  # 0.791338
```

If the Hamiltonian is affine in the control, `H(z) = H_d + z H_c`, pass the two matrices instead of a function,
`ControlModel(H_d=..., H_c=...)`, which is considerably faster. More examples are available in the
[`examples/`](https://github.com/geodesiq-control/geodesiq-control/tree/main/examples) notebooks.

## Public API

Top-level imports are intentionally kept small and explicit:

- `ControlModel`: Hamiltonian, eigensystem, geodesic metric and optimal (dimensionless) control pulse.
- `PulseControl`: the pulse at a physical duration; resampling, spectrum, filtering, plotting and export.
- `Dynamics`: closed- or open-system time evolution driven by the optimal pulse or any custom pulse.
- `decompose_hamiltonian`: SVD decomposition of a sampled time-dependent Hamiltonian into a QuTiP `QobjEvo`.
- `about`: version and environment information.
- `GeodesiQError` and the typed exception hierarchy
- `GeodesiQWarning` and related warnings
- `__version__`

The user guide and the full API reference are available at [geodesiq.readthedocs.io](https://geodesiq.readthedocs.io).

# Citing `geodesiq`

If you use `geodesiq` in your research, please cite it. The citation metadata is in
[`CITATION.cff`](https://github.com/geodesiq-control/geodesiq-control/blob/main/CITATION.cff), and GitHub's
"Cite this repository" button exports it as BibTeX or APA.

## Development

Use the following checks before submitting changes:

```bash
uv run ruff check src tests
uv run ruff format --check src
uv run ty check src
uv run pytest
uv run mkdocs build --strict
uv run python -m build
```