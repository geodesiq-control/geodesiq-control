# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

### Added

- `Dynamics(..., options=...)` configures QuTiP solver settings for all propagation methods, including tolerances,
  internal step limits and integration method. Numerical defaults are unchanged; integration failures now raise
  `SolverError` with the original exception chained, and rejected solver configurations raise `ValidationError`.
- `PulseControl.plot_pulse()` accepts optional `fig` and `ax` arguments to draw on an existing figure, subfigure or
  axis, like `ControlModel.plot_eigenvalues()` and `ControlModel.plot_metric_tensor()`. A provided figure is never
  closed; pass `show=False` to keep drawing on it.
- `Dynamics.populations()` returns the population of every state at each time sample, in the instantaneous eigenbasis
  of the Hamiltonian (`basis="adiabatic"`, default) or in the basis of the Hamiltonian matrix (`basis="diabatic"`). It
  accepts the same initial states as `state_fidelity()` and optional collapse operators.
- `Dynamics.plot_populations()` plots these populations over time, with the same `fig`, `ax`, `legend` and label
  arguments as `ControlModel.plot_eigenvalues()` and an optional `states` selection.
- `Dynamics(..., order=...)` selects how the pulse is interpolated between its samples: `0` holds each sample (as an
  arbitrary waveform generator does), `1` is linear and `3` is a cubic spline.
- `HamiltonianDecomposition`, the return type of `decompose_hamiltonian()`, is exported from `geodesiq` and has a
  docstring describing its attributes.

### Changed

- `PulseControl.filtered_pulse()` holds the first and last values before and after the pulse (`padding="hold"`, the
  new default), as an arbitrary waveform generator does, instead of SciPy's odd extension. Both ends are now treated
  alike (the odd extension moved the end of a steep pulse much more than its start), so filtered results change; pass
  `padding="odd"` for the former behaviour. `causal=True` applies the filter forward only (-3 dB at the cutoff, with
  a phase lag), as a transmission line would. The docstring now states that the default zero-phase filtering squares
  the magnitude response (-6 dB at `cutoff_freq`, effective order `2 * filter_order`), and a `GeodesiQWarning` reports
  when an endpoint moves by more than 1% of the pulse range, since `Dynamics` still measures fidelities against the
  eigenstates at the original endpoints.
- Rename the normalized-time grid from `ControlModel.s` to `ControlModel.tau` (`tau = t / t_f`) to match the
  manuscript notation. Update callers to use `model.tau`; numerical results and defensive-copy behavior are unchanged.
- The source distribution includes the test suite, so downstream packagers can run it (`pytest` from the unpacked
  sdist). The test of the `dev/` release script skips itself there.
- `Dynamics` no longer has the public, undocumented `evaluate_hamiltonian` attribute. Use the model's
  `ControlModel.evaluate_hamiltonian()` instead.
- The private `_flags_verbose` argument of `ControlModel` is keyword-only and no longer takes the third positional
  slot of the public signature.
- Type annotations follow what the functions accept and return. Array inputs that are converted with `np.asarray`
  (`ControlModel(H_d=..., H_c=...)` and the matrices returned by `H_func`/`partial_H_func`, `PulseControl(pulse=...)`,
  `Dynamics(times=..., pulse=...)` and `decompose_hamiltonian(times=...)`) are `numpy.typing.ArrayLike`, so lists
  pass strict type checkers. Real outputs are typed `NDArray[np.float64]` (`control_sol`, `tau`, `eigenenergies`,
  `control_pulse`, `Dynamics.populations()`, the arrays returned by `PulseControl`). `c_ops` is
  `Sequence[qt.Qobj | qt.QobjEvo | np.ndarray]`, so tuples and mixed lists type-check. `typing.List`, `Tuple` and
  `Optional` were replaced by the built-in generics and `X | None`.

- `Dynamics.average_gate_fidelity()` requires `target_gate`, as a keyword argument. Without it, QuTiP silently
  compared with the identity (e.g. 0.3398 for the Landau–Zener pulse, the same as `target_gate=qt.qeye(2)`), which the
  docstring did not say. Omitting it now raises `TypeError`, and `target_gate=None` raises `ValidationError`.
- `Dynamics` with a custom pulse (`times`/`pulse`) now interpolates it linearly between samples by default, instead of
  with a cubic spline, so that e.g. the piecewise-linear `discretized_pulse()` is simulated as the waveform it
  describes. With 4 samples the spline deviated from it by up to 2.1 (for a sweep of width 20) and changed the state
  fidelity from 0.969 to 0.935; on a coarse step it overshot the sampled range. The optimal pulse (`duration`) keeps the
  cubic spline, so its results are unchanged. The Hamiltonian evaluated between samples now uses the same
  interpolation as the propagation (it was always linear before).

- The diabatic transition list now depends on the control value. For `initial_state != final_state` the occupied
  instantaneous eigenstate is followed along the sweep: anticrossings with the neighbouring state towards
  `final_state` are passed diabatically (`dia_alpha`, `dia_beta`) until `final_state` is reached, and all other
  anticrossings adiabatically (`alpha`, `beta`). A pair of states that anticrosses twice is no longer crossed
  diabatically back and forth. Only transitions from the occupied state contribute to the metric tensor, as in the
  adiabatic case. Towards the gap maxima that bound a diabatic anticrossing, the exponents are blended smoothly
  (C1) into the adiabatic ones, so the metric does not jump where a diabatic passage ends. Overlapping anticrossings (e.g. a fan of levels crossing at one point) are passed together. The list
  is recomputed whenever the eigenproblem changes.
- The diabatic transition list stores only the transitions out of the occupied state, as an `(n_points, dim)`
  adiabatic fraction plus the occupied state at each point, instead of a dense `(n_points, dim, dim)` array, and the
  diabatic metric is evaluated without a Python loop over all pairs of states. For `dim = 200` and `num_steps = 1025`
  building the list drops from 658 MB peak memory and 136 ms to 3 MB and 3 ms. The metric is unchanged.
- The eigenproblem needs less memory. The Hamiltonian samples are centered in place and released right after the
  diagonalization, samples are written into one preallocated array, the numerical derivative runs before the
  eigenvectors exist and forms its sums in pieces, and `|<m|dH/dx|n>|` is computed in chunks of grid points. The peak
  drops from about 5, 6.5 and 7.5 to about 2, 2.2 and 3.5 complex `(num_steps, dim, dim)` arrays for affine models,
  callable models with `partial_H_func`, and callable models with the numerical derivative (for `dim = 256` and
  `num_steps = 1025`, one such array is 1.07 GB). The results are unchanged.
- `solve_problem()` and `plot_metric_tensor()` no longer evaluate the Hamiltonian once per call just to get its
  dimension once it is known, and a `num_steps` that is incompatible with the default `romb` integrator (not
  `2**k + 1`) is rejected before the eigenproblem is solved instead of after.

### Fixed

- A custom ODE solver that returns other time points than the requested uniform `t_eval` is resampled onto it (PCHIP),
  so `tau` is always uniform. Before, its own grid was stored as `tau`, while `Dynamics(duration=...)`,
  `synthesize_pulse()` and `PulseControl` rebuilt a uniform time grid from the sample count, which distorted the pulse.
- `Dynamics` captures the model's Hamiltonian at construction (copies of `H_d`/`H_c`, or of the fixed parameters and
  `control_name` of a callable model), as it already did for the pulse and the state indices. Before, the
  propagation was frozen but the endpoint eigenstates and the adiabatic basis of `populations()` used the live model,
  so changing a parameter afterwards mixed two Hamiltonians without an error. Create a new `Dynamics` to use a
  changed model.
- Document that, with the default `solve_ivp`, `solver_kwargs` act on the inverse equation `dtau/dx`: its independent
  variable is the control, so `max_step`, `first_step` and `min_step` are in control units and `atol`/`rtol` apply to
  `tau(x)`. Behaviour is unchanged.
- The occupied state is followed through true (symmetry-protected) level crossings by eigenvector overlap. Before, the
  state was tracked by its energy index, so at a crossing with a level that the control does not couple to it, the
  "initial state" became that level, and the metric vanished (or `solve_problem()` failed) where it should not. Now
  the state keeps its identity and changes its energy index at the crossing, also when the crossing falls on a grid
  point, where the degenerate eigenvectors are chosen to diagonalize `dH/dx`. A swap of eigenvectors between two grid
  points between states that `dH/dx` couples is an anticrossing the grid does not resolve, and keeps the energy order
  as before. Results without true crossings are unchanged. `initial_state` and `final_state` still refer to the energy
  order at `pulse_initial` and `pulse_final`; as long as `final_state` has not been set, it is the index in which the
  followed state ends after `solve_problem()` (used as the default target of `Dynamics`). The transfer is diabatic only
  if an explicit `final_state` differs from that index, so `dia_alpha` and `dia_beta` are now checked during
  `solve_problem()`, once the spectrum is known, instead of before it. A degenerate occupied state over more than one
  grid point still raises `MetricComputationError`.
- Clarify that the sampled-metric normalization cancels algebraically from both normalized solver equations;
  custom solvers use the PCHIP interpolant's own integral to avoid a quadrature-induced endpoint mismatch.
- Align plotting docstrings with the supported `Figure`/`SubFigure` inputs, tuple returns and validation errors;
  correct the documented eigenvalue-plot default title and document Dynamics configuration errors.
- Complete API docstrings for Hamiltonian evaluation, fixed parameters, decomposition helpers and state fidelity;
  document the `Dynamics` class and `about` module, and remove the unused installation-path reporting helper.
- `ControlModel.plot_eigenvalues()` and `ControlModel.plot_metric_tensor()` raise `ValidationError` when `ax` is not
  a matplotlib `Axes` or `fig` is not a `Figure`/`SubFigure` (e.g. a figure passed positionally as `ax`), instead of
  silently returning `None`. Their return type is now `tuple[Figure | SubFigure, Axes]` without `| None`, so
  `fig, ax = model.plot_eigenvalues()` needs no narrowing.
- `ControlModel.solve_problem()` with the default solver rejects `solver_kwargs` that the chosen `solve_ivp` method
  does not accept (e.g. the typo `{"rtl": 1e-8}`, or `min_step` with `RK45`) with a `ValidationError`. Before, SciPy
  only warned and the solve silently kept the default tolerances.
- `ControlModel.solve_problem()` no longer fails with "Required step size is less than spacing between numbers" when
  the metric tensor spans many orders of magnitude (e.g. large `alpha` and `beta`). The default solver now integrates
  the inverse equation `dtau/dx = sqrt(G)/a` and inverts it, exactly through the PCHIP antiderivative when no
  `solver_kwargs` are given, or with `solve_ivp` (default `rtol=1e-10`, `atol=1e-12`) otherwise. Custom solvers still
  integrate the forward equation (see below).
- The numerical derivative of a callable Hamiltonian no longer fails with "Numerical differentiation failed to
  converge" when the control values carry a large offset (e.g. a sweep from `1e6 - 0.3` to `1e6 + 0.3`). The rounding
  of the control grid made the uniformity check fail, so every point was sent to `scipy.differentiate.jacobian`, which
  could not reach its tolerance. The step is now taken from the grid endpoints, and the finite-difference and
  `jacobian` tolerances are floored at the relative rounding error of the control steps. If `jacobian` still does not
  converge at a point where a finite-difference value exists, that value is used with a `NumericalStabilityWarning`
  instead of raising `SolverError`.
- A custom `solver` in `ControlModel.solve_problem()` (any callable other than `scipy.integrate.solve_ivp` itself,
  including wrappers and `functools.partial(solve_ivp, ...)`) no longer gives a pulse that misses `pulse_final`, e.g.
  ending at x = 414 instead of 5 on a coarse grid. The forward equation `dx/dtau` is now normalized with the exact
  integral of the PCHIP interpolant it uses (the same interpolant as the default solver), the speed vanishes beyond
  `pulse_final`, and both endpoints are set exactly. The result must follow the exact normalized time `tau(x)` to within
  1e-2 at every sample, otherwise `SolverError` is raised.
- `ControlModel.set_parameters()` now stores deep copies of the values. Before, an array changed in place and passed
  again compared equal to the stored (same) object, so `solve_problem()` returned the old eigensystem and pulse; and
  changing it without calling `set_parameters()` silently changed what later Hamiltonian evaluations saw.
- `ControlModel.eigenenergies` is real (`float64`) for complex Hermitian Hamiltonians. The trace offset added back to
  the centered energies was kept complex, so the property returned `complex128` and caused `ComplexWarning`s
  downstream.
- The Hermiticity check no longer lets a large identity offset hide a non-Hermitian part. It was relative to
  `||H||_F`, so e.g. `1e8 * I + [[0, 1], [0.5, 0]]` was accepted and `eigh` silently used only one triangle. It is now
  relative to the norm of the traceless part, plus an allowance of `100 * eps * ||H||_F` for rounding errors, so
  Hermitian matrices with a large offset are still accepted. This also applies to `hermitian_tol` in
  `decompose_hamiltonian`.
- A `final_state` that was never set explicitly now follows `initial_state`. Before, it kept the value of the first
  `initial_state`, so a later `set_control(initial_state=1)` silently turned an adiabatic problem into a diabatic
  transfer `1 -> 0`, which then failed asking for `dia_alpha` and `dia_beta`. An explicitly set `final_state` keeps its
  value.
- The numerical derivative of a callable Hamiltonian with a large identity offset (e.g. `1e8 * I + ...`) no longer
  sends most grid points to `scipy.differentiate.jacobian`, which cost 11-28 times more Hamiltonian evaluations and
  did not improve the result. Only the traceless part of H is differentiated (the trace part does not change the
  metric), and the tolerances are floored at the rounding error the offset leaves in the samples. For
  `H = 1e10 * I + x * sz + 0.1 * sx` this also reduces the metric error from 28% to 0.1%.
- A metric tensor that is exactly zero at isolated grid points (e.g. a coupling that vanishes at `x = 0` on a grid
  containing 0) is now solved by the default solver, which integrates the finite `dtau/dx = sqrt(G)/a`; the pulse passes
  such a point infinitely fast. Before, the warning about the zero was followed by "ODE speed factor is non-finite or
  non-positive". A custom `solver`, which integrates the forward equation `dx/dtau`, raises `MetricComputationError`
  naming the zero points instead.
- `solve_problem()` accepts array values in `solver_kwargs` and `metric_integrator_kwargs` (e.g. `x` for
  `scipy.integrate.trapezoid`) on repeated calls. Comparing them with the stored kwargs raised "The truth value of an
  array ... is ambiguous". They are now compared element-wise and stored as deep copies, so an array changed in place
  and passed again also invalidates the cached results.
- With a callable `partial_H_func`, the first derivative evaluation validates the `H_func` result it uses for the
  dimension, so an `H_func` that returns e.g. a nested list no longer raises `AttributeError`. `str(model)` now also
  shows `final_state`, `dia_alpha` and `dia_beta`.
- The single-threaded BLAS mode for small Hamiltonians (dimension up to 256) also covers BLAS libraries loaded by an
  import after the first solve (e.g. `import torch`). The library list was read once and cached for the whole session.
  The API docs now state that this limit applies to the whole process while it is active.
- `PulseControl.filtered_pulse()` raises `ValidationError` for a pulse that is too short for the filter, naming the
  required length, instead of SciPy's raw `ValueError` ("The length of the input vector x must be greater than
  padlen"). The zero-phase filtering needs more than `3 * (2 * n + 1)` samples for `n` second-order sections (12 for
  `filter_order=3`, 15 for 4).
- `Dynamics` validates explicit states, gates and collapse operators. Unnormalized kets and density matrices without
  unit trace raise `ValidationError` (before, `state_fidelity(2 * a, 2 * b)` returned 15.85 and `populations(2 * a)`
  summed to 4), and states, `gate`, `target_gate` and `c_ops` of the wrong dimension raise `ValidationError` instead of
  a raw `TypeError` from QuTiP. Explicit states may be kets, bras or density matrices, as arrays or `Qobj` (also mixed,
  and with any tensor structure), and `c_ops` may be a tuple.

## [0.3.0] - 2026-10-01

### Added

- `Dynamics` can be driven by a custom pulse via the keyword-only `times` and `pulse` arguments, e.g. a pulse filtered
  with `PulseControl.filtered_pulse()`. Passing only `duration` keeps using the solved optimal pulse.
- `Dynamics.times`, `Dynamics.pulse` and `Dynamics.duration` read-only properties.
- `final_only` option in `Dynamics.time_evolution_operator()` and `Dynamics.average_gate_fidelity()` to keep only the
  final propagator instead of one dense matrix per time sample.
- A read-only property with the normalized time grid of the control solution (now `ControlModel.tau`).
- `CITATION.cff` with the citation metadata of the package.

### Changed

- `Dynamics` now takes `model` as its first (mandatory) argument: `Dynamics(model, duration=...)`. Calls that pass
  `duration` and `model` positionally (`Dynamics(tf, model)`) must be updated; keyword calls are unaffected. `hbar` is
  now keyword-only.
- All numerical tolerances are now relative, so results no longer depend on the units of the Hamiltonian (e.g. Joules
  or Hz instead of natural units):
  - an energy gap is degenerate when it is below `1e-12` times the spectral bandwidth (was an absolute `1e-14`);
  - the metric tensor is zero when it is below `1e-20` times its maximum;
  - Hermiticity is checked as `||H - H^dagger|| <= 1e-8 ||H||` in `ControlModel` and `decompose_hamiltonian`, whose
    `hermitian_tol` is now relative (default `1e-8`);
  - the absolute tolerance of the numerical derivative scales with `|H| / |control range|`.
- `ValidationError` now also derives from `ValueError`. `decompose_hamiltonian` raises `ValidationError` instead of a
  plain `ValueError`, so existing `except ValueError` code keeps working.
- `solve_problem()` without `pulse_accuracy` reuses the accuracy of the previous solve (1000 on the first solve). This
  also applies to the automatic re-solves of `control_sol` and `synthesize_pulse()` after a configuration change, which
  previously fell back to 1000 samples.
- The `ControlModel` control setters (`model.alpha = ...`, etc.) delegate to `set_control()`, so they share its
  validation and cache invalidation. As with `set_control()`, `num_steps` now defaults to `2**10 + 1` when unset.
- `PulseControl.export_pulse()` returns the written path and no longer prints. Unsupported extensions raise
  `ValidationError` (was `MissingArgsError`) before anything else is checked. `txt` files are now whitespace-delimited
  (`csv` stays comma-delimited).
- `matplotlib.pyplot` is imported lazily by `ControlModel` plotting methods.
- The numerical derivative of the Hamiltonian is computed with 8th-order finite differences of the samples that the
  eigenproblem already evaluated, instead of `scipy.differentiate.jacobian` at every grid point. This removes the ~11
  extra Hamiltonian evaluations per grid point (about 9x faster for a 16-level model). The adaptive `jacobian` is still
  used where the finite-difference error estimate is too large, e.g. near a kink.
- Batched eigendecompositions and the `Dynamics` solvers run BLAS/LAPACK single-threaded for Hilbert spaces up to
  dimension 256 (via the new `threadpoolctl` dependency). Multi-threading many small independent problems is slower,
  in some environments by more than 10x: the 64-level eigenproblem went from 10.7 s to 0.35 s on a 24-core machine.

### Dev

- Documentation site built with MkDocs and mkdocstrings and hosted on Read the Docs (`.readthedocs.yaml`):
  <https://geodesiq.readthedocs.io>. Docstrings use the numpy style.
- `tests/conftest.py` selects the non-interactive matplotlib backend for all tests.
- Coverage gate raised from 70% to 90%; CI also lints `tests/` and checks the formatting of `src/`.
- New tests: Landau-Zener transition probability of `Dynamics`, finite-difference derivative and its fallback, BLAS
  thread limiting.
- `dynamics_dim_scaling` benchmark scenario for `Dynamics` (callable and affine models).
- `dev/bump_version.py` also updates the version in `CITATION.cff`.
- The lock file of the benchmark runner (`benchmark_history.parquet.lock`) is no longer tracked.

### Fixed

- `PulseControl.export_pulse()` wrote `txt`/`csv` values with 8 fixed decimals, so e.g. nanosecond times in seconds were
  saved as zero. Values are now written with full double precision.
- Unit-dependent degeneracy and Hermiticity checks: SI-scale Hamiltonians were rejected as degenerate, small-scale
  non-Hermitian matrices were accepted, and GHz-scale numerical derivatives failed to converge.
- Non-finite Hamiltonians are reported as such instead of as non-Hermitian.
- `Dynamics.state_fidelity()` accepts NumPy integer state indices and rejects booleans.
- `PulseControl.discretized_pulse()` and `PulseControl.filtered_pulse()` validate `linear_steps`, `cutoff_freq` and
  `filter_order`.
- `about()` no longer prints an empty "Reference paper" line.

## [0.2.0] - 2026-09-22

### Added

- Added the possibility to use affine Hamiltonian control, `H(u) = H_d + u H_c`, by passing the constant drift and
  control matrices to `ControlModel(H_d=..., H_c=...)`. This is far more efficient than a general Hamiltonian
  function.

### Dev

- Use `twine>=7` for publishing packages.

### Changed

- Switched static type checking tooling from `mypy` to `ty` in project dependencies, CI, and contributor documentation.

## [0.1.5] - 2026-08-06

### Fixed

- `README.md` badges for PyPI, and logo image.
- Correctly raise `NumericalStabilityWarning` as a warning, instead of raising it as an exception.

## [0.1.4] - 2026-08-06

### Added

- `ControlModel` now raises a `MetricComputationError` when the metric tensor is never larger than the _SINGULARITY_RTOL
  threshold, indicating that the metric tensor is always close to zero.
- Test to verify that the DQD model does not raise a `MetricComputationError` when the metric tensor is close to zero at
  some point, but not always.

### Changed

- Relax the _SINGULARITY_RTOL from 1e-12 to 1e-20, so values closer to zero are accepted.
- Instead of returning a `MetricComputationError` when the metric tensor is close to zero at some point, now the
  `ControlModel` will raise a `NumericalStabilityWarning` with a message indicating that the metric tensor is close to
  zero at some point.

## [0.1.3] - 2026-07-28

### Added

- Badges for PyPI, Python version, GitHub Actions CI status, code coverage, license and maintenability.

## [0.1.2] - 2026-07-27

### Removed

- Duplicate code in verification of `filtered_pulse`.
- Possibility to directly call `PulseControl`. Instead, use `ControlModel` to create a control model and then call
  `ControlModel.synthesize_pulse()` to obtain the pulse control, and later use the provided methods.
- Access of private attributes of `ControlModel` in `Dynamics`.

### Added

- Validate `hamiltonian` and `partial_hamiltonian` for shape, values, ... . Also, a change of shape during the lifetime
  of the `ControlModel` is not allowed.
- Better comparison between previous and new `parameters` to check if the eigenproblem must be solved again.
- `gitattributes` file to fix line endings as LF for all files.
- `ControlModel.parameters` to obtain a read-only view of the parameters.
- Add checks for `initial_state` and `target_state` in `ControlModel` to ensure they are within the valid range of the
  Hilbert space defined by the Hamiltonian.
- Atomic implementation of `set_control`, to ensure that the control is only updated if all validations pass, preventing
  partial updates that could lead to inconsistent states.

### Fixed

- Validation of parameters in `filtered_pulse`.
- Modify Hatch to only include the source files in the package, not the tests and other files.
- Comparison between previous and new `parameters` to check if the eigenproblem must be solved again. Now, the
  comparison is done using `numpy.allclose()` instead of `numpy.array_equal()`, which allows for a more flexible
  comparison that accounts for numerical precision issues. Also, it compares non-numpy parameters.
- Solve Hamiltonians independently of a general shift in the energies. This is done by subtracting the mean of the
  eigenvalues from the Hamiltonian before solving the eigenproblem. This ensures that the eigenvalues are always
  centered around zero, which improves numerical stability and accuracy.
- Fixed the CI so the GitHub tag, release, and PyPI release are only created when everything is successful. This
  prevents the creation of a release when the build is broken.

## [0.1.1] - 2026-07-10

### Fixed

- `hbar` validation was too strict.
- Fix CI to publish the release

## [0.1.0] - 2026-06-19

### Added

- Initial release of the project. This version includes the basic functionality and features as outlined in the project
  proposal.
