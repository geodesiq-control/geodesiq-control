# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

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