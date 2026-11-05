# Contributing

Thanks for contributing to `geodesiq`.

## Development setup

1. Create a branch from `dev`.
2. Commit your changes there.
3. Open a pull request into `dev`.
4. Once `dev` is stable, maintainers open a pull request from `dev` into `main`.

Direct pushes to `main` are not allowed.
Only `dev` is merged into `main` (no feature branch PRs directly into `main`).

Automation flow:
- CI runs on every push to `dev` or `main` and on pull requests into `main`; a pull request into `main` fails unless
  it comes from `dev`. Pull requests into `dev` do not run CI; their checks run once they are merged and pushed.
- Release runs after every successful CI run triggered by a push to `main`. Since only `dev` is merged into `main`,
  this is in practice the merge of a pull request from `dev`.
- Publish runs only after a successful `Release` workflow from `main`.

To recreate the development environment, run:
```bash
uv sync --group dev
```

If you do not use `uv`, create a virtual environment and install equivalent dev dependencies from `pyproject.toml`.

The benchmark history (`benchmarks/results/benchmark_history.parquet`) is stored in [Git LFS](https://git-lfs.com).
Run `git lfs install` once, before cloning, to get the file itself; the package and its tests do not need it.

## Run quality checks

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src
uv run ty check src tests
uv run mkdocs build --strict
uv run python -m build
uv run python -m twine check --strict dist/*
uv run jupyter execute examples/landau_zener.ipynb
```

`pytest` fails below 90% coverage. The package sources are formatted with `ruff format`; tests are linted but keep their
own formatting. CI builds the documentation on every push and executes `examples/landau_zener.ipynb` on pull requests
into `main`.

## Documentation

The documentation is built with MkDocs (`mkdocs.yml`, pages in `docs/`, API reference from the docstrings) and hosted
on [Read the Docs](https://geodesiq.readthedocs.io), configured by `.readthedocs.yaml`. Read the Docs rebuilds it on
every push to the tracked branches and tags. To preview it locally with live reload:

```bash
uv run mkdocs serve
```

Docstrings use the numpy style. `uv run mkdocs build --strict` fails on broken references, as the Read the Docs build
does.

## Pull request guidelines

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update `README.md` and `CHANGELOG.md` when user-facing behavior changes.
- Ensure all checks pass before requesting review.

## Release notes

- Add entries to the `[Unreleased]` section in `CHANGELOG.md`.
- At release time, move those entries under a versioned heading.

## Bumping the version

Use the `dev/bump_version.py` script to update the version consistently across all project files
(`src/geodesiq/_meta.py`, `CHANGELOG.md`, and the `version` and `date-released` fields of `CITATION.cff`, which
gets the same date as the new changelog heading).

**Bump by increment type:**
```bash
python dev/bump_version.py --patch   # e.g. 0.1.0 → 0.1.1
python dev/bump_version.py --minor   # e.g. 0.1.1 → 0.2.0
python dev/bump_version.py --major   # e.g. 0.2.0 → 1.0.0
```

**Set an explicit version:**
```bash
python dev/bump_version.py 1.2.3
```

The script accepts PEP 440 versions, rejects non-increasing versions, validates the changelog, supports dry runs
(`--dry-run`), and updates `_meta.py`, `CHANGELOG.md` and `CITATION.cff` with rollback protection.

After running the version-bump script:

1. Review the changes to `src/geodesiq/_meta.py`, `CHANGELOG.md` and `CITATION.cff`.
2. Commit the changes to `dev` and push them.
3. Open a pull request from `dev` into `main`.
4. Once the pull request is merged and CI succeeds on `main`, the `Release` workflow automatically:
   * reads the package version;
   * verifies that the corresponding `v<version>` tag does not already exist;
   * creates the `v<version>` tag;
   * creates the GitHub release.

The `Publish` workflow then runs after a successful release and publishes the package to PyPI.

Do not manually create or push the release tag.

