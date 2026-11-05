# Benchmark results

Benchmark history is generated locally by:

```bash
uv run --only-group bench python -m benchmarks.run_benchmarks
```

Each run appends its results, tagged with the package version and machine information, to `benchmark_history.parquet`.
The file is tracked in the repository so that results can be compared across versions and machines; commit it when a
run should become part of the published history. The runner rewrites the file atomically, guarded by
`benchmark_history.parquet.lock`.

The history is stored in [Git LFS](https://git-lfs.com) (see `.gitattributes`). Run `git lfs install` once before
cloning, or `git lfs pull` in an existing clone; without it, `benchmark_history.parquet` is a small text pointer that
pandas cannot read.
