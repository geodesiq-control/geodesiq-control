# Benchmark results

Benchmark history is generated locally by:

```bash
uv run --only-group bench python -m benchmarks.run_benchmarks
```

Each run appends its results, tagged with the package version and machine information, to `benchmark_history.parquet`.
The file is tracked in the repository so that results can be compared across versions and machines; commit it when a
run should become part of the published history. The runner rewrites the file atomically, guarded by
`benchmark_history.parquet.lock`.
