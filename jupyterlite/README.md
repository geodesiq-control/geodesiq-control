# JupyterLite site

Static site that runs the notebooks of `examples/` in the browser (JupyterLite with the xeus-python kernel and
WebAssembly builds from emscripten-forge), with nothing installed on the user's side. CI builds it on every push, runs
the notebooks in headless Chromium, and deploys it to GitHub Pages from `main`:
<https://geodesiq-control.github.io/geodesiq-control/>.

Links to a single notebook:

- JupyterLab: `https://geodesiq-control.github.io/geodesiq-control/lab/index.html?path=landau_zener.ipynb`
- Notebook interface (simpler): `https://geodesiq-control.github.io/geodesiq-control/notebooks/index.html?path=landau_zener.ipynb`

## Contents

- `environment.yml`: the kernel environment. NumPy, SciPy, QuTiP and Matplotlib come from `emscripten-forge-4x`,
  pure-Python packages from conda-forge, and `geodesiq` from this checkout through pip (pure Python, no dependency
  resolution), so the site always matches the commit it was built from.
- `build.sh`: builds the site into `_output/` (gitignored) from the four notebooks in `examples/` and the smoke test.
- `smoke_test.ipynb`: checks every part that could break in WebAssembly one by one and raises if any check fails.
- `run_notebooks.py`: serves the built site, runs notebooks in a headless browser through the JupyterLab menus and
  fails if a cell does not run or prints a traceback. CI uses it; locally, e.g.
  `uv run --no-project --with playwright python jupyterlite/run_notebooks.py --browser msedge smoke_test.ipynb`.

The examples run unchanged. `many_body.ipynb` builds the Hamiltonian with NumPy when QuSpin is missing and runs the
sweep sequentially when parameterrun is missing, as is the case here.

## Build locally

On Linux, macOS or WSL, not native Windows: the WebAssembly packages contain symlinks, which micromamba cannot create
there without Developer Mode. Needs `uv` and `micromamba` on the PATH (jupyterlite-xeus calls micromamba to solve the
`emscripten-wasm32` environment). The build tools go into `~/.venvs/geodesiq-lite` (override with
`GEODESIQ_LITE_ENV`), never into the repository. From the repository root:

```bash
bash jupyterlite/build.sh serve     # build (a few minutes), then serve on http://localhost:8000
```

From Windows through WSL, with the output written to the Windows checkout:

```bash
wsl -d Ubuntu -e bash -lc 'cd /mnt/c/Users/<user>/OneDrive-UniA/Projects/geodesiq-control && bash jupyterlite/build.sh serve'
```

## Behaviour in the browser (2026-10-29)

Kernel: Python 3.13.1 on Emscripten 4.0.9 (wasm32), NumPy 2.5.4, SciPy 1.18.0, QuTiP 5.3.1, Matplotlib 3.11.2. The
kernel environment is 74 MB, downloaded on the first visit and cached by the browser afterwards.

| Notebook | Run all (headless Edge, local server) |
| --- | --- |
| `smoke_test.ipynb` | 8 s, 20/20 checks |
| `landau_zener.ipynb` | 14 s |
| `four_level.ipynb` | 8 s; 1 − F = 5.39e-3 and 0.72, as in the paper |
| `dqd_model.ipynb` | 152 s; joblib falls back to sequential execution |
| `many_body.ipynb` | 357 s, of which L = 10 takes 5.4 min |

Speed is about half of native for small matrices and a third for L = 10 (dimension 528). `threadpoolctl` warns once
(`Unable to import LDSO from pyodide_js._module`) on the first `limit_blas_threads` call; this is harmless.

The kernel is single-threaded: emscripten-forge builds Python without pthreads (`can't start new thread`),
`os.cpu_count()` is 1, OpenBLAS is serial and joblib runs sequentially. WebAssembly threads would need the whole stack
(Python, NumPy, SciPy, OpenBLAS) rebuilt with `-pthread` and a cross-origin-isolated page (COOP/COEP headers, which
GitHub Pages does not set), so parallel sweeps remain a reason to run locally.
