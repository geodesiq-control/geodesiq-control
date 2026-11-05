#!/usr/bin/env bash
# Build the JupyterLite site with the example notebooks into jupyterlite/_output and optionally serve it.
#
#   bash jupyterlite/build.sh          # build
#   bash jupyterlite/build.sh serve    # build, then serve on http://localhost:8000
#
# Runs on Linux/macOS/WSL with micromamba on the PATH. Native Windows fails: the WebAssembly packages contain symlinks.
set -euo pipefail
cd "$(dirname "$0")"

# The build tools get their own environment outside the repository, next to the project one. It cannot be an ephemeral
# `uv run --with` environment: empack finds its share/empack/empack_config.yaml only next to a real sys.prefix.
tools_env="${GEODESIQ_LITE_ENV:-$HOME/.venvs/geodesiq-lite}"
[[ -x "$tools_env/bin/python" ]] || uv venv --quiet "$tools_env"
uv pip install --quiet --python "$tools_env/bin/python" \
    "jupyterlite-core==0.8.6" "jupyterlite-xeus==5.1.0" jupyter-server pip

rm -rf _output .jupyterlite.doit.db
"$tools_env/bin/jupyter" lite build \
    --XeusAddon.environment_file=environment.yml \
    --contents ../examples/landau_zener.ipynb \
    --contents ../examples/dqd_model.ipynb \
    --contents ../examples/many_body.ipynb \
    --contents ../examples/four_level.ipynb \
    --contents smoke_test.ipynb \
    --output-dir _output

if [[ "${1:-}" == "serve" ]]; then
    # A plain static server is enough: JupyterLite needs no backend.
    "$tools_env/bin/python" -m http.server 8000 --directory _output
fi
