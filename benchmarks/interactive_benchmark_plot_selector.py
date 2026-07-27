"""Interactive cascading selector for plotting full benchmark scenarios.

This module combines a metadata selection flow with six-panel plotting helpers
so all benchmark scenarios can be visualized and compared interactively inside
a Jupyter notebook.

Typical notebook use
--------------------

    from benchmarks.interactive_benchmark_plot_selector import load_history, launch_selector

or, when the notebook runs from inside the ``benchmarks`` directory,

    from interactive_benchmark_plot_selector import load_history, launch_selector

    df = load_history("benchmarks/results/benchmark_history.parquet")
    ui = launch_selector(df)

The widget builds two cascading metadata selectors.  Each selector identifies a
benchmark data set via metadata columns only.  A checkbox next to each selector
controls whether that dataset appears in the six-panel figure, so you can plot a
single dataset or compare A vs B.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Matplotlib style overrides applied via rc_context around every figure.
_RC: dict[str, Any] = {"font.family": "serif", "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                       "axes.titleweight": "bold", "legend.fontsize": 7.5, "legend.framealpha": 0.85, "figure.dpi": 150,
                       "lines.linewidth": 1.8, "lines.markersize": 5, }

# Ordered benchmark stages and their display properties.
STAGES = ["eigenproblem", "metric", "ode", "total"]

STAGE_LABEL: dict[str, str] = {"eigenproblem": "Eigenproblem", "metric": "Metric tensor", "ode": "ODE",
                               "total": "Total", }

STAGE_COLOR: dict[str, str] = {"eigenproblem": "C0", "metric": "C1", "ode": "C2", "total": "C3", }

STAGE_MARKER: dict[str, str] = {"eigenproblem": "o", "metric": "s", "ode": "^", "total": "D", }

# Visual styles used when color distinguishes the selected comparison value (A/B).
COMPARE_COLOR_STYLE: dict[str, dict[str, Any]] = {
    "A": {"color": "C0", "marker": "o", "linestyle": "-", "linewidth": 2.4, "alpha": 1.0, "zorder": 10},
    "B": {"color": "C1", "marker": "s", "linestyle": "-", "linewidth": 2.2, "alpha": 0.95, "zorder": 9}, }

# Visual styles used when color is already reserved for stage (panel C / E / F).
COMPARE_LINE_STYLE: dict[str, dict[str, Any]] = {"A": {"linestyle": "-", "linewidth": 2.2, "alpha": 1.0, "zorder": 10},
                                                 "B": {"linestyle": "--", "linewidth": 2.0, "alpha": 0.95,
                                                       "zorder": 9}, }

# Default metadata columns used to identify a benchmark run in the selectors.
DEFAULT_SELECTOR_COLUMNS = ["geodesiq_version", "python_version", "platform", "processor", "cpu_count"]

# Placeholder shown for missing / NaN metadata values in dropdowns.
NA_LABEL = "<NA>"

__all__ = ["DEFAULT_SELECTOR_COLUMNS", "CascadingSelector", "SelectedDataSet", "_build_dual_benchmark_figure",
           "launch_selector", "load_history", ]


# ---------------------------------------------------------------------------
# Public data helpers
# ---------------------------------------------------------------------------


def load_history(path: str | Path) -> pd.DataFrame:
    """Load a benchmark-history Parquet file and return a normalized DataFrame."""
    return prepare_history(pd.read_parquet(path))


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich a raw benchmark DataFrame with display-friendly columns.

    Adds ``run_label`` (human-readable timestamp) and ``version_label``.
    Raises ``ValueError`` when mandatory columns are absent.
    """
    required = {"scenario", "variant", "stage", "sweep_val_num", "mean_s"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Benchmark history is missing required columns: {missing}")

    out = df.copy()

    # Build a human-readable run label from the timestamp when available.
    if "run_timestamp" in out.columns:
        run_datetime = pd.to_datetime(out["run_timestamp"], errors="coerce", utc=True)
        out["run_datetime"] = run_datetime
        out["run_label"] = run_datetime.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        out["run_label"] = out["run_label"].fillna(out["run_timestamp"].astype(str))
    else:
        out["run_datetime"] = pd.NaT
        out["run_label"] = "unknown run"

    # Prefix the version number with "v" for display purposes.
    if "geodesiq_version" in out.columns:
        out["version_label"] = "v" + out["geodesiq_version"].astype(str)
    else:
        out["version_label"] = "unknown version"

    # Store string copies of key metadata columns so widget comparisons are
    # always done on strings, leaving the original dtypes untouched.
    for col in ["geodesiq_version", "python_version", "platform", "cpu_count", "run_label"]:
        if col in out.columns:
            out[f"__{col}_str"] = out[col].astype(str)

    return out


# ---------------------------------------------------------------------------
# Selector utilities
# ---------------------------------------------------------------------------


def _default_selector_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of DEFAULT_SELECTOR_COLUMNS that exist in *df*, plus a run column."""
    columns = [col for col in DEFAULT_SELECTOR_COLUMNS if col in df.columns]
    if "run_label" in df.columns:
        columns.append("run_label")
    elif "run_timestamp" in df.columns:
        columns.append("run_timestamp")
    return columns


def _selector_view(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a string-cast copy of *df* restricted to *columns* for widget filtering."""
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].astype("string").fillna(NA_LABEL)
    return out


def _sort_key(value: object) -> tuple[int, float | str]:
    """Sort key that orders numeric strings numerically, then falls back to case-insensitive text."""
    text = str(value)
    try:
        return 0, float(text)
    except ValueError:
        return 1, text.lower()


def _unique_sorted(series: pd.Series) -> list[str]:
    """Return deduplicated, sorted values from *series* (excludes NaN)."""
    values = [v for v in series.dropna().unique().tolist()]
    return sorted(values, key=_sort_key)


def _shorten(text: object, max_len: int = 65) -> str:
    """Truncate *text* to *max_len* characters, appending an ellipsis if needed."""
    value = str(text)
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _make_label(signature: dict[str, str], columns: Iterable[str]) -> str:
    """Build a compact ``key=value`` label string from the given *signature*."""
    pieces = [f"{col}={_shorten(signature[col], 32)}" for col in columns if col in signature]
    return " | ".join(pieces)


# ---------------------------------------------------------------------------
# SelectedDataSet dataclass
# ---------------------------------------------------------------------------


@dataclass
class SelectedDataSet:
    """Container for a single selector's resolved dataset and its metadata label."""

    signature: dict[str, str]  # metadata key→value identifying this dataset
    data: pd.DataFrame  # rows from the benchmark history matching the signature
    label: str  # human-readable summary string


# ---------------------------------------------------------------------------
# CascadingSelector
# ---------------------------------------------------------------------------


class CascadingSelector:
    """Cascading dropdown selectors that progressively filter a benchmark DataFrame.

    Each dropdown narrows the available options for the dropdowns below it, so
    users can drill down to a unique metadata combination.
    """

    def __init__(self, df: pd.DataFrame, columns: list[str], title: str,
                 on_change: Callable[[], None] | None = None, ) -> None:
        # Store the full dataset and a string-cast view used for comparisons.
        self.df = df
        self.columns = columns
        self.view = _selector_view(df, columns)
        self.title = title
        self.on_change = on_change
        self.excluded_signature: dict[str, str] | None = None
        self._updating = False  # guard against recursive widget updates

        # Create one Dropdown per metadata column.
        self.widgets: dict[str, widgets.Dropdown] = {}
        for col in columns:
            # Fixed label width keeps all controls aligned and avoids oversized gaps.
            self.widgets[col] = widgets.Dropdown(description=col, options=[],
                                                 layout=widgets.Layout(width="560px", align_self="flex-start"),
                                                 style={"description_width": "130px"}, )
            self.widgets[col].observe(self._on_widget_change, names="value")

        self.box = widgets.VBox([widgets.HTML(f"<b>{title}</b>")] + [self.widgets[c] for c in columns],
                                layout=widgets.Layout(width="100%"), )
        self.refresh()

    def set_on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked whenever any dropdown value changes."""
        self.on_change = callback

    def set_excluded_signature(self, signature: dict[str, str] | None) -> None:
        """Exclude rows matching *signature* so this selector never duplicates another."""
        self.excluded_signature = signature
        self.refresh()

    def available_view(self) -> pd.DataFrame:
        """Return the string view with rows that exactly match *excluded_signature* removed."""
        view = self.view
        if self.excluded_signature:
            exact = np.ones(len(view), dtype=bool)
            for col, value in self.excluded_signature.items():
                if col in view.columns:
                    exact &= view[col].eq(str(value)).to_numpy()
            view = view.loc[~exact]
        return view

    def _on_widget_change(self, change: dict[str, object]) -> None:
        """Handle a dropdown change: refresh all downstream dropdowns and notify."""
        if self._updating:
            return

        # Identify which column triggered the change.
        changed_col = next((col for col, w in self.widgets.items() if w is change["owner"]), None)
        if changed_col is None:
            return

        self.refresh(start=self.columns.index(changed_col) + 1)
        if self.on_change is not None:
            self.on_change()

    def refresh(self, start: int = 0) -> None:
        """Recompute dropdown options from *start* onward based on current values."""
        self._updating = True
        try:
            view = self.available_view()
            if view.empty:
                for widget in self.widgets.values():
                    widget.options = []
                    widget.value = None
                return

            mask = pd.Series(True, index=view.index)

            for i, col in enumerate(self.columns):
                if i < start:
                    # Apply the already-committed filter for upstream columns.
                    value = self.widgets[col].value
                    if value is not None:
                        mask &= view[col].eq(str(value))
                    continue

                # Rebuild options and preserve the previous selection when possible.
                candidates = view.loc[mask, col]
                options = _unique_sorted(candidates)
                old_value = self.widgets[col].value

                self.widgets[col].options = options
                if old_value in options:
                    self.widgets[col].value = old_value
                elif options:
                    self.widgets[col].value = options[0]
                else:
                    self.widgets[col].value = None

                value = self.widgets[col].value
                if value is not None:
                    mask &= view[col].eq(str(value))
        finally:
            self._updating = False

    def signature(self) -> dict[str, str]:
        """Return the current metadata signature as a ``{column: value}`` dict."""
        return {col: str(self.widgets[col].value) for col in self.columns if self.widgets[col].value is not None}

    def selected(self) -> SelectedDataSet:
        """Resolve the current selection to a :class:`SelectedDataSet`."""
        signature = self.signature()
        mask = pd.Series(True, index=self.view.index)
        for col, value in signature.items():
            mask &= self.view[col].eq(str(value))
        data = self.df.loc[mask].copy()
        label = _make_label(signature, self.columns)
        return SelectedDataSet(signature=signature, data=data, label=label)


# ---------------------------------------------------------------------------
# Public entry-point: launch_selector
# ---------------------------------------------------------------------------


def launch_selector(df_or_path: pd.DataFrame | str | Path = "results/benchmark_history.parquet",
                    selector_columns: list[str] | None = None, *, show_std_band: bool = True,
                    figsize: tuple[float, float] = (12.0, 7.0), ):
    """Display the interactive benchmark selector and comparison plot.

    Shows two cascading metadata selectors plus a six-panel figure.  An
    *Enable* checkbox next to each selector controls whether that dataset
    appears in the figure, so the same UI supports both single-dataset
    inspection and A-vs-B comparison.

    Parameters
    ----------
    df_or_path
        Pre-loaded DataFrame or path to a ``benchmark_history.parquet`` file.
    selector_columns
        Metadata columns to expose in the dropdowns.  Defaults to the set
        returned by :func:`_default_selector_columns`.
    show_std_band
        Initial state of the ±std_s band toggle.
    figsize
        Matplotlib figure size passed to :func:`build_dual_benchmark_figure`.
    """
    # Load data from disk when a path is supplied.
    if isinstance(df_or_path, (str, Path)):
        df = load_history(df_or_path)
    else:
        df = prepare_history(df_or_path.copy())

    # Resolve which metadata columns to show in the selectors.
    if selector_columns is None:
        selector_columns = _default_selector_columns(df)
    else:
        selector_columns = [c for c in selector_columns if c in df.columns]

    if not selector_columns:
        raise ValueError("No valid selector columns were found in the DataFrame.")
    resolved_selector_columns: list[str] = list(selector_columns)

    # --- Control widgets -------------------------------------------------------
    std_checkbox = widgets.Checkbox(value=show_std_band, description="Show ± std_s bands")
    enable_a = widgets.Checkbox(value=True, description="Enable A", indent=False, layout=widgets.Layout(width="120px"))
    enable_b = widgets.Checkbox(value=False, description="Enable B", indent=False, layout=widgets.Layout(width="120px"))
    output = widgets.Output()

    # --- Cascading selectors ---------------------------------------------------
    selector_a = CascadingSelector(df, resolved_selector_columns, "Data set A")
    selector_b = CascadingSelector(df, resolved_selector_columns, "Data set B")

    # --- Update callback -------------------------------------------------------

    def update(_=None) -> None:
        """Re-render the figure whenever any control or selector changes."""
        a = selector_a.selected()
        # Prevent selector B from accidentally duplicating selector A's pick.
        selector_b.set_excluded_signature(a.signature if enable_a.value else None)
        b = selector_b.selected()

        # Gray out selector boxes that are not currently enabled.
        selector_a.box.layout.opacity = "1.0" if enable_a.value else "0.4"
        selector_b.box.layout.opacity = "1.0" if enable_b.value else "0.4"

        with output:
            clear_output(wait=True)

            # Decide which datasets to pass based on the enable checkboxes.
            df_a = a.data if enable_a.value else None
            df_b = b.data if enable_b.value else None

            if df_a is None and df_b is None:
                print("Enable at least one dataset to display a figure.")
                return

            if enable_a.value and df_a is not None and df_a.empty:
                print("Dataset A is empty for the selected metadata.")
                return
            if enable_b.value and df_b is not None and df_b.empty:
                print("Dataset B is empty for the selected metadata.")
                return

            fig = _build_dual_benchmark_figure(df_a, df_b, show_std_band=bool(std_checkbox.value), figsize=figsize)
            display(fig)
            plt.close(fig)

    # Wire all controls to the update callback.
    selector_a.set_on_change(update)
    selector_b.set_on_change(update)
    std_checkbox.observe(update, names="value")
    enable_a.observe(update, names="value")
    enable_b.observe(update, names="value")

    # --- Layout ----------------------------------------------------------------
    # Pair each enable checkbox with its selector box.
    panel_a = widgets.VBox([enable_a, selector_a.box], layout=widgets.Layout(width="auto", align_items="flex-start"))
    panel_b = widgets.VBox([enable_b, selector_b.box], layout=widgets.Layout(width="auto", align_items="flex-start"))

    controls = widgets.VBox([widgets.HTML("<h3>Benchmark scenario comparison</h3>"
                                          "<p>Select metadata for one or two benchmark datasets. "
                                          "Use the <b>Enable</b> checkboxes to toggle each dataset in the figure.</p>"),
                             std_checkbox, widgets.HBox([panel_a, panel_b],
                                                        layout=widgets.Layout(width="100%", gap="20px",
                                                                              justify_content="flex-start",
                                                                              align_items="flex-start"), ), output, ])

    display(controls)
    update()
    return controls


# ---------------------------------------------------------------------------
# Private entry-point: build_dual_benchmark_figure
# ---------------------------------------------------------------------------


def _build_dual_benchmark_figure(df_a: pd.DataFrame | None, df_b: pd.DataFrame | None, *, show_std_band: bool = True,
                                 figsize: tuple[float, float] = (12.0, 7.0), ) -> plt.Figure:
    """Build a six-panel benchmark figure for one or two datasets.

    Either *df_a* or *df_b* may be ``None`` to plot a single dataset.
    When both are supplied the two datasets are drawn with contrasting
    visual styles (color A vs B) so differences are easy to spot.

    Parameters
    ----------
    df_a, df_b
        Benchmark history DataFrames to compare.  Pass ``None`` to omit a
        dataset entirely.
    show_std_band
        Draw ± ``std_s`` uncertainty bands when available.
    figsize
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Determine which datasets are active and build display labels.
    active: list[tuple[pd.DataFrame, str, str]] = []
    if df_a is not None:
        active.append((prepare_history(df_a), "A", "A"))
    if df_b is not None:
        active.append((prepare_history(df_b), "B", "B"))

    if not active:
        raise ValueError("At least one of df_a or df_b must be a non-None DataFrame.")

    # Only show dataset legends when both A and B are active.
    show_dataset_legend = len(active) > 1

    # Choose the figure suptitle based on how many datasets are shown.
    if len(active) == 2:
        suptitle = "geodesiq CPU benchmark  |  A vs B"
    else:
        suptitle = f"geodesiq CPU benchmark  |  Dataset {active[0][1]}"

    with mpl.rc_context(_RC):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.subplots_adjust(hspace=0.42, wspace=0.32)

        # -- Panel A: Total time vs Hilbert-space dimension --------------------
        ax = axes[0, 0]
        any_data = False
        for df, key, label in active:
            sub = _select(df, scenario="dim_scaling", variant="analytical", stage="total")
            any_data |= _plot_curve(ax, sub, label=label, style=COMPARE_COLOR_STYLE[key], show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("Hilbert-space dimension  N")
            ax.set_ylabel("CPU time (ms)")
            ax.set_title(_panel_title("A", "Total time vs dimension"))
            _format_log_axes(ax)
            _set_xticks(ax, [2, 4, 8, 16, 32, 64])
            if show_dataset_legend:
                ax.legend(loc="upper left", fontsize=7.5)
        else:
            _no_data(ax)

        # -- Panel B: Total time vs num_steps ----------------------------------
        ax = axes[0, 1]
        any_data = False
        for df, key, label in active:
            sub = _select(df, scenario="num_steps_scaling", variant="analytical", stage="total")
            any_data |= _plot_curve(ax, sub, label=label, style=COMPARE_COLOR_STYLE[key], show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("Number of λ grid points  (num_steps)")
            ax.set_ylabel("CPU time (ms)")
            ax.set_title(_panel_title("B", "Total time vs num_steps"))
            _format_log_axes(ax)
            if show_dataset_legend:
                ax.legend(loc="upper left", fontsize=7.5)
        else:
            _no_data(ax)

        # -- Panel C: Stage breakdown vs dimension -----------------------------
        ax = axes[0, 2]
        any_data = False
        for df, key, label in active:
            sub_all = _select(df, scenario="dim_scaling", variant="analytical")
            for stage in STAGES:
                style = dict(COMPARE_LINE_STYLE[key])
                style.update({"color": STAGE_COLOR[stage], "marker": STAGE_MARKER[stage]})
                sub = sub_all[sub_all["stage"] == stage]
                any_data |= _plot_curve(ax, sub, label=f"{label} | {STAGE_LABEL[stage]}", style=style,
                                        show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("Hilbert-space dimension  N")
            ax.set_ylabel("CPU time (ms)")
            ax.set_title(_panel_title("C", "Stage breakdown vs dimension"))
            _format_log_axes(ax)
            _set_xticks(ax, [2, 4, 8, 16, 32, 64])
            # Separate legends: colours → stages, line style → dataset.
            stage_handles = [
                plt.Line2D([], [], color=STAGE_COLOR[s], marker=STAGE_MARKER[s], linewidth=1.7, label=STAGE_LABEL[s])
                for s in STAGES]
            leg1 = ax.legend(handles=stage_handles, loc="upper left", fontsize=6.5, title="Stages")
            ax.add_artist(leg1)
            if show_dataset_legend:
                dataset_handles = [
                    plt.Line2D([], [], color="k", linestyle=COMPARE_LINE_STYLE[key]["linestyle"], linewidth=2.0,
                               label=label) for _, key, label in active]
                ax.legend(handles=dataset_handles, loc="upper right", fontsize=6.5, title="Datasets")
        else:
            _no_data(ax)

        # -- Panel D: ODE time vs pulse_accuracy -------------------------------
        ax = axes[1, 0]
        any_data = False
        for df, key, label in active:
            sub = _select(df, scenario="pulse_accuracy_scaling", stage="ode")
            any_data |= _plot_curve(ax, sub, label=label, style=COMPARE_COLOR_STYLE[key], show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("ODE evaluation points  (pulse_accuracy)")
            ax.set_ylabel("CPU time (ms)")
            ax.set_title(_panel_title("D", "ODE stage vs pulse_accuracy"))
            _format_log_axes(ax)
            if show_dataset_legend:
                ax.legend(loc="upper left", fontsize=7.5)
        else:
            _no_data(ax)

        # -- Panel E: Analytical vs numerical derivative -----------------------
        ax = axes[1, 1]
        any_data = False
        for df, key, label in active:
            sub_all = _select(df, scenario="analytical_vs_numerical", stage="eigenproblem")
            for variant, linestyle in (("analytical", "-"), ("numerical", "--")):
                style = dict(COMPARE_COLOR_STYLE[key])
                style["linestyle"] = linestyle
                sub = sub_all[sub_all["variant"] == variant]
                any_data |= _plot_curve(ax, sub, label=f"{label} | {variant}", style=style,
                                        show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("Hilbert-space dimension  N")
            ax.set_ylabel("Eigenproblem stage (ms)")
            ax.set_title(_panel_title("E", "Analytical vs numerical  ∂H/∂λ"))
            _format_log_axes(ax)
            _set_xticks(ax, [2, 4, 8, 16, 32, 64])
            # Separate legends: colours → datasets, line style → variant.
            variant_handles = [plt.Line2D([], [], color="k", linestyle="-", linewidth=1.7, label="analytical"),
                               plt.Line2D([], [], color="k", linestyle="--", linewidth=1.7, label="numerical"), ]
            if show_dataset_legend:
                dataset_handles = [plt.Line2D([], [], **COMPARE_COLOR_STYLE[key], label=label) for _, key, label in
                                   active]
                leg1 = ax.legend(handles=dataset_handles, loc="upper left", fontsize=6.5, title="Datasets")
                ax.add_artist(leg1)
            ax.legend(handles=variant_handles, loc="upper right", fontsize=6.5, title="Variants")
        else:
            _no_data(ax)

        # -- Panel F: Adiabatic vs diabatic metric -----------------------------
        ax = axes[1, 2]
        any_data = False
        for df, key, label in active:
            sub_all = _select(df, scenario="adiabatic_vs_diabatic", stage="metric")
            for variant, linestyle in (("adiabatic", "-"), ("diabatic", "--")):
                style = dict(COMPARE_COLOR_STYLE[key])
                style["linestyle"] = linestyle
                sub = sub_all[sub_all["variant"] == variant]
                any_data |= _plot_curve(ax, sub, label=f"{label} | {variant}", style=style,
                                        show_std_band=show_std_band, )
        if any_data:
            ax.set_xlabel("Hilbert-space dimension  N")
            ax.set_ylabel("Metric stage (ms)")
            ax.set_title(_panel_title("F", "Adiabatic vs diabatic metric"))
            _format_log_axes(ax)
            _set_xticks(ax, [4, 8, 16, 32, 64])
            # Separate legends: colours → datasets, line style → variant.
            variant_handles = [plt.Line2D([], [], color="k", linestyle="-", linewidth=1.7, label="adiabatic"),
                               plt.Line2D([], [], color="k", linestyle="--", linewidth=1.7, label="diabatic"), ]
            if show_dataset_legend:
                dataset_handles = [plt.Line2D([], [], **COMPARE_COLOR_STYLE[key], label=label) for _, key, label in
                                   active]
                leg1 = ax.legend(handles=dataset_handles, loc="upper left", fontsize=6.5, title="Datasets")
                ax.add_artist(leg1)
            ax.legend(handles=variant_handles, loc="upper right", fontsize=6.5, title="Variants")
        else:
            _no_data(ax)

        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.995)

    return fig


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _format_log_axes(ax: plt.Axes) -> None:
    """Apply log scales and a readable tick formatter to both axes."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    formatter = mticker.FuncFormatter(_format_log_tick)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", alpha=0.22)


def _set_xticks(ax: plt.Axes, ticks: list[float]) -> None:
    """Fix major tick positions on a log-scale x-axis and suppress minor ticks."""
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _no_data(ax: plt.Axes, text: str = "no data") -> None:
    """Show a centred 'no data' message and hide the axes frame."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _select(df: pd.DataFrame, *, scenario: str, variant: str | None = None, stage: str | None = None, ) -> pd.DataFrame:
    """Filter *df* to rows matching *scenario* and optionally *variant* / *stage*."""
    out = df[df["scenario"] == scenario]
    if variant is not None:
        out = out[out["variant"] == variant]
    if stage is not None:
        out = out[out["stage"] == stage]
    return out


def _format_log_tick(value: float, _pos: int | None = None) -> str:
    """Format a tick value on a log axis with appropriate precision."""
    if value <= 0 or not np.isfinite(value):
        return ""
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def _panel_title(letter: str, title: str) -> str:
    """Return a formatted panel title string like ``'A  –  My title'``."""
    return f"{letter}  –  {title}"


def _plot_curve(ax: plt.Axes, df: pd.DataFrame, *, label: str, style: dict[str, Any], x_col: str = "sweep_val_num",
                y_col: str = "mean_s", err_col: str = "std_s", show_std_band: bool = True, ) -> bool:
    """Plot one aggregated curve on *ax*.

    Returns ``True`` when data were available and a line was drawn,
    ``False`` when the input was empty or contained no valid points.
    """
    curve = _aggregate_curve(df, x_col=x_col, y_col=y_col, err_col=err_col)
    if curve.empty:
        return False

    x = curve[x_col].to_numpy(dtype=float)
    y = _ms(curve["y"].to_numpy(dtype=float))
    err = _ms(curve["err"].to_numpy(dtype=float))

    # Log axes cannot display non-positive values — keep only valid points.
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not np.any(valid):
        return False
    x, y, err = x[valid], y[valid], err[valid]

    ax.plot(x, y, label=label, **style)

    if show_std_band and np.isfinite(err).any():
        lower, upper = _safe_log_band(y, err)
        ax.fill_between(x, lower, upper, color=style.get("color", "0.5"), alpha=0.10,
                        zorder=style.get("zorder", 5) - 1, )

    return True


def _ms(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert seconds to milliseconds."""
    return np.asarray(values, dtype=float) * 1e3


def _safe_log_band(y: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive lower/upper band suitable for log-scale fill_between."""
    err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    lower = y - err
    positive = y[y > 0]
    floor = positive.min() * 1e-3 if positive.size else 1e-30
    return np.maximum(lower, floor), y + err


def _aggregate_curve(df: pd.DataFrame, *, x_col: str = "sweep_val_num", y_col: str = "mean_s",
                     err_col: str = "std_s", ) -> pd.DataFrame:
    """Average duplicate x-values so each x maps to exactly one plotted point.

    Duplicates arise when a coarse metadata column (e.g. ``machine``) combines
    several runs at the same sweep value.  The mean is taken for both *y* and
    *err* columns.
    """
    if df.empty:
        return pd.DataFrame(columns=[x_col, "y", "err"])

    agg_spec: dict[str, tuple[str, str]] = {"y": (y_col, "mean")}
    if err_col in df.columns:
        agg_spec["err"] = (err_col, "mean")

    out = df.groupby(x_col, as_index=False).agg(**agg_spec).sort_values(x_col)
    if "err" not in out.columns:
        out["err"] = np.nan
    return out
