import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from ._utils import real_samples
from .exceptions import IOErrorGeodesiQ, ValidationError
from .warnings import GeodesiQWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


class PulseControl:
    """
    A class to represent a control pulse for quantum optimal control. This class depends on the control pulse computed
    in the ControlModel class. The class allows for the synthesis of the pulse, filtering, and plotting of the pulse
    shape and its Fourier spectrum.
    """

    # Padding modes of filtered_pulse(): hold the end values, or a padtype of scipy.signal.sosfiltfilt.
    _FILTER_PADDINGS = ("hold", "odd", "even", "constant", None)
    # filtered_pulse() warns when an endpoint moves by more than this fraction of the pulse range.
    _FILTER_ENDPOINT_RTOL = 1e-2

    def __init__(self, pulse: ArrayLike, duration: float) -> None:
        """
        Initialize the PulseControl object with the control pulse and its duration. The samples are assumed to be
        uniformly spaced in time, from 0 to ``duration``.

        Parameters
        ----------
        pulse : ArrayLike
            Control pulse values, uniformly sampled in time.
        duration : float
            Duration of the control pulse (t_f).
        """

        values = np.asarray(pulse)
        if values.ndim != 1:
            raise ValidationError(f"pulse must be one-dimensional; received shape {values.shape}.")
        if values.size < 2:
            raise ValidationError("pulse must contain at least two samples.")
        values = np.asarray(real_samples(values, "pulse"), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValidationError("pulse contains NaN or infinite values.")

        if not isinstance(duration, (int, float, np.integer, np.floating)) or isinstance(duration, bool):
            raise ValidationError("duration must be a finite positive number.")
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValidationError("duration must be a finite positive number.")

        self._pulse = values.copy()
        self._duration = duration
        self._pulse_times = np.linspace(0.0, duration, values.size, dtype=float)

    @property
    def pulse(self) -> NDArray[np.float64]:
        """Return a copy of the pulse samples."""
        return self._pulse.copy()

    @property
    def times(self) -> NDArray[np.float64]:
        """Return a copy of the physical-time sample grid."""
        return self._pulse_times.copy()

    @property
    def duration(self) -> float:
        """Total pulse duration."""
        return self._duration

    # ------------------------------------------------------------
    #               Methods of PulseControl class
    # ------------------------------------------------------------

    def discretized_pulse(self, linear_steps: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Obtains the piecewise linear function from control pulse.


        Parameters
        ----------
        linear_steps : int
            Number of samples (>= 2) of the piecewise linear approximation of the control pulse.

        Returns
        -------
        new_time : np.ndarray
            Physical time array for the piecewise linear approximation.
        approx_sol : np.ndarray
            Control pulse values corresponding to the new time array for the piecewise linear approximation.

        """
        if not isinstance(linear_steps, (int, np.integer)) or isinstance(linear_steps, bool) or linear_steps < 2:
            raise ValidationError("linear_steps must be an integer >= 2.")

        new_time = np.linspace(self._pulse_times[0], self._pulse_times[-1], int(linear_steps))
        approx_sol = np.interp(new_time, self._pulse_times, self._pulse)

        return new_time, approx_sol

    def fourier_spectrum(
        self, window_len: int = 256, hop: int = 32, detrend: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the Short-Time Fourier Transform (STFT) spectrum of the control pulse.

        Parameters
        ----------
        window_len : int, optional
            The length of the window segment in number of samples. Default is 256.
        hop : int, optional
            The number of samples to advance between successive windows. Default is 32.
        detrend : bool, optional
            Subtract the mean of each window segment before the Fourier transform, which removes the DC part of the
            spectrum (``scipy.signal.ShortTimeFFT.stft_detrend`` with ``detr="constant"``). A geodesic pulse sweeps
            between two control values, so without it the zero-frequency row dominates the spectrum. Default is
            False.

        Returns
        -------
        frequencies : np.ndarray
            Frequencies corresponding to the STFT rows.
        times : np.ndarray
            Time array corresponding to the centers of each STFT time slice.
        magnitude : np.ndarray
            2D magnitude spectrum of the control pulse (shape: [frequencies, times]).
        """
        if not isinstance(window_len, (int, np.integer)) or isinstance(window_len, bool) or int(window_len) < 2:
            raise ValidationError("window_len must be an integer >= 2.")
        window_len = int(window_len)
        if window_len >= self._pulse.size:
            raise ValidationError("window_len must be smaller than the length of the STFT array.")

        if not isinstance(hop, (int, np.integer)) or isinstance(hop, bool) or int(hop) < 1:
            raise ValidationError("hop must be a positive integer.")
        hop = int(hop)
        if hop > window_len:
            raise ValidationError("hop must not exceed window_len.")
        if not isinstance(detrend, (bool, np.bool_)):
            raise ValidationError("detrend must be a boolean.")

        dt = float(self._pulse_times[1] - self._pulse_times[0])
        if not np.isfinite(dt) or dt <= 0:
            raise ValidationError("Pulse sampling interval must be finite and positive.")

        fs = float(1.0 / dt)
        win = hann(window_len, sym=False)
        sft = ShortTimeFFT(win, hop=hop, fs=fs, scale_to="magnitude")

        stft_matrix = sft.stft_detrend(self._pulse, "constant") if detrend else sft.stft(self._pulse)

        magnitude = np.abs(stft_matrix)
        frequencies = sft.f
        times = sft.t(len(self._pulse))

        return frequencies, times, magnitude

    def filtered_pulse(
        self, cutoff_freq: float, filter_order: int = 3, padding: str | None = "hold", causal: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Apply a low-pass Butterworth filter to the control pulse.

        By default the filter is applied forward and backward (``scipy.signal.sosfiltfilt``): there is no phase shift,
        but the filter is acausal, the magnitude response is the square of the designed one (gain 1/2, i.e. -6 dB, at
        ``cutoff_freq``) and the effective order is ``2 * filter_order``. With ``causal=True`` it is applied once
        (``scipy.signal.sosfilt``), as a transmission line would: gain 1/sqrt(2) (-3 dB) at ``cutoff_freq`` and a
        phase lag.

        Low-pass filtering moves the endpoints of a pulse that is steep there, as geodesic pulses usually are.
        ``Dynamics`` still measures fidelities against the eigenstates at ``pulse_initial`` and ``pulse_final``, so a
        ``GeodesiQWarning`` is issued when an endpoint moves by more than 1% of the pulse range.

        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency in units of 1 / time (e.g., Hz if time is in seconds) for the low-pass Butterworth filter.
        filter_order : int
            Order of the Butterworth filter.
        padding : str | None
            How the pulse is extended beyond its ends for the zero-phase filter. ``"hold"`` (default) holds the first
            and last values for as long as the filter's impulse response takes to decay, as an arbitrary waveform
            generator does before and after the pulse. ``"odd"``, ``"even"``, ``"constant"`` and None are passed to
            ``sosfiltfilt`` as ``padtype``; ``"odd"`` was the default up to version 0.3.0. Not used with
            ``causal=True``, which starts in the steady state of the first value (as after holding it).
        causal : bool
            Apply the filter forward only, instead of forward and backward.

        Returns
        -------
        pulse_times : np.ndarray
            Rescaled time array corresponding to the filtered control pulse.
        filtered_pulse : np.ndarray
            Returns the (butterworth-)filtered control pulse.

        Raises
        ------
        ValidationError
            If ``cutoff_freq`` is not positive and below the Nyquist frequency, if ``filter_order`` is not a positive
            integer, if ``padding`` or ``causal`` is invalid, or if the pulse is too short for a zero-phase filter
            without hold padding: ``sosfiltfilt`` pads ``3 * (2 * n + 1)`` samples at each end, with ``n`` the number
            of second-order sections (``3`` less for an odd order, e.g. 12 for order 3 and 15 for order 4), and needs
            more samples than that.

        Warns
        -----
        GeodesiQWarning
            If the first or last filtered value differs from the original one by more than 1% of the pulse range.
        """
        if not isinstance(cutoff_freq, (int, float, np.integer, np.floating)) or isinstance(cutoff_freq, bool):
            raise ValidationError("cutoff_freq must be a finite positive number.")
        cutoff_freq = float(cutoff_freq)
        if not np.isfinite(cutoff_freq) or cutoff_freq <= 0:
            raise ValidationError(f"Cutoff frequency must be positive and finite. Given: {cutoff_freq}")

        if not isinstance(filter_order, (int, np.integer)) or isinstance(filter_order, bool) or filter_order < 1:
            raise ValidationError("filter_order must be a positive integer.")
        filter_order = int(filter_order)

        if padding not in self._FILTER_PADDINGS:
            raise ValidationError(f"padding must be one of {', '.join(map(repr, self._FILTER_PADDINGS))}.")
        if not isinstance(causal, (bool, np.bool_)):
            raise ValidationError("causal must be True or False.")

        # Compute the sampling rate from the rescaled time array
        dt = float(np.abs(self._pulse_times[1] - self._pulse_times[0]))  # Uniform spacing by construction

        sampling_rate = 1.0 / dt
        nyquist_freq = sampling_rate / 2.0

        if cutoff_freq >= nyquist_freq:
            raise ValidationError(f"cutoff_freq must be smaller than the Nyquist frequency {nyquist_freq:.5g}.")

        # Design Butterworth filter
        sos = sp.signal.butter(N=filter_order, Wn=cutoff_freq, btype="low", fs=sampling_rate, output="sos")

        if causal:
            # Start in the steady state of the first value, as after holding it before the pulse.
            initial = sp.signal.sosfilt_zi(sos) * self._pulse[0]
            filtered_pulse, _ = sp.signal.sosfilt(sos, self._pulse, zi=initial)
        elif padding == "hold":
            # Hold the end values until the impulse response has decayed by 1e-12. Its slowest pole decays at the rate
            # 2 pi f_c sin(pi / (2 N)) of the analog Butterworth prototype.
            decay_time = np.log(1e12) / (2 * np.pi * cutoff_freq * np.sin(np.pi / (2 * filter_order)))
            hold = int(np.ceil(decay_time / dt))
            held = np.concatenate((np.full(hold, self._pulse[0]), self._pulse, np.full(hold, self._pulse[-1])))
            filtered_pulse = sp.signal.sosfiltfilt(sos, held, padtype="constant")[hold:-hold]
        else:
            # sosfiltfilt pads both ends with padlen samples and needs a longer pulse; padlen as computed by SciPy
            # (first-order sections of odd filter orders have zero trailing coefficients, which shorten it).
            n_taps = 2 * len(sos) + 1 - min(int(np.sum(sos[:, 2] == 0)), int(np.sum(sos[:, 5] == 0)))
            padlen = 3 * n_taps if padding is not None else 0
            if self._pulse.size <= padlen:
                raise ValidationError(
                    f"The pulse has {self._pulse.size} samples, but zero-phase filtering with "
                    f"filter_order={filter_order} pads {padlen} samples at each end and needs more than {padlen}. "
                    "Use a larger pulse_accuracy in solve_problem(), a lower filter_order or padding='hold'."
                )
            filtered_pulse = sp.signal.sosfiltfilt(sos, self._pulse, padtype=padding)

        # Dynamics compares with the eigenstates at the original endpoints, which a moved endpoint no longer reaches.
        # The rounding term keeps a constant pulse (zero range) from warning.
        scale = float(np.max(np.abs(self._pulse)))
        tolerance = self._FILTER_ENDPOINT_RTOL * float(np.ptp(self._pulse)) + 100 * np.finfo(float).eps * scale
        moved = [
            f"{label} {new:.6g} instead of {old:.6g}"
            for label, new, old in (
                ("starts at", filtered_pulse[0], self._pulse[0]),
                ("ends at", filtered_pulse[-1], self._pulse[-1]),
            )
            if abs(new - old) > tolerance
        ]
        if moved:
            warnings.warn(
                f"The filtered pulse {' and '.join(moved)}. Fidelities from Dynamics still refer to the eigenstates at "
                "the original endpoints; use a higher cutoff_freq if the endpoints must be kept.",
                GeodesiQWarning,
                stacklevel=2,
            )

        return self._pulse_times.copy(), np.asarray(filtered_pulse, dtype=float)

    def plot_pulse(
        self,
        fig: "Figure | SubFigure | None" = None,
        ax: "Axes | None" = None,
        show: bool = True,
        **plot_kwargs: Any,
    ) -> "tuple[Figure | SubFigure, Axes]":
        """
        Plot the (real-time) control pulse.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created. When ``ax`` is given, the pulse is
            drawn on it and ``fig`` is ignored.
        show : bool
            Whether to call ``plt.show()``. Pass ``show=False`` to keep drawing on a provided figure/axis.
        plot_kwargs : dict
            Dictionary of style changes to ax.plot()

        Returns
        -------
        fig, ax: Figure | SubFigure, Axes
            Figure and axes for the construction of a custom plot.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        fig = ax.figure

        ax.plot(self._pulse_times, self._pulse, **plot_kwargs)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Control Pulse")
        if show:
            plt.show()

        return fig, ax

    def export_pulse(self, filename: str | Path, file_extension: str = "npz", overwrite: bool = False) -> Path:
        """
        Export (real-time) pulse data to a (npz, txt, csv) file.

        Values are written with full double precision (``%.17g``), so that e.g. nanosecond-scale times expressed in
        seconds are preserved. The ``txt`` format is whitespace-delimited and the ``csv`` format is comma-delimited;
        both have a one-line header with the column names.

        Parameters
        ----------
        filename : str | Path
            Name for the data file saved. The extension is appended unless the name already ends with it, so dots in
            the name are kept (``"pulse_tf1.5"`` is written to ``pulse_tf1.5.npz``).
        file_extension : str
            Data type the pulse should be stored in ('npz', 'txt' or 'csv', case-insensitive, with or without a
            leading dot). Default is 'npz'.
        overwrite : bool
            If False (default), raise instead of replacing an existing file.

        Returns
        -------
        output_path: Path
            Path of the written file.

        Raises
        ------
        ValidationError
            If ``file_extension`` is not a supported type.
        IOErrorGeodesiQ
            If the file exists and ``overwrite`` is False.
        """

        if not isinstance(file_extension, str):
            raise ValidationError(
                f"file_extension must be a string ('npz', 'txt' or 'csv'); received {type(file_extension).__name__}."
            )
        file_extension = file_extension.lower().removeprefix(".")
        if file_extension not in ("npz", "txt", "csv"):
            raise ValidationError(
                f"Unsupported file_extension '{file_extension}'. Supported types are: 'npz', 'txt', and 'csv'."
            )

        output_path = Path(filename)
        if not output_path.name.lower().endswith(f".{file_extension}"):
            output_path = output_path.with_name(f"{output_path.name}.{file_extension}")

        t: np.ndarray = np.asarray(self._pulse_times)
        pulse: np.ndarray = np.asarray(self._pulse)

        # Mode "x" creates the file only if it does not exist yet, so the check and the write are one step.
        try:
            with open(output_path, "wb" if overwrite else "xb") as file:
                if file_extension == "npz":
                    np.savez(file, times=t, pulse=pulse)
                else:
                    delimiter, header = (" ", "t pulse") if file_extension == "txt" else (",", "t,pulse")
                    np.savetxt(
                        file, np.column_stack((t, pulse)), delimiter=delimiter, header=header, comments="", fmt="%.17g"
                    )
        except FileExistsError:
            raise IOErrorGeodesiQ(
                f"File already exists (choose overwrite=True to remove safety check.): {output_path}"
            ) from None

        return output_path
