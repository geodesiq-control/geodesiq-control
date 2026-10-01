from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import scipy as sp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from .exceptions import IOErrorGeodesiQ, ValidationError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class PulseControl:
    """
    A class to represent a control pulse for quantum optimal control. This class depends on the control pulse computed
    in the ControlModel class. The class allows for the synthesis of the pulse, filtering, and plotting of the pulse
    shape and its Fourier spectrum.
    """

    def __init__(self, pulse: np.ndarray, duration: float):
        """
        Initialize the PulseControl object with the control pulse and its duration. The samples are assumed to be
        uniformly spaced in time, from 0 to ``duration``.

        Parameters
        ----------
        pulse : np.ndarray
            Control pulse values, uniformly sampled in time.
        duration : float
            Duration of the control pulse (t_f).
        """

        values = np.asarray(pulse)
        if values.ndim != 1:
            raise ValidationError(f"pulse must be one-dimensional; received shape {values.shape}.")
        if values.size < 2:
            raise ValidationError("pulse must contain at least two samples.")
        if np.iscomplexobj(values):
            if not np.allclose(values.imag, 0.0):
                raise ValidationError("pulse must contain real-valued samples.")
            values = values.real
        values = np.asarray(values, dtype=float)
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
    def pulse(self) -> np.ndarray:
        """Return a copy of the pulse samples."""
        return self._pulse.copy()

    @property
    def times(self) -> np.ndarray:
        """Return a copy of the physical-time sample grid."""
        return self._pulse_times.copy()

    @property
    def duration(self) -> float:
        """Total pulse duration."""
        return self._duration

    # ------------------------------------------------------------
    #               Methods of PulseControl class
    # ------------------------------------------------------------

    def discretized_pulse(self, linear_steps: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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

    def fourier_spectrum(self, window_len: int = 256, hop: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Short-Time Fourier Transform (STFT) spectrum of the control pulse.

        Parameters
        ----------
        window_len : int, optional
            The length of the window segment in number of samples. Default is 256.
        hop : int, optional
            The number of samples to advance between successive windows. Default is 32.

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

        dt = float(self._pulse_times[1] - self._pulse_times[0])
        if not np.isfinite(dt) or dt <= 0:
            raise ValidationError("Pulse sampling interval must be finite and positive.")

        fs = float(1.0 / dt)
        win = hann(window_len, sym=False)
        sft = ShortTimeFFT(win, hop=hop, fs=fs, scale_to="magnitude")

        stft_matrix = sft.stft(self._pulse)

        magnitude = np.abs(stft_matrix)
        frequencies = sft.f
        times = sft.t(len(self._pulse))

        return frequencies, times, magnitude

    def filtered_pulse(self, cutoff_freq: float, filter_order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a low-pass Butterworth filter to the control pulse.

        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency in units of 1 / time (e.g., Hz if time is in seconds) for the low-pass Butterworth filter.
        filter_order : int
            Order of the Butterworth filter.

        Returns
        -------
        pulse_times : np.ndarray
            Rescaled time array corresponding to the filtered control pulse.
        filtered_pulse : np.ndarray
            Returns the (butterworth-)filtered control pulse.
        """
        if not isinstance(cutoff_freq, (int, float, np.integer, np.floating)) or isinstance(cutoff_freq, bool):
            raise ValidationError("cutoff_freq must be a finite positive number.")
        cutoff_freq = float(cutoff_freq)
        if not np.isfinite(cutoff_freq) or cutoff_freq <= 0:
            raise ValidationError(f"Cutoff frequency must be positive and finite. Given: {cutoff_freq}")

        if not isinstance(filter_order, (int, np.integer)) or isinstance(filter_order, bool) or filter_order < 1:
            raise ValidationError("filter_order must be a positive integer.")
        filter_order = int(filter_order)

        # Compute the sampling rate from the rescaled time array
        dt = float(np.abs(self._pulse_times[1] - self._pulse_times[0]))  # Uniform spacing by construction

        sampling_rate = 1.0 / dt
        nyquist_freq = sampling_rate / 2.0

        if cutoff_freq >= nyquist_freq:
            raise ValidationError(f"cutoff_freq must be smaller than the Nyquist frequency {nyquist_freq:.5g}.")

        # Design Butterworth filter
        sos = sp.signal.butter(N=filter_order, Wn=cutoff_freq, btype="low", fs=sampling_rate, output="sos")

        # Apply the filter to the pulse using filtfilt for zero-phase filtering
        filtered_pulse = sp.signal.sosfiltfilt(sos, self._pulse)

        return self._pulse_times.copy(), filtered_pulse

    def plot_pulse(self, show: bool = True, **plot_kwargs: Any) -> "Tuple[Figure, Axes]":
        """
        Plot the (real-time) control pulse.

        Parameters
        ----------
        show : bool
            Show plot before possibly adding plot_kwargs
        plot_kwargs : dict
            Dictionary of style changes to ax.plot()

        Returns
        -------
        fig, ax: Figure, Axes
            Figure and axes for the construction of a custom plot.

        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self._pulse_times, self._pulse, **plot_kwargs)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Control Pulse")
        if show:
            plt.show()
        else:
            plt.close(fig)

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
            Name for the data file saved.
        file_extension : str
            Data type the pulse should be stored in (i.e. 'txt', 'npz', 'csv'). Default is 'npz'.
        overwrite : bool
            Ensures accidental overwrites.

        Returns
        -------
        output_path: Path
            Path of the written file.
        """

        # Remove possible file_extension starting with a dot
        if file_extension.startswith("."):
            file_extension = file_extension[1:]

        if file_extension not in ("npz", "txt", "csv"):
            raise ValidationError(
                f"Unsupported data_type '{file_extension}'. Supported types are: 'npz', 'txt', and 'csv'."
            )

        output_path = Path(filename)
        if output_path.suffix != f".{file_extension}":
            output_path = output_path.with_suffix(f".{file_extension}")

        if output_path.exists() and not overwrite:
            raise IOErrorGeodesiQ(f"File already exists (choose overwrite=True to remove safety check.): {output_path}")

        t: np.ndarray = np.asarray(self._pulse_times)
        pulse: np.ndarray = np.asarray(self._pulse)

        # Save data depending on users preference
        if file_extension == "npz":
            np.savez(output_path, times=t, pulse=pulse)
        elif file_extension == "txt":
            np.savetxt(output_path, np.column_stack((t, pulse)), header="t pulse", comments="", fmt="%.17g")
        else:
            np.savetxt(
                output_path, np.column_stack((t, pulse)), delimiter=",", header="t,pulse", comments="", fmt="%.17g"
            )

        return output_path
