import os

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from geodesiq.exceptions import IOErrorGeodesiQ, ValidationError
from geodesiq.pulses import PulseControl

# ------------------------------------------------------------
# Ramp pulse as pytest.fixtures
# ------------------------------------------------------------


@pytest.fixture
def sample_pulse_data():
    """Generates a simple linear ramp for deterministic testing."""
    duration = 10.0
    t = np.linspace(0, duration, 100)
    pulse = 20 * t - 10  # Linear ramp from -10 to 10 over 10 seconds
    return pulse, duration


@pytest.fixture
def default_pulse(sample_pulse_data):
    """Provides a basic PulseControl instance without an explicit method."""
    pulse, duration = sample_pulse_data
    return PulseControl(pulse=pulse, duration=duration)


# ------------------------------------------------------------
# Testing discretized_pulse and fourier_spectrum methods
# ------------------------------------------------------------


def test_pulse_accepts_a_list():
    """pulse is ArrayLike, so a list is accepted and stored as float64 (P-08)."""
    pulse = PulseControl(pulse=[0.0, 1.0, 3.0], duration=2.0)

    assert pulse.pulse.dtype == np.float64
    np.testing.assert_allclose(pulse.pulse, [0.0, 1.0, 3.0])
    np.testing.assert_allclose(pulse.times, [0.0, 1.0, 2.0])


def test_discretized_pulse_bounds(default_pulse):
    """Ensure discretization keeps the original boundaries of the pulse time."""
    new_times, approx_sol = default_pulse.discretized_pulse(linear_steps=5)

    assert len(new_times) == 5
    assert len(approx_sol) == 5
    assert new_times[0] == default_pulse._pulse_times[0]
    assert new_times[-1] == default_pulse._pulse_times[-1]


def test_fourier_spectrum_with_linear_ramp(sample_pulse_data):
    """Verify FFT properties for a linear ramp from -10 to 10 over 10 seconds."""
    pulse, duration = sample_pulse_data
    pc = PulseControl(pulse, duration)

    frequencies, times, magnitude = pc.fourier_spectrum(window_len=50)
    assert np.all(frequencies >= 0)


def test_fourier_spectrum_small_size(sample_pulse_data):
    """Verify that an error is raised if the length of the pulse is smaller than the window length."""
    pulse, duration = sample_pulse_data
    pc = PulseControl(pulse, duration)

    with pytest.raises(ValidationError, match="window_len must be smaller"):
        pc.fourier_spectrum(window_len=len(pc._pulse_times) * 2)


def test_fourier_spectrum_detrend_removes_the_dc_part():
    """detrend=True subtracts each window's mean: an offset vanishes, a sine is kept (F-23)."""
    duration, samples, window_len = 10.0, 1001, 100
    t = np.linspace(0, duration, samples)
    pc = PulseControl(pulse=5.0 + np.sin(2 * np.pi * 2.0 * t), duration=duration)

    frequencies, times, raw = pc.fourier_spectrum(window_len=window_len, hop=10)
    _, _, detrended = pc.fourier_spectrum(window_len=window_len, hop=10, detrend=True)

    # Windows that lie fully inside the pulse (the boundary ones also contain padding)
    half = window_len / 2 * duration / (samples - 1)
    inside = (times >= half) & (times <= duration - half)
    peak = np.argmin(np.abs(frequencies - 2.0))
    np.testing.assert_allclose(raw[0, inside], 5.0, rtol=1e-2)
    assert np.max(detrended[0, inside]) < 1e-2
    np.testing.assert_allclose(detrended[peak, inside], raw[peak, inside], rtol=2e-2)


@pytest.mark.parametrize(("frequency", "amplitude", "window_len"), [(2.0, 3.0, 100), (2.5, 0.7, 200), (8.0, 1.0, 50)])
def test_fourier_spectrum_frequency_axis_and_sine_amplitude(frequency, amplitude, window_len):
    """A sine of frequency f and amplitude A on a frequency bin peaks at f with magnitude A (D-07, D-20 (v)).

    The one-sided spectrum doubles the positive frequencies, so it gives the physical amplitude and not A/2.
    """
    duration, samples = 10.0, 1001
    t = np.linspace(0, duration, samples)
    pc = PulseControl(pulse=amplitude * np.sin(2 * np.pi * frequency * t), duration=duration)

    frequencies, times, magnitude = pc.fourier_spectrum(window_len=window_len, hop=10)

    fs = (samples - 1) / duration
    np.testing.assert_allclose(frequencies, np.arange(window_len // 2 + 1) * fs / window_len)
    assert magnitude.shape == (frequencies.size, times.size)
    # Windows that lie fully inside the pulse (the boundary ones also contain padding)
    half = window_len / 2 / fs
    inside = (times >= half) & (times <= duration - half)
    assert np.all(frequencies[np.argmax(magnitude[:, inside], axis=0)] == pytest.approx(frequency))
    peak = np.argmin(np.abs(frequencies - frequency))
    np.testing.assert_allclose(magnitude[peak, inside], amplitude, rtol=1e-6)


def test_fourier_spectrum_keeps_the_dc_value_of_an_offset_sine():
    """Only the positive frequencies are doubled: an offset c stays c at zero frequency, a sine of amplitude A is A."""
    duration, samples, window_len = 10.0, 1001, 100
    t = np.linspace(0, duration, samples)
    pc = PulseControl(pulse=2.0 + 0.3 * np.cos(2 * np.pi * 2.0 * t), duration=duration)

    frequencies, times, magnitude = pc.fourier_spectrum(window_len=window_len, hop=10, mfft=4 * window_len)

    inside = (times >= 0.5) & (times <= duration - 0.5)
    np.testing.assert_allclose(magnitude[0, inside], 2.0, rtol=1e-6)
    np.testing.assert_allclose(magnitude[np.argmin(np.abs(frequencies - 2.0)), inside], 0.3, rtol=1e-6)


def test_fourier_spectrum_rejects_a_non_boolean_detrend(default_pulse):
    with pytest.raises(ValidationError, match="detrend must be a boolean"):
        default_pulse.fourier_spectrum(window_len=50, detrend="constant")


def test_fourier_spectrum_hold_padding_keeps_the_boundary_windows_flat():
    """The default hold padding extends a constant pulse by itself; zero padding adds a jump to 0 (F-22)."""
    duration, samples, window_len = 10.0, 1001, 100
    pc = PulseControl(pulse=np.full(samples, 5.0), duration=duration)

    _, times, held = pc.fourier_spectrum(window_len=window_len, hop=10)
    _, _, zeros = pc.fourier_spectrum(window_len=window_len, hop=10, padding="zeros")
    _, _, detrended = pc.fourier_spectrum(window_len=window_len, hop=10, detrend=True)

    # Every window, also those that overhang the ends, sees the same constant as the central one
    assert times[0] < 0 and times[-1] > duration
    middle = held[:, [times.size // 2]]
    np.testing.assert_allclose(held[0], 5.0, rtol=1e-12)
    np.testing.assert_allclose(held, np.broadcast_to(middle, held.shape), rtol=0, atol=1e-12)
    np.testing.assert_allclose(detrended, 0.0, atol=1e-12)
    # The windows centred on the ends are half zeros: the DC part halves and higher frequencies appear
    edge = np.argmin(np.abs(times))
    assert zeros[0, edge] == pytest.approx(2.5, rel=5e-2)
    assert np.max(zeros[2:, edge]) > 0.5


@pytest.mark.parametrize("padding", ["zeros", "even", "odd"])
def test_fourier_spectrum_passes_the_padding_to_scipy(padding):
    from scipy import signal

    duration, samples, window_len, hop = 10.0, 1001, 100, 10
    t = np.linspace(0, duration, samples)
    values = t**2 - 3 * t
    pc = PulseControl(pulse=values, duration=duration)

    _, _, magnitude = pc.fourier_spectrum(window_len=window_len, hop=hop, padding=padding)

    sft = signal.ShortTimeFFT(
        signal.windows.hann(window_len, sym=False), hop=hop, fs=100.0, fft_mode="onesided2X", scale_to="magnitude"
    )
    np.testing.assert_allclose(magnitude, np.abs(sft.stft(values, padding=padding)), rtol=0, atol=1e-12)


def test_fourier_spectrum_mfft_interpolates_the_frequency_axis():
    """A zero-padded FFT refines the frequency grid and keeps the values on the original bins."""
    duration, samples, window_len, hop = 10.0, 1001, 100, 10
    t = np.linspace(0, duration, samples)
    pc = PulseControl(pulse=np.sin(2 * np.pi * 2.3 * t), duration=duration)

    frequencies, times, magnitude = pc.fourier_spectrum(window_len=window_len, hop=hop)
    fine_frequencies, fine_times, fine = pc.fourier_spectrum(window_len=window_len, hop=hop, mfft=4 * window_len)

    assert fine_frequencies.size == 2 * window_len + 1
    np.testing.assert_allclose(fine_frequencies[::4], frequencies)
    np.testing.assert_array_equal(fine_times, times)
    np.testing.assert_allclose(fine[::4], magnitude, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mfft", [49, 100.0, True, "256"])
def test_fourier_spectrum_rejects_an_invalid_mfft(default_pulse, mfft):
    with pytest.raises(ValidationError, match="mfft must be None or an integer >= window_len"):
        default_pulse.fourier_spectrum(window_len=50, mfft=mfft)


@pytest.mark.parametrize("padding", ["edge", "constant", None, 0])
def test_fourier_spectrum_rejects_an_unknown_padding(default_pulse, padding):
    with pytest.raises(ValidationError, match="padding must be one of"):
        default_pulse.fourier_spectrum(window_len=50, padding=padding)


# ------------------------------------------------------------
# Testing plot_pulse method
# ------------------------------------------------------------


def test_plot_pulse_without_showing(default_pulse):
    """Verify plotting logic executes and returns Matplotlib objects without blocking."""
    fig, ax = default_pulse.plot_pulse(show=False)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Time $t$"


def test_plot_pulse_with_show_invokes_matplotlib_show(default_pulse, monkeypatch):
    """The show=True branch should call matplotlib.pyplot.show()."""
    import matplotlib.pyplot as plt

    called = {"count": 0}

    def fake_show():
        called["count"] += 1

    monkeypatch.setattr(plt, "show", fake_show)

    fig, ax = default_pulse.plot_pulse(show=True)

    assert called["count"] == 1
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_pulse_on_provided_axes(default_pulse):
    """A provided axis is drawn on, its figure is returned, and it is not closed."""
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2)
    out_fig, out_ax = default_pulse.plot_pulse(ax=ax_right, show=False)

    assert out_ax is ax_right
    assert out_fig is fig
    assert len(ax_right.lines) == 1
    assert len(ax_left.lines) == 0
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_plot_pulse_on_provided_figure(default_pulse):
    """A provided figure without an axis gets a new subplot and is not closed."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    out_fig, ax = default_pulse.plot_pulse(fig=fig, show=False)

    assert out_fig is fig
    assert ax in fig.axes
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_plot_pulse_on_provided_subfigure(default_pulse):
    """A provided SubFigure gets a new subplot and is returned."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    assert isinstance(subfigs, np.ndarray)
    subfig = subfigs[0]
    out_fig, ax = default_pulse.plot_pulse(fig=subfig, show=False)

    assert out_fig is subfig
    assert ax in subfig.axes
    plt.close(fig)


def test_plot_pulse_labels_and_title(default_pulse):
    """The axis labels and the title can be set, as in plot_populations() (D-16)."""
    import matplotlib.pyplot as plt

    fig, ax = default_pulse.plot_pulse(xlabel="t (ns)", ylabel="Detuning", title="Pulse", show=False)

    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("t (ns)", "Detuning", "Pulse")
    _, default_ax = default_pulse.plot_pulse(show=False)
    assert (default_ax.get_ylabel(), default_ax.get_title()) == ("Control Pulse", "")
    plt.close("all")


def test_plot_pulse_reuses_the_single_axis_of_a_provided_figure(default_pulse):
    """A figure that already has one axis is drawn on it, not covered by a second, overlapping axis (D-16)."""
    import matplotlib.pyplot as plt

    fig, existing = plt.subplots()
    out_fig, ax = default_pulse.plot_pulse(fig=fig, show=False)
    default_pulse.plot_pulse(fig=fig, show=False)

    assert out_fig is fig
    assert ax is existing
    assert fig.axes == [existing]
    assert len(existing.lines) == 2
    plt.close(fig)


def test_plot_pulse_on_a_figure_with_several_axes_needs_ax(default_pulse):
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(1, 2)
    with pytest.raises(ValidationError, match="pass the one to draw on as ax"):
        default_pulse.plot_pulse(fig=fig, show=False)
    plt.close(fig)


# ------------------------------------------------------------
# Testing export_pulse method
# ------------------------------------------------------------


def test_export_pulse_as_npz(default_pulse, tmp_path):
    """Verify .npz file export works seamlessly using a temporary test directory."""

    test_file_base = os.path.join(tmp_path, "test_pulse")
    default_pulse.export_pulse(filename=test_file_base, file_extension="npz")
    expected_full_path = test_file_base + ".npz"
    assert os.path.exists(expected_full_path)

    # Load back data to verify integrity
    loaded_data = np.load(expected_full_path)
    np.testing.assert_array_equal(loaded_data["pulse"], default_pulse._pulse)


def test_export_pulse_as_txt(default_pulse, tmp_path):
    """Verify .txt file export works correctly and contains proper formatting."""
    test_file_base = os.path.join(tmp_path, "test_pulse_txt")
    default_pulse.export_pulse(filename=test_file_base, file_extension="txt")
    expected_full_path = test_file_base + ".txt"
    assert os.path.exists(expected_full_path)

    # Load back data to verify integrity
    loaded_data = np.loadtxt(expected_full_path, skiprows=1)
    assert loaded_data.shape[0] == len(default_pulse._pulse)
    np.testing.assert_array_equal(loaded_data[:, 0], default_pulse.times)
    np.testing.assert_array_equal(loaded_data[:, 1], default_pulse._pulse)


def test_export_pulse_raises_on_existing_file(default_pulse, tmp_path):
    """Verify export_pulse raises IOErrorGeodesiQ when file exists and overwrite=False."""
    from geodesiq.exceptions import IOErrorGeodesiQ

    test_file_base = os.path.join(tmp_path, "test_pulse_overwrite")

    # First export
    default_pulse.export_pulse(filename=test_file_base, file_extension="npz")

    # Second export should raise error
    with pytest.raises(IOErrorGeodesiQ, match="File already exists"):
        default_pulse.export_pulse(filename=test_file_base, file_extension="npz", overwrite=False)


def test_export_pulse_with_overwrite(default_pulse, tmp_path):
    """Verify export_pulse successfully overwrites when overwrite=True."""
    test_file_base = os.path.join(tmp_path, "test_pulse_overwrite2")

    # First export
    default_pulse.export_pulse(filename=test_file_base, file_extension="npz")
    original_path = test_file_base + ".npz"

    # Second export with overwrite=True should succeed
    default_pulse.export_pulse(filename=test_file_base, file_extension="npz", overwrite=True)
    assert os.path.exists(original_path)


def test_export_pulse_unsupported_extension(default_pulse, tmp_path):
    """Verify export_pulse raises error for unsupported file extensions."""
    test_file_base = os.path.join(tmp_path, "test_pulse_invalid")

    with pytest.raises(ValidationError, match="Unsupported file_extension"):
        default_pulse.export_pulse(filename=test_file_base, file_extension="json")

    assert not os.listdir(tmp_path)


def test_export_pulse_strips_leading_dot(default_pulse, tmp_path):
    """Verify export_pulse correctly handles file extensions with leading dot."""
    test_file_base = os.path.join(tmp_path, "test_pulse_dot")
    default_pulse.export_pulse(filename=test_file_base, file_extension=".npz")
    expected_full_path = test_file_base + ".npz"
    assert os.path.exists(expected_full_path)


# ------------------------------------------------------------
# Testing filtered_pulse method
# ------------------------------------------------------------


#  A low cutoff moves the endpoints of the ramp, which filtered_pulse() reports with a GeodesiQWarning (D-05).
IGNORE_MOVED_ENDPOINTS = pytest.mark.filterwarnings("ignore:.*The filtered pulse:geodesiq.warnings.GeodesiQWarning")


@IGNORE_MOVED_ENDPOINTS
def test_filtered_pulse_returns_correct_shapes(default_pulse):
    """Verify filtered_pulse returns arrays of correct shape."""
    cutoff_freq = 0.1
    times, filtered = default_pulse.filtered_pulse(cutoff_freq=cutoff_freq)

    assert len(times) == len(default_pulse._pulse)
    assert len(filtered) == len(default_pulse._pulse)


def test_filtered_pulse_negative_cutoff_raises_error(default_pulse):
    """Verify filtered_pulse raises ValidationError for negative cutoff frequency."""
    with pytest.raises(ValidationError, match="Cutoff frequency must be positive"):
        default_pulse.filtered_pulse(cutoff_freq=-0.1)


def test_filtered_pulse_zero_cutoff_raises_error(default_pulse):
    """Verify filtered_pulse raises ValidationError for zero cutoff frequency."""
    with pytest.raises(ValidationError, match="Cutoff frequency must be positive"):
        default_pulse.filtered_pulse(cutoff_freq=0)


def test_filtered_pulse_cutoff_exceeds_nyquist(default_pulse):
    """Verify filtered_pulse raises ValidationError when cutoff exceeds Nyquist frequency."""
    # For 100 samples over 10 seconds, dt = 0.1, Nyquist = 5 Hz
    with pytest.raises(ValidationError, match="cutoff_freq must be smaller than the Nyquist frequency"):
        default_pulse.filtered_pulse(cutoff_freq=10)


def test_filtered_pulse_invalid_filter_order_zero(default_pulse):
    """Verify filtered_pulse raises ValidationError for filter_order < 1."""
    with pytest.raises(ValidationError, match="filter_order must be a positive integer"):
        default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=0)


def test_filtered_pulse_invalid_filter_order_negative(default_pulse):
    """Verify filtered_pulse raises ValidationError for negative filter_order."""
    with pytest.raises(ValidationError, match="filter_order must be a positive integer"):
        default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=-1)


def test_filtered_pulse_non_integer_filter_order(default_pulse):
    """Verify filtered_pulse raises ValidationError for non-integer filter_order."""
    with pytest.raises(ValidationError, match="filter_order must be a positive integer"):
        default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=3.5)


@IGNORE_MOVED_ENDPOINTS
def test_filtered_pulse_preserves_time_array(default_pulse):
    """Verify filtered_pulse returns the same time array."""
    times, _ = default_pulse.filtered_pulse(cutoff_freq=0.1)
    np.testing.assert_array_equal(times, default_pulse._pulse_times)


def test_filter_order_bool(default_pulse):
    with pytest.raises(ValidationError, match="filter_order must be a positive integer"):
        default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=True)


# sosfiltfilt pads padlen samples at each end: 3 * (2 * n_sections + 1), minus 3 for the first-order section of an odd
# order, i.e. 12 for orders 3 and 15 for order 4.
@pytest.mark.parametrize(("filter_order", "padlen"), [(3, 12), (4, 15)])
def test_filtered_pulse_too_short_raises_validation_error(filter_order: int, padlen: int):
    # SciPy raised a raw ValueError "The length of the input vector x must be greater than padlen" (D-06). This
    # applies to the sosfiltfilt padding modes; the default hold padding extends the pulse itself.
    pulse = PulseControl(pulse=np.linspace(-1.0, 1.0, padlen), duration=10.0)

    with pytest.raises(ValidationError, match=f"pads {padlen} samples"):
        pulse.filtered_pulse(cutoff_freq=0.1, filter_order=filter_order, padding="odd")


@IGNORE_MOVED_ENDPOINTS
@pytest.mark.parametrize(("filter_order", "padlen"), [(3, 12), (4, 15)])
@pytest.mark.parametrize("padding", ["odd", "hold"])
def test_filtered_pulse_accepts_the_shortest_valid_pulse(filter_order: int, padlen: int, padding: str):
    pulse = PulseControl(pulse=np.linspace(-1.0, 1.0, padlen + 1), duration=10.0)

    _, filtered = pulse.filtered_pulse(cutoff_freq=0.1, filter_order=filter_order, padding=padding)

    assert filtered.shape == (padlen + 1,)
    assert np.all(np.isfinite(filtered))


def geodesic_lz_pulse() -> PulseControl:
    """Geometric LZ pulse from -10 to 10 at t_f = 10, which is steep at both ends (the review's reference pulse)."""
    from geodesiq import ControlModel

    model = ControlModel(H_d=np.array([[0.0, 1.0], [1.0, 0.0]]), H_c=np.diag([1.0, -1.0]))
    model.set_control(pulse_initial=-10.0, pulse_final=10.0, initial_state=0, alpha=2.0, beta=2.0)
    model.solve_problem(pulse_accuracy=1000)
    return model.synthesize_pulse(10.0)


@pytest.mark.filterwarnings("ignore::geodesiq.warnings.NumericalStabilityWarning")
class TestFilterEndpointsAndResponse:
    """D-05: endpoint handling, gain at the cutoff, and the causal mode of filtered_pulse()."""

    def test_moved_endpoints_are_reported(self):
        from geodesiq.warnings import GeodesiQWarning

        pulse = geodesic_lz_pulse()

        with pytest.warns(GeodesiQWarning, match="starts at -8.76.* instead of -10 and ends at 8.76.* instead of 10"):
            _, filtered = pulse.filtered_pulse(cutoff_freq=1.0)

        # Hold padding treats both ends alike: the antisymmetric pulse stays antisymmetric.
        np.testing.assert_allclose(filtered, -filtered[::-1], atol=1e-10)

    def test_hold_padding_converges_and_odd_padding_keeps_the_former_result(self):
        import scipy.signal as signal

        pulse = geodesic_lz_pulse()
        values, dt = pulse.pulse, pulse.times[1] - pulse.times[0]
        sos = signal.butter(3, 0.3, fs=1.0 / dt, output="sos")
        hold = 100_000
        held = np.concatenate((np.full(hold, values[0]), values, np.full(hold, values[-1])))

        with pytest.warns(Warning):
            _, filtered = pulse.filtered_pulse(cutoff_freq=0.3)
        with pytest.warns(Warning):
            _, odd = pulse.filtered_pulse(cutoff_freq=0.3, padding="odd")

        np.testing.assert_allclose(filtered, signal.sosfiltfilt(sos, held)[hold:-hold], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(odd, signal.sosfiltfilt(sos, values))

    def test_a_pulse_that_is_flat_at_its_ends_keeps_its_endpoints(self):
        t = np.linspace(0.0, 10.0, 1001)
        pulse = PulseControl(pulse=np.tanh(4.0 * (t - 5.0)), duration=10.0)

        for causal in (False, True):
            _, filtered = pulse.filtered_pulse(cutoff_freq=2.0, causal=causal)
            assert abs(filtered[0] - pulse.pulse[0]) < 1e-6
            if not causal:
                assert abs(filtered[-1] - pulse.pulse[-1]) < 1e-6

    @IGNORE_MOVED_ENDPOINTS
    @pytest.mark.parametrize(("causal", "gain"), [(False, 0.5), (True, 1.0 / np.sqrt(2.0))])
    def test_gain_at_the_cutoff(self, causal: bool, gain: float):
        # Zero-phase filtering squares the magnitude response: -6 dB at the cutoff instead of -3 dB.
        t = np.linspace(0.0, 200.0, 20001)
        pulse = PulseControl(pulse=np.sin(2.0 * np.pi * t), duration=200.0)

        _, filtered = pulse.filtered_pulse(cutoff_freq=1.0, causal=causal)

        assert np.max(np.abs(filtered[5000:15000])) == pytest.approx(gain, abs=2e-3)

    def test_causal_filter_starts_in_the_steady_state_of_the_first_value(self):
        pulse = geodesic_lz_pulse()

        with pytest.warns(Warning, match="ends at"):
            _, filtered = pulse.filtered_pulse(cutoff_freq=1.0, causal=True)

        assert filtered[0] == pytest.approx(-10.0, abs=1e-9)

    @pytest.mark.parametrize(("kwargs", "message"), [({"padding": "zero"}, "padding must be one of"),
                                                     ({"causal": 1}, "causal must be True or False")])
    def test_invalid_options_raise_validation_error(self, default_pulse, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            default_pulse.filtered_pulse(cutoff_freq=0.1, **kwargs)


# ------------------------------------------------------------
# Testing initialization and attributes
# ------------------------------------------------------------


def test_pulse_control_initialization(sample_pulse_data):
    """Verify PulseControl initializes correctly with all parameters."""
    pulse, duration = sample_pulse_data

    pc = PulseControl(pulse=pulse, duration=duration)

    assert pc._duration == duration
    np.testing.assert_array_equal(pc._pulse, pulse)


def test_pulse_control_pulse_times_generation(sample_pulse_data):
    """Verify _pulse_times are generated correctly."""
    pulse, duration = sample_pulse_data
    pc = PulseControl(pulse=pulse, duration=duration)

    # Should have same length as pulse
    assert len(pc._pulse_times) == len(pulse)
    # Should start at 0
    assert pc._pulse_times[0] == 0
    # Should end at duration
    assert pc._pulse_times[-1] == duration
    # Should be uniformly spaced
    differences = np.diff(pc._pulse_times)
    np.testing.assert_array_almost_equal(differences, differences[0] * np.ones_like(differences))


# ------------------------------------------------------------
# Export precision, return value and input validation
# ------------------------------------------------------------


@pytest.mark.parametrize(("file_extension", "delimiter"), [("txt", None), ("csv", ",")])
def test_export_keeps_full_precision_for_nanosecond_pulses(tmp_path, file_extension: str, delimiter):
    """Times in seconds for a ns-scale pulse used to be written as 0.00000000."""
    pc = PulseControl(pulse=np.linspace(1e-6, 3e-6, 11) + np.pi * 1e-12, duration=5e-9)

    output_path = pc.export_pulse(tmp_path / "ns_pulse", file_extension=file_extension)

    loaded = np.loadtxt(output_path, delimiter=delimiter, skiprows=1)
    np.testing.assert_array_equal(loaded[:, 0], pc.times)
    np.testing.assert_array_equal(loaded[:, 1], pc.pulse)


def test_export_csv_has_header_and_comma_delimiter(default_pulse, tmp_path):
    output_path = default_pulse.export_pulse(tmp_path / "pulse", file_extension="csv")

    lines = output_path.read_text().splitlines()
    assert lines[0] == "times,pulse"
    assert len(lines[1].split(",")) == 2


@pytest.mark.parametrize("file_extension", ["npz", "txt", "csv"])
def test_export_uses_the_same_column_names_in_every_format(default_pulse, tmp_path, file_extension: str):
    """The text headers name the columns as the NPZ keys do (W2-108)."""
    output_path = default_pulse.export_pulse(tmp_path / "pulse", file_extension=file_extension)

    if file_extension == "npz":
        with np.load(output_path) as data:
            names = list(data.files)
    else:
        delimiter = "," if file_extension == "csv" else " "
        names = output_path.read_text().splitlines()[0].split(delimiter)
    assert names == ["times", "pulse"]


def test_export_returns_path_and_does_not_print(default_pulse, tmp_path, capsys):
    output_path = default_pulse.export_pulse(str(tmp_path / "pulse"), file_extension=".npz")

    assert output_path == tmp_path / "pulse.npz"
    assert output_path.exists()
    assert capsys.readouterr().out == ""


def test_export_unsupported_extension_is_checked_before_existing_file(default_pulse, tmp_path):
    (tmp_path / "pulse.json").write_text("{}")

    with pytest.raises(ValidationError, match="Unsupported file_extension"):
        default_pulse.export_pulse(tmp_path / "pulse", file_extension="json")


# ------------------------------------------------------------
# Export file names with dots and extension validation (D-12)
# ------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "file_extension", "expected"),
    [
        ("pulse_tf1.5", "npz", "pulse_tf1.5.npz"),
        ("pulse_tf1.5", "txt", "pulse_tf1.5.txt"),
        ("pulse.v2.csv", "csv", "pulse.v2.csv"),
        ("pulse.npz", "txt", "pulse.npz.txt"),
        ("pulse.NPZ", "npz", "pulse.NPZ"),
        ("pulse", "NPZ", "pulse.npz"),
        ("pulse", ".CSV", "pulse.csv"),
    ],
)
def test_export_appends_suffix_and_keeps_dots(default_pulse, tmp_path, filename: str, file_extension: str, expected):
    output_path = default_pulse.export_pulse(tmp_path / filename, file_extension=file_extension)

    assert output_path == tmp_path / expected
    assert sorted(path.name for path in tmp_path.iterdir()) == [expected]


def test_export_names_differing_after_a_dot_do_not_collide(default_pulse, tmp_path):
    first = default_pulse.export_pulse(tmp_path / "pulse_tf2.5")
    second = default_pulse.export_pulse(tmp_path / "pulse_tf2.7")

    assert first != second
    with np.load(second) as data:
        np.testing.assert_array_equal(data["pulse"], default_pulse.pulse)


@pytest.mark.parametrize("file_extension", [None, 3, b"npz"])
def test_export_rejects_non_string_extension(default_pulse, tmp_path, file_extension):
    with pytest.raises(ValidationError, match="file_extension must be a string"):
        default_pulse.export_pulse(tmp_path / "pulse", file_extension=file_extension)

    assert not any(tmp_path.iterdir())


def test_export_without_overwrite_leaves_existing_file_untouched(default_pulse, tmp_path):
    (tmp_path / "pulse.txt").write_text("keep")

    with pytest.raises(IOErrorGeodesiQ, match="File already exists"):
        default_pulse.export_pulse(tmp_path / "pulse", file_extension="txt")

    assert (tmp_path / "pulse.txt").read_text() == "keep"


# ------------------------------------------------------------
# Relative check of the imaginary part (D-13)
# ------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_complex_pulse_with_negligible_imaginary_part_is_accepted(scale: float):
    values = scale * np.linspace(1.0, 2.0, 5)

    pc = PulseControl(pulse=values * (1 + 1e-14j), duration=1.0)

    np.testing.assert_array_equal(pc.pulse, values)
    assert pc.pulse.dtype == np.float64


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_complex_pulse_with_relevant_imaginary_part_is_rejected(scale: float):
    with pytest.raises(ValidationError, match="pulse must contain real-valued samples"):
        PulseControl(pulse=scale * np.linspace(1.0, 2.0, 5) * (1 + 1e-6j), duration=1.0)


def test_purely_imaginary_pulse_is_rejected():
    with pytest.raises(ValidationError, match="pulse must contain real-valued samples"):
        PulseControl(pulse=np.array([0.0, 1e-20j]), duration=1.0)


@pytest.mark.parametrize("linear_steps", [1, 0, -3, 2.5, True, "4"])
def test_discretized_pulse_rejects_invalid_steps(default_pulse, linear_steps):
    with pytest.raises(ValidationError, match="linear_steps must be an integer >= 2"):
        default_pulse.discretized_pulse(linear_steps=linear_steps)


def test_discretized_pulse_accepts_numpy_integer_and_matches_linear_interpolation(default_pulse):
    times, values = default_pulse.discretized_pulse(linear_steps=np.int64(7))

    assert times.shape == (7,)
    np.testing.assert_allclose(values, np.interp(times, default_pulse.times, default_pulse.pulse))


@pytest.mark.parametrize("cutoff_freq", ["0.1", None, True, np.nan, np.inf])
def test_filtered_pulse_rejects_invalid_cutoff_types(default_pulse, cutoff_freq):
    with pytest.raises(ValidationError):
        default_pulse.filtered_pulse(cutoff_freq=cutoff_freq)


@IGNORE_MOVED_ENDPOINTS
def test_filtered_pulse_accepts_numpy_filter_order(default_pulse):
    times, filtered = default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=np.int64(2))

    assert filtered.shape == times.shape
