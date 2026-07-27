import os

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from geodesiq.exceptions import ValidationError
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
    loaded_data = np.loadtxt(expected_full_path, delimiter=",", skiprows=1)
    assert loaded_data.shape[0] == len(default_pulse._pulse)
    np.testing.assert_array_almost_equal(loaded_data[:, 1], default_pulse._pulse)


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
    from geodesiq.exceptions import MissingArgsError

    test_file_base = os.path.join(tmp_path, "test_pulse_invalid")

    with pytest.raises(MissingArgsError, match="Unsupported data_type"):
        default_pulse.export_pulse(filename=test_file_base, file_extension="json")


def test_export_pulse_strips_leading_dot(default_pulse, tmp_path):
    """Verify export_pulse correctly handles file extensions with leading dot."""
    test_file_base = os.path.join(tmp_path, "test_pulse_dot")
    default_pulse.export_pulse(filename=test_file_base, file_extension=".npz")
    expected_full_path = test_file_base + ".npz"
    assert os.path.exists(expected_full_path)


# ------------------------------------------------------------
# Testing filtered_pulse method
# ------------------------------------------------------------


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


def test_filtered_pulse_preserves_time_array(default_pulse):
    """Verify filtered_pulse returns the same time array."""
    times, _ = default_pulse.filtered_pulse(cutoff_freq=0.1)
    np.testing.assert_array_equal(times, default_pulse._pulse_times)


def test_filter_order_bool(default_pulse):
    with pytest.raises(ValidationError, match="filter_order must be a positive integer"):
        default_pulse.filtered_pulse(cutoff_freq=0.1, filter_order=True)


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
