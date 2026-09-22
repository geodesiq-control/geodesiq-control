from geodesiq.warnings import (
    ExperimentalFeatureWarning,
    GeodesiQWarning,
    NumericalStabilityWarning,
    PerformanceWarning,
)


def test_warning_hierarchy():
    assert issubclass(NumericalStabilityWarning, GeodesiQWarning)
    assert issubclass(PerformanceWarning, GeodesiQWarning)
    assert issubclass(ExperimentalFeatureWarning, GeodesiQWarning)


def test_base_warning_without_message():
    warn = GeodesiQWarning()
    assert str(warn) == "[geodesiq]"


def test_warning_message_has_prefix():
    warn = PerformanceWarning("high cost expected")
    assert str(warn) == "[geodesiq] high cost expected"
