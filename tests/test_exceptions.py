from geodesiq.exceptions import (
    GeodesiQError,
    MissingArgsError,
    ValidationError,
)


def test_exception_hierarchy():
    assert issubclass(ValidationError, GeodesiQError)


def test_validation_error_is_a_value_error():
    """Code catching ValueError keeps working when geodesiq raises ValidationError."""
    assert issubclass(ValidationError, ValueError)
    assert issubclass(MissingArgsError, ValueError)


def test_base_exception_without_message():
    err = GeodesiQError()
    assert str(err) == "[geodesiq]"


def test_exception_message_has_prefix():
    err = ValidationError("invalid parameter")
    assert str(err) == "[geodesiq] invalid parameter"
