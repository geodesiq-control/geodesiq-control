from geodesiq.exceptions import (
    GeodesiQError,
    ValidationError,
)


def test_exception_hierarchy():
    assert issubclass(ValidationError, GeodesiQError)


def test_base_exception_without_message():
    err = GeodesiQError()
    assert str(err) == "[geodesiq]"


def test_exception_message_has_prefix():
    err = ValidationError("invalid parameter")
    assert str(err) == "[geodesiq] invalid parameter"
