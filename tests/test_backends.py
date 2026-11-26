"""Backend probing before rollout/trainer placement."""

from megarl.backends import (
    BackendValidation,
    KNOWN_BACKENDS,
    usable_backends,
    validate_backend,
    validate_backends,
)


def test_cpu_backend_is_available_and_smokes():
    result = validate_backend("cpu")
    assert isinstance(result, BackendValidation)
    assert result.available and result.smoke_passed
    assert "cpu" in usable_backends()


def test_unavailable_devices_are_reported_not_raised():
    result = validate_backend("definitely-not-a-device")
    assert not result.available
    assert not result.smoke_passed
    assert result.reason


def test_validate_all_returns_every_known_device():
    results = validate_backends()
    assert set(results) == set(KNOWN_BACKENDS)
    assert results["cpu"].smoke_passed
