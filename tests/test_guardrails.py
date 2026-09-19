import pytest

from agent.backends.base import (
    MIB,
    GuardrailError,
    check_cpu,
    check_cpu_request,
    check_memory,
    check_memory_request,
    check_window,
    parse_duration,
)

from .conftest import make_metrics


@pytest.mark.parametrize(
    "text, seconds",
    [("30m", 1800), ("1h", 3600), ("1h30m", 5400), ("24h", 86400), ("7d", 604800)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "1x", "h1", "1h foo"])
def test_parse_duration_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_memory_limit_hard_floor():
    with pytest.raises(GuardrailError, match="hard floor of 128"):
        check_memory(100, None)


def test_memory_limit_must_clear_observed_peak():
    metrics = make_metrics(mem_max_bytes=900 * MIB)
    with pytest.raises(GuardrailError, match="minimum 1080 MiB"):
        check_memory(512, metrics)
    check_memory(1200, metrics)  # clears 1.2x


def test_cpu_limit_floor():
    with pytest.raises(GuardrailError):
        check_cpu(0.1)
    check_cpu(0.25)


def test_request_floors_are_lower_than_limit_floors():
    check_memory_request(64, make_metrics())
    check_cpu_request(0.05)
    with pytest.raises(GuardrailError):
        check_memory_request(16, None)
    with pytest.raises(GuardrailError):
        check_cpu_request(0.02)


def test_request_must_clear_observed_peak():
    with pytest.raises(GuardrailError, match="request leaves less than"):
        check_memory_request(300, make_metrics(mem_max_bytes=900 * MIB))


def test_declared_window_blocks_short_observation():
    with pytest.raises(GuardrailError, match="at least 24h"):
        check_window(make_metrics(window="1h", min_window="24h"))
    check_window(make_metrics(window="24h", min_window="24h"))
    check_window(make_metrics(window="1h", min_window=None))


def test_unreadable_declared_window_blocks_rather_than_passes():
    with pytest.raises(GuardrailError, match="unreadable"):
        check_window(make_metrics(min_window="a day"))
