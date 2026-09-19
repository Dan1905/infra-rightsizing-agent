"""DockerBackend guardrails through preflight -- no Docker daemon needed."""

import pytest

from rightsizer.backends.base import MIB
from rightsizer.backends.docker import DockerBackend, _resolve_name, configured_limits

from .conftest import make_metrics


def backend(allow_stop=False):
    b = DockerBackend.__new__(DockerBackend)  # skip docker.from_env()
    b.allow_stop = allow_stop
    return b


@pytest.mark.parametrize(
    "action, params, metrics, blocked_by",
    [
        ("set_memory_limit", {"memory_mib": 256}, make_metrics(), None),
        ("set_memory_limit", {"memory_mib": 64}, make_metrics(), "hard floor"),
        ("set_memory_limit", {"memory_mib": 100}, make_metrics(mem_max_bytes=95 * MIB),
         "hard floor"),
        ("set_memory_limit", {"memory_mib": 130}, make_metrics(mem_max_bytes=200 * MIB),
         "minimum 240"),
        ("set_cpu_limit", {"cpu_cores": 0.5}, make_metrics(), None),
        ("set_cpu_limit", {"cpu_cores": 0.1}, make_metrics(), "hard floor"),
        ("stop_container", {}, make_metrics(), "--allow-stop"),
        ("set_memory_limit", {"memory_mib": 256}, make_metrics(min_window="24h"), "24h"),
    ],
)
def test_preflight(action, params, metrics, blocked_by):
    result = backend().preflight(target="x", action=action, params=params, metrics=metrics)
    if blocked_by is None:
        assert result is None
    else:
        assert blocked_by in result


def test_stop_allowed_with_flag():
    assert backend(allow_stop=True).preflight(
        target="x", action="stop_container", params={}, metrics=make_metrics()) is None


def test_resolve_name_prefers_label_then_cgroup_id():
    cid = "a" * 64
    ids = {cid: "web-frontend"}
    assert _resolve_name({"name": "direct"}, ids) == "direct"
    assert _resolve_name({"id": f"/docker/{cid}"}, ids) == "web-frontend"
    assert _resolve_name({"id": "/system.slice"}, ids) is None


@pytest.mark.parametrize(
    "host_config, expected",
    [
        ({"Memory": 1073741824, "NanoCpus": 2_000_000_000}, (2.0, 1073741824.0)),  # --cpus
        ({"Memory": 0, "CpuQuota": 150_000, "CpuPeriod": 100_000}, (1.5, None)),   # quota
        ({"CpuQuota": 50_000, "CpuPeriod": 0}, (0.5, None)),       # default period
        ({}, (None, None)),                                           # unlimited
    ],
)
def test_limits_come_from_host_config(host_config, expected):
    assert configured_limits(host_config) == expected
