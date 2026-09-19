"""DockerBackend guardrails through preflight -- no Docker daemon needed."""

import pytest

from agent.backends.base import MIB
from agent.backends.docker import DockerBackend, _resolve_name

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
