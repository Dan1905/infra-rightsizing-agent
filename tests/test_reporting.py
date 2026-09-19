from rightsizer.backends.base import MIB
from rightsizer.reporting import format_for_llm, format_table

from .conftest import make_metrics


def test_docker_table_has_no_kubernetes_columns_and_flags_lifetime_restarts():
    m = make_metrics(kind="container", restart_scope="lifetime", restart_count=4,
                     mem_limit_bytes=1024 * MIB)
    table = format_table([m])
    assert "REPL" not in table and "req/lim" not in table
    assert "4*" in table and "not within the window" in table


def test_kubernetes_table_shows_requests_and_replicas():
    m = make_metrics(replicas=3, replicas_available=2, mem_request_bytes=256 * MIB,
                     mem_limit_bytes=512 * MIB, cpu_request_cores=0.2, cpu_limit_cores=1.0)
    table = format_table([m])
    assert "REPL" in table and "2/3" in table and "256MiB/512MiB" in table


def test_prompt_rendering_states_scope_and_declared_window():
    lifetime = format_for_llm([make_metrics(restart_scope="lifetime", restart_count=9)])
    assert "NOT limited to the window" in lifetime
    windowed = format_for_llm([make_metrics(min_window="24h")])
    assert "declared min window: 24h" in windowed
    assert "per pod, from the busiest replica" in windowed
