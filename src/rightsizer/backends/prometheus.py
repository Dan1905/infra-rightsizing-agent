"""Minimal Prometheus HTTP API client shared by both backends."""

from __future__ import annotations

import requests

from .base import MetricsError


def instant_query(
    prom_url: str, expr: str, timeout: float = 10.0
) -> list[tuple[dict[str, str], float]]:
    """Run an instant query and return [(labels, value)]."""
    try:
        resp = requests.get(
            f"{prom_url.rstrip('/')}/api/v1/query",
            params={"query": expr},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise MetricsError(f"Prometheus query failed ({prom_url}): {exc}") from exc

    payload = resp.json()
    if payload.get("status") != "success":
        raise MetricsError(f"Prometheus error for `{expr}`: {payload.get('error')}")

    out: list[tuple[dict[str, str], float]] = []
    for series in payload["data"]["result"]:
        try:
            out.append((series["metric"], float(series["value"][1])))
        except (TypeError, ValueError):
            continue
    return out
