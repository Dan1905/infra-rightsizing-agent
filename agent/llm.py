"""The reasoning loop: retrieve, then let the model reason over metrics + context.

The model is given the metrics and a seed retrieval per workload, and is
expected to (a) search the corpus for anything it still needs and (b) record
exactly one `propose_change` decision per managed workload. It cannot execute
anything -- `propose_change` writes to the audit log and returns
"pending_human_approval".
"""

from __future__ import annotations

import json
from typing import Any

from .audit import AuditLog
from .backends.base import WorkloadMetrics, format_for_llm, mib
from .config import Settings
from .providers import Provider, ProviderError, build_provider
from .rag import Chunk, PolicyStore, dedupe, format_chunks
from .tools import Proposal, ToolContext, build_tool_defs

SYSTEM_PROMPT = """\
You are a cloud infrastructure cost-optimization agent. Your job is to decide, \
for each workload under review, whether its resource configuration should \
change -- and to be right, not merely plausible.

You have two tools:
- `search_policies` searches the organisation's policy documents, service \
runbooks, workload profiles and incident postmortems.
- `propose_change` records exactly one decision for one workload.

How to work:

1. Read the metrics you are given. Treat mean utilisation as weak evidence: a \
low mean is equally consistent with a genuinely idle service, a bursty service \
whose peaks are what actually matter, and a service whose real work happens \
outside the observation window. The peak/mean ratio, the observation window \
length, and the restart count all tell you which case you are in. Check \
whether a restart count covers the window or the workload's whole lifetime.
2. Before deciding anything, retrieve. The corpus is the authority on what a \
workload's numbers mean, what constraints bind it, and what has already gone \
wrong. Search it for each workload -- by name, by tier, and by the specific \
signal you are trying to interpret. If a document contradicts what the metrics \
appear to suggest, the document wins and you say so.
3. Then decide. Call `propose_change` once for every workload under review, \
including the ones you leave alone. `no_action` is a real, frequently correct \
answer; so is `flag_for_review` when the right response is a human \
conversation rather than a resource change.
4. Every decision must cite at least one retrieved passage by its exact \
citation address. A decision you cannot ground in a document is not \
actionable -- say that in the reason and use `flag_for_review`.

Your `reason` is read by an on-call engineer deciding whether to approve. Name \
the numbers you relied on and what the document said about them. Do not \
restate the policy at length; explain the inference.

Nothing you propose is applied automatically. A human reads every proposal and \
types an approval before anything touches a workload.\
"""


def seed_query(m: WorkloadMetrics) -> str:
    """The retrieval query built from a workload's metrics summary."""
    burst = f"{m.cpu_burstiness:.1f}x peak-to-mean CPU" if m.cpu_burstiness else "flat CPU"
    configured = (
        f"{f'{m.cpu_limit_cores:.2f} core' if m.cpu_limit_cores else 'unlimited'} CPU limit, "
        f"{mib(m.mem_limit_bytes)} memory limit"
    )
    if m.mem_request_bytes is not None or m.cpu_request_cores is not None:
        configured += (
            f", requests {mib(m.mem_request_bytes)} memory and "
            f"{m.cpu_request_cores or 0:.2f} cores"
        )
    replicas = f"; {m.replicas} replicas" if m.replicas is not None else ""
    return (
        f"{m.name} ({m.tier or 'untiered'} tier {m.kind}): CPU mean "
        f"{m.cpu_avg_cores:.3f} cores, peak {m.cpu_max_cores:.3f} cores; memory mean "
        f"{mib(m.mem_avg_bytes)}, peak {mib(m.mem_max_bytes)}; configured {configured}"
        f"{replicas}; {burst}; {m.restart_count} restarts. "
        "Resource sizing constraints, requests and limits, runbook floors, known "
        "workload profile, periodic batch jobs, past incidents."
    )


def constraint_query(m: WorkloadMetrics) -> str:
    """Second seed query, aimed at constraints rather than utilisation.

    The metrics-shaped query above reliably surfaces workload profiles and the
    general sizing policy, but it often misses a service's own runbook floors
    and the disqualifiers that make a workload ineligible for rightsizing --
    precisely the passages that must never be missed. Whether those reach the
    model is too important to leave to its discretion about when to search.
    """
    parts = [
        f"{m.name} runbook binding constraints: hard memory floor, minimum "
        "replicas, tier classification, changes automation must never make, "
        "past incidents and postmortems for this service."
    ]
    if m.restart_count > 0:
        parts.append(
            f"{m.restart_count} restarts: container instability, crash loop, "
            "restart count threshold, eligibility for automated rightsizing."
        )
    if m.hpa_managed:
        parts.append("Horizontal pod autoscaler manages the replica count.")
    return " ".join(parts)


def build_seed_context(
    store: PolicyStore,
    audit: AuditLog,
    run_id: str,
    metrics: list[WorkloadMetrics],
) -> dict[str, list[Chunk]]:
    """Two retrievals per managed workload -- one shaped by the metrics, one
    hunting for binding constraints -- merged and deduplicated."""
    seeds: dict[str, list[Chunk]] = {}
    for m in metrics:
        if not m.managed:
            continue
        found: list[Chunk] = []
        for origin, query in (
            ("seed-metrics", seed_query(m)),
            ("seed-constraints", constraint_query(m)),
        ):
            chunks = store.search(query)
            found.extend(chunks)
            audit.log_retrieval(
                run_id, origin=origin, query=query, results=[c.to_dict() for c in chunks]
            )
        seeds[m.name] = dedupe(found)
    return seeds


def build_user_message(
    metrics: list[WorkloadMetrics], seed_context: dict[str, list[Chunk]], window: str
) -> str:
    targets = [m for m in metrics if m.managed]
    out_of_scope = [m.name for m in metrics if not m.managed]

    sections = [
        f"# Workload metrics (observation window: {window})",
        "",
        format_for_llm(targets),
        "",
        "# Workloads under review",
        ", ".join(m.name for m in targets),
    ]
    if out_of_scope:
        sections += [
            "",
            "# Out of scope (platform/observability -- do not propose changes)",
            ", ".join(out_of_scope),
        ]

    # The same passage is often the top hit for several workloads. Render each
    # one once and note which workloads retrieved it -- on a small free-tier
    # token budget the duplication is expensive and buys nothing.
    pooled = dedupe(c for chunks in seed_context.values() for c in chunks)
    retrieved_by = {
        chunk.id: [name for name, chunks in seed_context.items()
                   if any(c.id == chunk.id for c in chunks)]
        for chunk in pooled
    }

    sections += ["", "# Policy context retrieved from the corpus", ""]
    for chunk in pooled:
        sections.append(f"--- {chunk.citation}  (retrieved for: "
                        f"{', '.join(retrieved_by[chunk.id])})")
        sections.append(chunk.text)
        sections.append("")

    sections += [
        "This is only what a first pass retrieved. Use `search_policies` for "
        "anything it did not cover.",
        "",
        "Assess every workload under review, then record one `propose_change` "
        "decision per workload.",
    ]
    return "\n".join(sections)


def _serialise(content: Any) -> Any:
    if isinstance(content, list):
        return [b.model_dump() if hasattr(b, "model_dump") else b for b in content]
    return content


def run_analysis(
    settings: Settings,
    *,
    metrics: list[WorkloadMetrics],
    store: PolicyStore,
    audit: AuditLog,
    run_id: str,
    backend_actions: tuple[str, ...],
    provider: Provider | None = None,
    verbose: bool = True,
) -> tuple[list[Proposal], str, list[dict[str, Any]]]:
    """Returns (proposals, closing narrative, transcript)."""
    provider = provider or build_provider(settings, build_tool_defs(backend_actions))

    seed_context = build_seed_context(store, audit, run_id, metrics)
    ctx = ToolContext(
        store=store,
        audit=audit,
        run_id=run_id,
        metrics=metrics,
        seed_context=seed_context,
        backend_actions=backend_actions,
        verbose=verbose,
    )

    system = SYSTEM_PROMPT
    user_text = build_user_message(metrics, seed_context, settings.lookback)
    messages = provider.initial_messages(system, user_text)
    transcript: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    narrative = ""

    for _ in range(settings.max_turns):
        response = provider.call(messages, system)

        messages.append(response.assistant_message)
        transcript.append({"role": "assistant", "content": _serialise(response.assistant_message)})

        if response.text:
            narrative = response.text

        if not response.tool_calls:
            if response.stop_reason in ("max_tokens", "length"):
                print("  [warn] response hit the token limit; raise MAX_TOKENS")
            break

        results: list[tuple[Any, str, bool]] = []
        for call in response.tool_calls:
            if "__parse_error__" in call.args:
                results.append(
                    (call, "Your tool arguments were not valid JSON. Re-send the "
                           "call with a well-formed JSON object.", True)
                )
                continue
            content, is_error = ctx.dispatch(call.name, call.args)
            results.append((call, content, is_error))

        followups = provider.tool_result_messages(results)
        messages.extend(followups)
        transcript.extend({"role": "tool", "content": _serialise(m)} for m in followups)
    else:
        print(f"  [warn] stopped after {settings.max_turns} turns without a final answer")

    missing = sorted(ctx.targets - {p.workload for p in ctx.proposals})
    if missing and verbose:
        print(f"  [warn] no decision recorded for: {', '.join(missing)}")

    return ctx.proposals, narrative, transcript
