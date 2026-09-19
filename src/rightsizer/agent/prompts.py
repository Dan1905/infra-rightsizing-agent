"""Everything the model reads: the system prompt, the retrieval queries built
from each workload's metrics, and the user message that presents both."""

from __future__ import annotations

from ..backends.base import WorkloadMetrics
from ..reporting import format_for_llm, mib
from ..retrieval.store import Chunk, dedupe


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


def platform_query(m: WorkloadMetrics) -> str | None:
    """Third seed query, for platform-specific rules.

    The two queries above are phrased around the workload, so they surface the
    workload's own documents -- and never the generic platform policy that
    says, for example, that on Kubernetes requests rather than limits are what
    cost money. Only issued where a platform policy applies.
    """
    if m.kind != "Deployment":
        return None
    return (
        "Kubernetes requests versus limits: which drives cost and node count, "
        "sizing requests, when a request change is worthwhile, limits, replica "
        "floors, horizontal pod autoscaler, tier 1 service restrictions."
    )


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
        # Chunks are stored with their address as a first line (it helps the
        # embedding); the header above already carries it.
        body = chunk.text.split("\n", 1)[1] if chunk.text.startswith("[") else chunk.text
        sections.append(body)
        sections.append("")

    sections += [
        "This is only what a first pass retrieved. Use `search_policies` for "
        "anything it did not cover.",
        "",
        "Assess every workload under review, then record one `propose_change` "
        "decision per workload.",
    ]
    return "\n".join(sections)
