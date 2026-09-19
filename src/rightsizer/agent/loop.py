"""The reasoning loop: retrieve, then let the model reason over metrics + context.

The model is given the metrics and a seed retrieval per workload, and is
expected to (a) search the corpus for anything it still needs and (b) record
exactly one `propose_change` decision per managed workload. It cannot execute
anything -- `propose_change` writes to the audit log and returns
"pending_human_approval".
"""

from __future__ import annotations

from typing import Any

from ..audit import AuditLog
from ..backends.base import WorkloadMetrics
from ..config import Settings
from ..llm.providers import Provider, build_provider
from ..retrieval.store import Chunk, PolicyStore, dedupe
from .prompts import (
    SYSTEM_PROMPT,
    build_user_message,
    constraint_query,
    platform_query,
    seed_query,
)
from .proposals import Proposal, ToolContext
from .tools import build_tool_defs


def build_seed_context(
    store: PolicyStore,
    audit: AuditLog,
    run_id: str,
    metrics: list[WorkloadMetrics],
) -> dict[str, list[Chunk]]:
    """Two or three retrievals per managed workload -- shaped by its metrics,
    by its binding constraints, and by its platform -- merged and deduplicated."""
    seeds: dict[str, list[Chunk]] = {}
    for m in metrics:
        if not m.managed:
            continue
        found: list[Chunk] = []
        queries = [
            ("seed-metrics", seed_query(m)),
            ("seed-constraints", constraint_query(m)),
        ]
        if (pq := platform_query(m)) is not None:
            queries.append(("seed-platform", pq))
        for origin, query in queries:
            # The platform query is identical for every workload and covers a
            # whole policy document, so it is worth a wider net.
            chunks = store.search(query, k=5 if origin == "seed-platform" else None)
            found.extend(chunks)
            audit.log_retrieval(
                run_id, origin=origin, query=query, results=[c.to_dict() for c in chunks]
            )
        seeds[m.name] = dedupe(found)
    return seeds


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
    preflight: Any = None,
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
        preflight=preflight,
        verbose=verbose,
    )

    system = SYSTEM_PROMPT
    user_text = build_user_message(metrics, seed_context, settings.lookback)
    messages = provider.initial_messages(system, user_text)
    transcript: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    narrative = ""

    nudges_left = 2
    for _ in range(settings.max_turns):
        response = provider.call(messages, system)

        messages.append(response.assistant_message)
        transcript.append({"role": "assistant", "content": _serialise(response.assistant_message)})

        if response.text:
            narrative = response.text

        if not response.tool_calls:
            if response.stop_reason in ("max_tokens", "length"):
                print("  [warn] response hit the token limit; raise MAX_TOKENS")
            # Smaller models sometimes end a turn -- occasionally with an empty
            # message -- before deciding every workload. Point at what is
            # missing rather than silently accepting a partial plan.
            undecided = sorted(ctx.targets - {p.workload for p in ctx.proposals})
            if undecided and nudges_left > 0:
                nudges_left -= 1
                if verbose:
                    print(f"  [loop] no decision yet for {', '.join(undecided)}; asking again")
                nudge = {
                    "role": "user",
                    "content": (
                        "You have not recorded a decision for: "
                        f"{', '.join(undecided)}. Call `propose_change` once for "
                        "each of them now."
                    ),
                }
                messages.append(nudge)
                transcript.append(nudge)
                continue
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
