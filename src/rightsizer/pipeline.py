"""One run end to end: collect -> retrieve and reason -> show the plan ->
typed approval -> execute -> audit.

The approval gate is here, and only here. There is no way to skip it: the
non-interactive path (`analyze`) stops after printing the plan.
"""

from __future__ import annotations

import sys
import textwrap

from .agent.loop import run_analysis
from .agent.proposals import Proposal
from .agent.tools import build_tool_defs
from .audit import AuditLog
from .backends import build_backend
from .backends.base import Backend, MetricsError
from .config import Settings
from .llm.providers import ProviderError, build_provider
from .reporting import format_table
from .retrieval.store import PolicyStore

RULE = "=" * 78


SETUP_HINT = {
    "docker": "Is the Docker sandbox up?  make up",
    "kubernetes": (
        "Is the cluster up and Prometheus port-forwarded?  make k8s-up, then "
        "make port-forward (in its own terminal)"
    ),
}


def connect(s: Settings, *, allow_stop: bool = False) -> tuple[Backend, list] | None:
    """Build the backend and collect metrics, or print why not."""
    try:
        backend = build_backend(s, allow_stop=allow_stop)
        return backend, backend.collect()
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
    except Exception as exc:  # docker/kubernetes client setup failures
        print(f"error: could not connect to the {s.backend} backend: {exc}", file=sys.stderr)
    print(SETUP_HINT.get(s.backend, ""), file=sys.stderr)
    return None


def print_proposal(index: int, proposal: Proposal) -> None:
    print(f"\n{RULE}")
    print(f"[{index}] {proposal.workload}: {proposal.describe().upper()}")
    print(f"     confidence: {proposal.confidence}", end="")
    if proposal.estimated_saving:
        print(f"   estimated saving: {proposal.estimated_saving}", end="")
    print(f"\n{RULE}")
    print("Reasoning:")
    for line in textwrap.wrap(proposal.reason, width=76):
        print(f"  {line}")
    print("\nGrounded in:")
    for citation in proposal.policy_cited:
        print(f"  - {citation}")


def run(s: Settings, *, interactive: bool, allow_stop: bool = False) -> int:
    """Collect, reason, print the plan -- and, if interactive, ask for approval
    of each executable change and apply the approved ones. Returns an exit code."""
    audit = AuditLog(s.audit_db)

    connected = connect(s, allow_stop=allow_stop)
    if connected is None:
        return 2
    backend, metrics = connected

    managed = [m for m in metrics if m.managed]
    if not managed:
        print(f"No workloads in scope ({backend.describe_scope()}); nothing to assess.")
        return 0

    print(f"Backend: {backend.name}   window: {s.lookback}   provider: {s.provider}\n")
    print(format_table(metrics))

    store = PolicyStore(s)
    if store.count() == 0:
        print("\nerror: policy index is empty. Run `rightsize index` first.",
              file=sys.stderr)
        return 2

    try:
        provider = build_provider(s, build_tool_defs(backend.actions))
    except ProviderError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    run_id = audit.start_run(
        model=f"{provider.name}:{provider.model}",
        lookback=s.lookback,
        prometheus_url=s.prometheus_url,
        backend=backend.name,
    )
    print(f"\nrun: {run_id}\nReasoning over metrics and retrieved policy context ...\n")

    try:
        proposals, narrative, transcript = run_analysis(
            s, metrics=metrics, store=store, audit=audit, run_id=run_id,
            backend_actions=backend.actions, preflight=backend.preflight,
            provider=provider
        )
    except ProviderError as exc:
        audit.finish_run(run_id, container_count=len(managed), proposal_count=0)
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    audit.finish_run(
        run_id,
        container_count=len(managed),
        proposal_count=len(proposals),
        transcript=transcript,
    )

    if narrative:
        print(f"\n{RULE}\nAgent summary\n{RULE}")
        for paragraph in narrative.split("\n"):
            print("\n".join(textwrap.wrap(paragraph, width=78)) if paragraph.strip() else "")

    if not proposals:
        print("\nNo proposals recorded.")
        return 0

    print(f"\n{RULE}\nPROPOSED PLAN ({len(proposals)} decisions)\n{RULE}")
    for i, proposal in enumerate(proposals, start=1):
        print_proposal(i, proposal)

    executable = [p for p in proposals if p.is_executable]
    if not interactive:
        print(f"\n{RULE}")
        print(f"analyze mode: nothing executed. {len(executable)} of {len(proposals)} "
              "proposals are executable; re-run with `run` to approve them.")
        return 0

    if not executable:
        print(f"\n{RULE}\nNo executable changes proposed; nothing to approve.")
        return 0

    metrics_by_name = {m.name: m for m in metrics}

    print(f"\n{RULE}\nAPPROVAL ({len(executable)} executable changes)")
    print("Type `yes` to apply a change, `no` to skip it. Anything else counts as no.")
    print(RULE)

    applied = 0
    for proposal in executable:
        print(f"\n  {proposal.workload}: {proposal.describe()}")
        try:
            answer = input("  apply? [yes/no] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  aborted; remaining proposals left unapproved.")
            break

        if answer != "yes":
            audit.record_decision(proposal.decision_id, "no")
            print("  -> skipped")
            continue

        audit.record_decision(proposal.decision_id, "yes")
        result = backend.apply(
            target=proposal.workload,
            action=proposal.action,
            params=proposal.params,
            metrics=metrics_by_name.get(proposal.workload),
        )
        audit.record_execution(
            proposal.decision_id,
            ok=result.ok,
            result=result.detail,
            error=result.error,
        )
        if result.ok:
            applied += 1
            print(f"  -> applied: {result.detail}")
        else:
            print(f"  -> NOT applied ({result.detail}): {result.error}")

    for proposal in proposals:
        if not proposal.is_executable:
            audit.record_decision(proposal.decision_id, "skipped")

    print(f"\n{RULE}\n{applied} change(s) applied. Full trail: "
          f"rightsize audit")
    return 0
