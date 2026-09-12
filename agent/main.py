"""CLI entry point.

    python -m agent.main index      # build the policy vector index
    python -m agent.main metrics    # show what Prometheus currently reports
    python -m agent.main analyze    # propose only, never prompts, never executes
    python -m agent.main run        # propose -> human approval -> execute
    python -m agent.main audit      # read back the trail
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .actions import Executor
from .audit import AuditLog
from .config import Settings, settings as default_settings
from .llm import run_analysis
from .providers import ProviderError, build_provider
from .tools import TOOL_DEFS
from .metrics import MetricsError, collect, format_table
from .rag import PolicyStore
from .tools import Proposal

RULE = "=" * 78


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {}
    if getattr(args, "prometheus_url", None):
        overrides["prometheus_url"] = args.prometheus_url
    if getattr(args, "lookback", None):
        overrides["lookback"] = args.lookback
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider.lower()
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "top_k", None):
        overrides["top_k"] = args.top_k
    if not overrides:
        return default_settings
    from dataclasses import replace

    return replace(default_settings, **overrides)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_index(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    store = PolicyStore(s)
    print(f"Indexing {s.policies_dir} with {s.embedding_model} ...")
    count = store.index(rebuild=args.rebuild)
    print(f"Indexed {count} chunks into `{s.collection_name}` at {s.chroma_dir}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    metrics = collect(s)
    print(f"Observation window: {s.lookback}   source: {s.prometheus_url}\n")
    print(format_table(metrics))
    managed = [m for m in metrics if m.managed]
    print(f"\n{len(managed)} of {len(metrics)} containers are in scope "
          f"(label {s.managed_label}=true)")
    return 0


def _print_proposal(index: int, proposal: Proposal) -> None:
    print(f"\n{RULE}")
    print(f"[{index}] {proposal.container}: {proposal.describe().upper()}")
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


def _run_pipeline(args: argparse.Namespace, interactive: bool) -> int:
    s = _settings_from_args(args)
    audit = AuditLog(s.audit_db)

    try:
        metrics = collect(s)
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Is the stack up?  docker compose -f docker/docker-compose.yml up -d",
              file=sys.stderr)
        return 2

    managed = [m for m in metrics if m.managed]
    if not managed:
        print(f"No containers carry {s.managed_label}=true; nothing to assess.")
        return 0

    print(f"Observation window: {s.lookback}   provider: {s.provider}\n")
    print(format_table(metrics))

    store = PolicyStore(s)
    if store.count() == 0:
        print("\nerror: policy index is empty. Run `python -m agent.main index` first.",
              file=sys.stderr)
        return 2

    try:
        provider = build_provider(s, TOOL_DEFS)
    except ProviderError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    run_id = audit.start_run(
        model=f"{provider.name}:{provider.model}",
        lookback=s.lookback,
        prometheus_url=s.prometheus_url,
    )
    print(f"\nrun: {run_id}\nReasoning over metrics and retrieved policy context ...\n")

    try:
        proposals, narrative, transcript = run_analysis(
            s, metrics=metrics, store=store, audit=audit, run_id=run_id, provider=provider
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
        _print_proposal(i, proposal)

    executable = [p for p in proposals if p.is_executable]
    if not interactive:
        print(f"\n{RULE}")
        print(f"analyze mode: nothing executed. {len(executable)} of {len(proposals)} "
              "proposals are executable; re-run with `run` to approve them.")
        return 0

    if not executable:
        print(f"\n{RULE}\nNo executable changes proposed; nothing to approve.")
        return 0

    executor = Executor(managed_label=s.managed_label, allow_stop=args.allow_stop)
    metrics_by_name = {m.name: m for m in metrics}

    print(f"\n{RULE}\nAPPROVAL ({len(executable)} executable changes)")
    print("Type `yes` to apply a change, `no` to skip it. Anything else counts as no.")
    print(RULE)

    applied = 0
    for proposal in executable:
        print(f"\n  {proposal.container}: {proposal.describe()}")
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
        result = executor.apply(
            container_name=proposal.container,
            action=proposal.action,
            params=proposal.params,
            metrics=metrics_by_name.get(proposal.container),
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
          f"python -m agent.main audit")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    return _run_pipeline(args, interactive=False)


def cmd_run(args: argparse.Namespace) -> int:
    return _run_pipeline(args, interactive=True)


def cmd_audit(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    audit = AuditLog(s.audit_db)

    if args.show_context:
        rows = audit.recent_decisions(limit=args.limit)
        for row in rows:
            print(f"\n{RULE}\n#{row['id']} {row['created_at']}  {row['container']}  "
                  f"{row['action']}  approval={row['approval']}  executed={row['executed']}")
            print(f"reason: {row['reason']}")
            print(f"cited:  {json.loads(row['policy_cited_json'])}")
            print("retrieved context:")
            for chunk in json.loads(row["context_json"]):
                print(f"  - {chunk['citation']} (distance {chunk.get('distance')})")
        return 0

    print("Recent runs")
    print(f"{'RUN':<32} {'STARTED':<22} {'CONTAINERS':<11} {'PROPOSALS':<10}")
    for row in audit.run_summary(limit=args.limit):
        print(f"{row['run_id']:<32} {row['started_at']:<22} "
              f"{row['container_count']:<11} {row['proposal_count']:<10}")

    print("\nRecent decisions")
    header = (f"{'ID':<5} {'CONTAINER':<17} {'ACTION':<18} {'APPROVAL':<9} "
              f"{'EXEC':<5} {'RESULT':<28}")
    print(header)
    print("-" * len(header))
    for row in audit.recent_decisions(limit=args.limit):
        print(f"{row['id']:<5} {row['container']:<17} {row['action']:<18} "
              f"{str(row['approval'] or '-'):<9} {row['executed']:<5} "
              f"{str(row['exec_result'] or row['error'] or '-')[:28]:<28}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Cloud infra cost-optimization agent (Docker + Prometheus + RAG).",
    )
    parser.add_argument("--prometheus-url", help="override PROMETHEUS_URL")
    parser.add_argument("--lookback", help="observation window, e.g. 30m, 1h, 6h")
    parser.add_argument("--provider", help="groq | anthropic | openai")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--top-k", type=int, help="passages per retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/refresh the policy vector index")
    p_index.add_argument("--rebuild", action="store_true", help="drop and rebuild")
    p_index.set_defaults(func=cmd_index)

    p_metrics = sub.add_parser("metrics", help="show current container metrics")
    p_metrics.set_defaults(func=cmd_metrics)

    p_analyze = sub.add_parser("analyze", help="propose a plan; never executes")
    p_analyze.add_argument("--allow-stop", action="store_true", help=argparse.SUPPRESS)
    p_analyze.set_defaults(func=cmd_analyze)

    p_run = sub.add_parser("run", help="propose, ask for approval, then execute")
    p_run.add_argument(
        "--allow-stop",
        action="store_true",
        help="permit stop_container actions (still requires typed approval)",
    )
    p_run.set_defaults(func=cmd_run)

    p_audit = sub.add_parser("audit", help="read the audit trail")
    p_audit.add_argument("--limit", type=int, default=15)
    p_audit.add_argument(
        "--show-context",
        action="store_true",
        help="include the retrieved passages behind each decision",
    )
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
