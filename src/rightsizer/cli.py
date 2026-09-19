"""Command-line entry point, installed as `rightsize`.

    rightsize index      build the policy vector index
    rightsize metrics    show what the metrics source currently reports
    rightsize analyze    propose a plan; never prompts, never executes
    rightsize run        propose -> typed human approval -> execute
    rightsize audit      read back the trail

Global options pick the backend (--backend docker|kubernetes), model provider,
observation window and retrieval depth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import pipeline
from .audit import AuditLog
from .config import Settings, settings as default_settings
from .pipeline import RULE, connect
from .reporting import format_table
from .retrieval.store import PolicyStore


def settings_from_args(args: argparse.Namespace) -> Settings:
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
    if getattr(args, "backend", None):
        overrides["backend"] = args.backend.lower()
        # The Prometheus default follows the backend unless set explicitly.
        if "prometheus_url" not in overrides and not os.environ.get("PROMETHEUS_URL"):
            overrides["prometheus_url"] = (
                "http://localhost:9091" if overrides["backend"] == "kubernetes"
                else "http://localhost:9090"
            )
    if not overrides:
        return default_settings
    from dataclasses import replace

    return replace(default_settings, **overrides)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_index(args: argparse.Namespace) -> int:
    s = settings_from_args(args)
    store = PolicyStore(s)
    print(f"Indexing {s.policies_dir} with {s.embedding_model} ...")
    count = store.index(rebuild=args.rebuild)
    print(f"Indexed {count} chunks into `{s.collection_name}` at {s.chroma_dir}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    s = settings_from_args(args)
    connected = connect(s)
    if connected is None:
        return 2
    backend, metrics = connected
    print(f"Backend: {backend.name}   window: {s.lookback}   source: {s.prometheus_url}\n")
    print(format_table(metrics))
    managed = [m for m in metrics if m.managed]
    print(f"\n{len(managed)} of {len(metrics)} workloads are in scope "
          f"({backend.describe_scope()})")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    return pipeline.run(settings_from_args(args), interactive=False)


def cmd_run(args: argparse.Namespace) -> int:
    return pipeline.run(settings_from_args(args), interactive=True, allow_stop=args.allow_stop)


def cmd_audit(args: argparse.Namespace) -> int:
    s = settings_from_args(args)
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
        prog="rightsize",
        description=(
            "Rightsize containers and Kubernetes workloads with an LLM agent grounded "
            "in your policies. Nothing changes without typed approval."
        ),
    )
    parser.add_argument("--prometheus-url", help="override PROMETHEUS_URL")
    parser.add_argument("--lookback", help="observation window, e.g. 30m, 1h, 6h")
    parser.add_argument("--backend", help="docker | kubernetes")
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
