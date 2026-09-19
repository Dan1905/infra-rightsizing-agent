# Infra Rightsizing Agent

An LLM agent that finds over-provisioned containers and Kubernetes workloads,
grounds every recommendation in your own policies, runbooks and incident
postmortems, and changes nothing until **a human types `yes`**.

"Flag anything using less than 10% of its limit" is easy to write and wrong in
the ways that matter. This agent is built around three properties a threshold
rule doesn't have:

- **It weighs signals against each other.** Peak-to-mean ratio, restarts,
  observation-window length and workload type decide whether a low average
  means "over-provisioned" or "you're looking at the wrong hour".
- **It is grounded in your documents.** Every decision cites a passage it
  actually retrieved. If a runbook sets a floor, the agent is bound by it and
  names the document.
- **It cannot act on its own.** A proposal is a row in an audit log until a
  person approves that specific change, and hard guardrails run again before
  anything is applied.

In the bundled sandbox, all three workloads look over-provisioned on a one-hour
window. The agent resizes one, flags one for review because it is crash-looping
and its runbook forbids changes, and leaves the third alone because its real
work happens at 02:00 and the window missed it.

## How it works

```
 Prometheus ─── usage over time ───┐          policies/*.md
 Docker API / Kubernetes API ──────┤                │ chunked by heading,
   (limits, requests, replicas,    │                │ embedded locally
    labels, restarts)              ▼                ▼
                        workload metrics ──► seed retrieval (Chroma)
                                   │                │
                                   └───────┬────────┘
                                           ▼
                      LLM tool-calling loop (Groq gpt-oss-120b by default)
                        ├─ search_policies  pull more context on demand
                        └─ propose_change   one decision per workload,
                                            validated + guardrail-checked
                                           │
                                           ▼
                            plan with reasoning and citations
                                           │
                                     typed yes / no
                                           │
                                           ▼
                      backend applies it, after re-checking guardrails
                     (Docker SDK, or a Kubernetes patch with a dry run first)
                                           │
                                           ▼
                              SQLite audit trail (data/audit.db)
```

## Quick start (Docker sandbox)

Needs Python 3.11+ (3.13 recommended), Docker, and a free
[Groq API key](https://console.groq.com/keys).

```bash
make install                    # .venv + package + test tools
cp .env.example .env            # then paste your GROQ_API_KEY into it
make up                         # cAdvisor, Prometheus, three sandbox workloads
make index                      # embed the policy corpus (first run downloads ~90 MB)
```

Give the workloads 15–20 minutes to build up history, then:

```bash
make metrics                    # what the agent sees
make analyze                    # propose a plan; never executes
make run                        # propose, approve each change, execute
make audit                      # read the trail
```

`make help` lists every target. The installed command is `rightsize`
(`.venv/bin/rightsize --help`), with options such as `--lookback 6h`,
`--provider`, `--model`, `--top-k`, and `rightsize audit --show-context` to see
the passages behind each decision.

## Kubernetes

The same agent runs against Deployments in a local minikube cluster. Only the
backend changes; retrieval, reasoning, validation, approval and auditing are
shared.

```bash
brew install minikube helm
make k8s-up                     # cluster, kube-prometheus-stack, workloads
make port-forward               # in its own terminal; Prometheus on :9091
make analyze BACKEND=kubernetes
```

Or set `BACKEND=kubernetes` in `.env`.

| | Docker | Kubernetes |
|---|---|---|
| Unit of change | container | Deployment (pods are rolled) |
| Actions | `set_memory_limit`, `set_cpu_limit`, `stop_container` | `set_requests`, `set_limits`, `scale_replicas` |
| Cost lever | limits | **requests**, since the scheduler packs nodes by them |
| Restart count | since the container was created | within the observation window |
| Crash reason | — | last termination reason, e.g. `OOMKilled` |
| Extra guardrails | — | namespace scope, server-side dry run before every patch, requests ≤ limits, no manual scaling of HPA-managed Deployments, single-container pods only |

Configuration (requests, limits, replicas, labels, autoscalers) comes from the
Kubernetes API, which holds the same objects the backend patches. Usage comes
from Prometheus. Per-pod usage is summarised per Deployment from its busiest
replica, since requests and limits are set per pod.

## Configuration

Everything is set in `.env`; see `.env.example` for the full list.

**Model provider.** The loop is provider-agnostic; only the tool-call wire
format differs.

| `LLM_PROVIDER` | Endpoint | Default model | Key |
|---|---|---|---|
| `groq` *(default)* | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| `anthropic` | Anthropic Messages API | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| `openai` | `LLM_BASE_URL`: any OpenAI-compatible API (xAI, Together, OpenRouter, vLLM) | `LLM_MODEL` | `LLM_API_KEY` |

**Groq's free tier** allows 8K tokens per request (prompt *plus* `max_tokens`)
and 200K per day. The defaults are tuned to fit: `max_tokens` 3072,
`reasoning_effort=low` for gpt-oss (its reasoning counts against the output
budget and otherwise truncates tool calls), the model's reasoning is not sent
back on later turns, and retrieved passages are deduplicated across workloads.
If you hit a limit, the error says whether it was the per-minute or the daily
quota.

## Safety model

- **Scope is structural.** Only workloads labelled `cost-opt.managed=true` (and,
  on Kubernetes, only in the `cost-opt-sandbox` namespace) can be changed. The
  monitoring stack is out of reach.
- **Nothing executes without typed approval**, one change at a time. There is
  no auto-approve flag. `analyze` is the non-interactive mode, and it cannot
  execute.
- **Guardrails don't depend on the model.** They are in
  `src/rightsizer/backends/guardrails.py` and include hard memory and CPU
  floors and a refusal to go below 1.2× the observed peak. A retrieved document
  can make the agent more conservative, never less.
- **Guardrails run twice.** At proposal time (preflight, against the observed
  state) a blocked change goes back to the model to correct before any human
  sees it. At execution they run again against live state.
- **Workloads can declare the history they need.** A label such as
  `cost-opt.min-window: 24h` makes any change on a shorter window impossible,
  whatever the model concluded. This is the lesson of the sandbox's March
  postmortem, moved from a document into enforcement.
- **Everything is recorded:** the retrievals, the proposal and its citations,
  the human decision, and the execution result.

## Project layout

```
src/rightsizer/
├── cli.py                 `rightsize` command: argument parsing only
├── pipeline.py            one run: collect → reason → approve → execute → audit
├── config.py              settings from the environment / .env
├── audit.py               SQLite audit trail
├── reporting.py           metrics table for people, labelled blocks for the model
├── agent/
│   ├── loop.py            the tool-calling loop and seed retrieval
│   ├── prompts.py         system prompt, retrieval queries, user message
│   ├── tools.py           tool schemas and per-backend action specs
│   └── proposals.py       Proposal, validation, what each tool call does
├── llm/providers.py       Groq / Anthropic / OpenAI-compatible
├── retrieval/store.py     markdown chunking, local embeddings, Chroma
└── backends/
    ├── base.py            WorkloadMetrics and the Backend contract
    ├── guardrails.py      every hard safety check
    ├── prometheus.py      query client
    ├── docker.py          cAdvisor metrics + Docker SDK
    └── kubernetes.py      Kubernetes API + Prometheus, dry-run patches
policies/                  the retrieval corpus: replace with your own
sandbox/
├── workloads/             the three load generators
├── docker/                compose stack: cAdvisor, Prometheus, workloads
└── kubernetes/            manifests, monitoring values, up.sh
tests/                     unit tests; no Docker, cluster, network or model needed
```

## Development

```bash
make test
```

The suite covers the guardrails, both backends against fake API clients,
proposal validation, the reasoning loop driven by a scripted model, chunking,
the provider helpers and report rendering.

## The sandbox

Three workloads, each a different way for a low average to mislead:

| Workload | Load shape | What the policy corpus says |
|---|---|---|
| `web-frontend` | steady ~8% of a core, tiny flat memory | genuinely oversized, tier 2, safe to resize |
| `payment-service` | idle, then hard bursts; crashes intermittently | tier 1, hard 768 MiB floor, restarts are a gateway fault |
| `batch-worker` | near-idle all day | real job runs at 02:00 UTC and peaks at ~1.2 GiB; needs 24h of history |

The workloads and the policy documents are synthetic, written so that metrics
and documentation disagree in realistic ways. The infrastructure, metrics
pipeline, agent and execution paths are real.

**cAdvisor on Docker Desktop.** The compose file deliberately doesn't mount
`/var/run/docker.sock` into cAdvisor or pass `--docker_only`. cAdvisor's Docker
handler expects a storage layout that Docker Desktop 29.x doesn't use, so with
the socket mounted it emits no per-container series at all. Without it, the raw
cgroup factory reports the same containers by cgroup id, and
`backends/docker.py` maps those ids back to names through the Docker API. On a
normal Linux host the series carry names directly, and the same code handles
both.
