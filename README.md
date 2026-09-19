# Cloud Infra Cost-Optimization Agent

An agent that watches container resource metrics, retrieves the organisation's
own policies, runbooks and incident postmortems, reasons over both with Claude,
and proposes remediation that **a human approves before anything executes**.

The point is not "flag containers using less than X%". A threshold rule can do
that, and it is wrong in the ways that matter. This agent is built around three
properties a rule engine does not have:

- **It weighs signals against each other.** Peak-to-mean ratio, restart counts,
  observation-window length and workload type all bear on whether a low average
  means "over-provisioned" or "you are looking at the wrong hour".
- **It is grounded in your documents.** Every decision must cite a retrieved
  passage. If a runbook sets a memory floor, the agent is bound by it and says
  which document bound it.
- **It cannot act on its own.** Proposals go into an audit log with the context
  that produced them, and stop there until someone types `yes`.

## How it works

```
Prometheus + cAdvisor          policies/*.md
        │                            │
        │ CPU/mem/limits             │ chunked by heading
        │ restarts (Docker API)      │ embedded locally (sentence-transformers)
        ▼                            ▼
   metrics summary  ──────►  seed retrieval (chromadb, top-k per container)
                                     │
                                     ▼
              LLM tool-calling loop (Groq / gpt-oss-120b by default)
                     ├─ search_policies   → pulls more context on demand
                     └─ propose_change    → one structured decision per container
                                     │
                                     ▼
                        proposals + citations printed
                                     │
                              typed yes / no
                                     │
                                     ▼
                      Docker SDK executes, behind guardrails
                                     │
                                     ▼
                       SQLite audit trail (data/audit.db)
```

`propose_change` never executes anything. It writes a row and returns
`pending_human_approval`. Execution happens after the loop ends, in
the backend's `apply()` (`agent/backends/`), and only for proposals a human approved.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...           # free tier: console.groq.com
```

### Choosing a model provider

The reasoning loop is provider-agnostic (`agent/providers.py`); only the tool-call
wire format differs between vendors.

| `LLM_PROVIDER` | Endpoint | Default model | Key |
|---|---|---|---|
| `groq` *(default)* | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| `anthropic` | Anthropic Messages API | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| `openai` | `LLM_BASE_URL` (any OpenAI-compatible: xAI, Together, OpenRouter, local vLLM) | `LLM_MODEL` | `LLM_API_KEY` |

Override per run with `--provider` / `--model`. See `.env.example`.

**On the free tier**, Groq's limits are per-minute token budgets (roughly 8K TPM
for `openai/gpt-oss-120b`), and this agent sends the metrics plus retrieved
passages on every turn. If you hit a rate limit, lower `RAG_TOP_K`, shorten
`--lookback`, or wait a minute — the error message says which knobs to reach for.
Retrieved passages are pooled and deduplicated across containers for the same
reason.

**On smaller open-weight models**, expect the loop to lean on its validation more than a
frontier model does. `propose_change` rejects proposals that target unmanaged
containers, omit citations, cite documents that were never retrieved, or leave
required parameters out — each rejection goes back as a tool error for the model
to correct. That validation is in `agent/tools.py` and runs regardless of which
provider you point at it.

Bring up the sandbox — cAdvisor, Prometheus, and three deliberately
over-provisioned workloads with different load shapes:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Build the policy index (first run downloads the ~90 MB embedding model):

```bash
python -m agent.main index
```

Give the workloads 15–20 minutes to accumulate history, then:

```bash
python -m agent.main metrics    # what Prometheus currently reports
python -m agent.main analyze    # propose a plan; never prompts, never executes
python -m agent.main run        # propose → approve → execute
python -m agent.main audit      # read the trail
```

Useful flags: `--lookback 30m`, `--provider`, `--model`, `--top-k`, and
`python -m agent.main audit --show-context` to see the passages behind each
decision.

## Kubernetes backend

The same agent runs against Deployments in a local minikube cluster. Only the
backend changes; retrieval, reasoning, validation, approval and audit are
shared.

```bash
brew install minikube helm
./scripts/k8s-up.sh                    # cluster, kube-prometheus-stack, workloads
kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9091:9090
python -m agent.main --backend kubernetes metrics
python -m agent.main --backend kubernetes analyze
```

What Kubernetes adds:

| | Docker | Kubernetes |
|---|---|---|
| Unit of change | container | Deployment (pods are rolled) |
| Actions | `set_memory_limit`, `set_cpu_limit`, `stop_container` | `set_requests`, `set_limits`, `scale_replicas` |
| Cost lever | limits | **requests** — the scheduler packs nodes by them |
| Restart count | since the container was created | `increase()` within the window |
| Crash reason | — | last termination reason, e.g. `OOMKilled` |
| Extra guardrails | label | namespace allowlist, server-side dry run before every patch, requests ≤ limits, no manual scaling of HPA-managed Deployments, single-container pods only |

Configuration (requests, limits, replicas, labels, autoscalers) is read from
the Kubernetes API — the same objects the backend patches. Usage comes from
Prometheus. Per-pod usage is summarised per Deployment from its busiest replica,
since requests and limits are set per pod.

## The sandbox workloads

Three containers, each a different way for "low average utilisation" to be
misleading:

| Container | Shape | Limits | What the corpus says |
|---|---|---|---|
| `web-frontend` | steady ~8% of a core, flat ~30 MiB | 1.5 cores / 1 GiB | genuinely oversized, tier 2, safe to resize |
| `payment-service` | idle then hard bursts, intermittent crash | 2 cores / 1 GiB | tier 1, hard 768 MiB floor, restarts are a gateway fault |
| `batch-worker` | near-idle all day | 1 core / 2 GiB | real job runs at 02:00 UTC and peaks at ~1.2 GiB |

A one-hour window makes all three look over-provisioned. Only one of them is.

## Layout

| Path | What it is |
|---|---|
| `agent/backends/base.py` | shared metrics model, guardrail floors, backend contract |
| `agent/backends/docker.py` | cAdvisor metrics + Docker SDK execution |
| `agent/backends/kubernetes.py` | Prometheus + Kubernetes API, patches with server-side dry run |
| `agent/rag.py` | heading-aware markdown chunking, embeddings, Chroma store |
| `agent/llm.py` | seed retrieval, system prompt, the tool-calling loop |
| `agent/providers.py` | Groq / Anthropic / OpenAI-compatible backends |
| `agent/tools.py` | the two tool schemas and their dispatcher + validation |
| `agent/audit.py` | SQLite: runs, retrievals, decisions, executions |
| `policies/` | the RAG corpus — replace with your own |
| `workloads/` | the load generators behind the three sandbox workloads |
| `k8s/`, `scripts/k8s-up.sh` | minikube manifests, monitoring values, bring-up script |

## Safety

- Only containers labelled `cost-opt.managed=true` can be modified. The
  observability stack is structurally out of reach.
- No auto-approve flag exists. `analyze` is the non-interactive mode and it
  cannot execute.
- Guardrails in `agent/backends/` run *after* approval and are independent of
  the model: a hard 128 MiB / 0.25 core floor, and a refusal to set any memory
  limit below 1.2× the observed peak. A retrieved document can make the agent
  more conservative, never less. This matters more, not less, on a small model.
- `stop_container` requires `--allow-stop` on top of the typed approval.
- Every run stores its retrievals, proposals, citations, the human decision and
  the execution result.

## Notes on the cAdvisor setup

`docker-compose.yml` deliberately does not mount `/var/run/docker.sock` into
cAdvisor and does not pass `--docker_only`. cAdvisor's Docker handler expects
the classic `image/overlayfs/layerdb` storage layout, which Docker Desktop 29.x
does not use; with the socket mounted it claims every container and then fails
to read the read-write layer, emitting no per-container series at all. Without
it, the raw cgroup factory reports the same containers keyed by cgroup id, and
`agent/backends/docker.py` resolves those ids back to names via the Docker API. Series
that do carry a `name` label (a normal Linux host) are used directly, so the
same code works either way.
