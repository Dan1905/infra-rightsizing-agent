# Resource Sizing Policy

Owner: Platform Engineering
Last reviewed: 2026-07-02
Applies to: all containerised workloads in the shared compute estate

## Headroom requirements

Container limits are sized against observed peaks, never against averages.
Averaging hides the exact events that limits exist to survive.

- **Memory limit** must be at least `1.5 x` the observed p99 working set over
  a window that includes the workload's heaviest periodic job. For workloads
  with a daily or weekly batch component, a one-hour observation window is
  **not** a valid basis for a memory change.
- **CPU limit** must be at least `2.0 x` the observed p95 CPU for
  latency-sensitive tiers (`frontend`, `payments`, `api`), and at least
  `1.25 x` p95 for asynchronous tiers (`batch`, `etl`).
- No container may be given a memory limit below **128 MiB** or a CPU limit
  below **0.25 cores**, regardless of measured usage.

## When a resize is considered worthwhile

Do not generate change requests for marginal savings. A proposed reduction
must free at least **25%** of the current limit *and* at least **256 MiB** of
memory or **0.5 cores** of CPU. Anything smaller is noise and costs more in
review time than it saves.

## Restart counts are a blocker, not an input

A container with more than **two restarts in the observation window** is
considered unstable. Unstable containers are **out of scope for cost
optimisation entirely** -- their resource usage is not trustworthy, and a
reduction can convert a recoverable crash loop into an unrecoverable one.
Route these to the owning team as a reliability issue instead.

## Scaling down replica counts

Replica reductions are governed per service by the service's own runbook.
Where a runbook specifies a replica floor, that floor is binding and this
policy does not override it.
