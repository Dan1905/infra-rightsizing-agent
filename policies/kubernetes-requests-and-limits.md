# Kubernetes Requests, Limits and Replicas

Owner: Platform Engineering
Last reviewed: 2026-09-01
Applies to: every Deployment on the shared Kubernetes clusters

## Why requests are the cost lever

The scheduler places pods by their **requests**, not by what they use. A node
is "full" once the requests of its pods add up to its capacity, even if the
pods are idle. Over-requested pods therefore buy nodes that do nothing, and
the cluster autoscaler keeps them running. Limits do not drive node count;
requests do. Rightsizing work on Kubernetes starts with requests.

## Sizing requests

- **Memory request** must be at least `1.3 x` the observed p99 working set per
  pod, over a window that includes the workload's heaviest periodic job. The
  same one-hour-window rule as the general sizing policy applies.
- **CPU request** must be at least `1.5 x` observed p95 for latency-sensitive
  tiers (`frontend`, `payments`, `api`) and at least `1.0 x` p95 for
  asynchronous tiers (`batch`, `etl`).
- A request may never exceed the matching limit.
- Floors: no memory request below **64 MiB**, no CPU request below **50m**.

## When a request change is worthwhile

Requests are multiplied by the replica count, so the bar is lower than for
limits. A proposed reduction must free at least **25%** of the current request
*and* at least **64 MiB** of memory or **0.1 cores** of CPU **per pod**.

## Limits

Limits follow the general Resource Sizing Policy headroom rules. Changing a
limit does not save money on its own; propose limit changes only to keep a
limit consistent with a reduced request, or to fix a limit that is below the
observed peak.

## Replicas

- Replica floors in a service's runbook are binding.
- Stateless tier 2 services keep **at least 2 replicas** so a rolling deploy
  never takes them to zero available pods.
- A Deployment whose replica count is managed by a **HorizontalPodAutoscaler**
  must never be scaled by hand; the autoscaler will undo it and the two will
  fight. Change the HPA's bounds instead, through its owner.

## Tier 1 services

Automation must not lower a tier 1 service's requests or its replica count.
Scheduling headroom for revenue-critical paths is decided by their owning team,
not by utilisation data.
