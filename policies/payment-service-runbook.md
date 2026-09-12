# Runbook: payment-service

Tier: 1 (revenue-critical)
On-call rotation: payments-oncall
Last reviewed: 2026-08-19

## What it does

`payment-service` authorises card transactions inline with checkout and runs
the settlement reconciliation batch at the top of every hour. Steady-state
load is low; the hourly settlement pass is where the real resource demand is.

## Binding constraints

These are not suggestions. Automation must treat them as hard limits.

- **Never scale `payment-service` below 3 replicas.** Two replicas cannot
  absorb the settlement pass while one is being restarted during a deploy.
- **Never reduce the memory limit below 768 MiB.** The settlement pass builds
  the full reconciliation set in memory. See
  [2026-03-14 payments OOM](incidents/2026-03-14-payments-oom.md).
- **Never stop or restart this container between 08:00 and 22:00 UTC.**
  Outside that window, changes still require payments-oncall sign-off.

## Reading its metrics

Average CPU and memory for this service are misleading by construction. The
service is idle for most of each hour and then spikes hard for 20-40 seconds.
Any sizing decision must be made against p99 and max, not mean.

## Restart behaviour

Intermittent exits with `upstream settlement timeout` are a known failure mode
against the downstream settlement gateway. They indicate a gateway problem,
**not** over-provisioning. Do not respond to them with a resource change.
