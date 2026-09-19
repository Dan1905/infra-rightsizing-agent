# Known Workload Profiles

Maintained by Platform Engineering as the reference for what "normal" looks
like per service. Automation should consult this before interpreting a
utilisation number.

## web-frontend

Stateless HTTP request serving behind the shared load balancer. Load is
genuinely steady: the diurnal peak is roughly 2.5x the trough and there is no
batch component. Working set is flat at ~30 MiB and has not moved in nine
months.

This service was sized during the 2025 migration by copying the limits from
the monolith it replaced. Those limits were never revisited and are known to
be far above what it needs. It is tier 2, stateless, and safe to resize inside
the headroom rules in the sizing policy. On Kubernetes its
requests were copied from the same estimate and are equally oversized; keep
at least 2 replicas for zero-downtime deploys.

## payment-service

See [the payment-service runbook](payment-service-runbook.md). Bursty by
design. Do not size from averages.

## batch-worker

Idle during business hours -- it polls an empty queue every five seconds and
uses under 10 MiB doing so. **Its actual job runs at 02:00 UTC nightly**,
where it loads the day's export set and peaks at approximately **1.2 GiB**
resident for 15-25 minutes.

Any observation window that does not include 02:00-03:00 UTC will show this
container as almost completely idle. That reading is correct and completely
useless for sizing. The 2 GiB limit is deliberate and provides the headroom
the sizing policy requires over the nightly peak.

CPU during the nightly run peaks around 0.8 cores; the 1.0 core limit is
tight but has been adequate.

On Kubernetes its memory request (512 MiB) is deliberately *below* the nightly
peak, with a 2 GiB limit: the job runs at 02:00 UTC on capacity nobody else is
using, so it is scheduled Burstable on purpose. Neither raising the request to
the peak nor lowering it further is wanted.
