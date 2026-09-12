# Postmortem: payment-service OOM during settlement (2026-03-14)

Severity: SEV-1
Duration: 47 minutes
Impact: 3,182 transactions failed to settle; manual reconciliation required

## Summary

An automated rightsizing job reduced the `payment-service` memory limit from
1 GiB to 256 MiB. The job based its recommendation on mean working-set usage
over a 30-minute window, which was 61 MiB. At 14:00 UTC the hourly settlement
pass allocated its reconciliation set, peaked at **640 MiB**, and was
OOM-killed by the kernel mid-write.

## What went wrong

1. The recommendation used a **mean over a window that contained no settlement
   pass**. The workload's entire memory demand is periodic and the window
   missed it.
2. The change was applied without runbook review. The runbook already
   documented the settlement pass and its memory profile.
3. The resulting crash loop was initially read as a further sign of
   over-provisioning, and a second reduction was queued before a human
   intervened.

## Actions taken

- Memory floor of 768 MiB written into the `payment-service` runbook as a
  binding constraint.
- Rightsizing automation must now cite the relevant runbook section in every
  proposal, and must observe a window covering at least one full period of any
  documented periodic job.
- Restart count above two in the observation window now disqualifies a
  container from automated rightsizing.
