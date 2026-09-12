# Change Management for Automated Remediation

Last reviewed: 2026-06-11

## Approval

Every proposed change produced by automation is a **proposal only**. No change
is applied without a human explicitly approving that specific proposal. Blanket
or standing approvals are not granted for resource changes.

## What automation is permitted to propose

- Reducing a container's memory limit, within the sizing policy's headroom
  rules and any binding runbook floor.
- Reducing a container's CPU limit, within the same rules.
- Stopping a container that is confirmed to be abandoned (no owner, no traffic,
  no scheduled job) -- and only for tier 3 workloads.

## What automation must never propose

- Any change to a tier 1 service that its runbook forbids.
- Any change to observability or platform infrastructure (Prometheus, cAdvisor,
  log shippers, the agent's own containers).
- Deleting data, images, or volumes.

## Citation requirement

Every proposal must name the policy, runbook, or postmortem section that
supports it. A proposal that cannot cite a supporting document is not
actionable and must be surfaced as a question to the owning team instead of as
a change.

## Audit

Proposals, the retrieved context they were grounded in, the human decision,
and the executed result are all retained for 400 days.
