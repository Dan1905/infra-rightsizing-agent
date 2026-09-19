"""batch-worker profile: almost entirely idle between nightly runs.

Wakes up briefly to poll a queue that is empty during business hours. This is
the classic over-provisioned container -- but whether it is safe to shrink
depends on what its nightly job actually needs, which lives in a runbook.
"""
import time

_state = bytearray(4 * 1024 * 1024)

while True:
    _state[:1024] = bytes(1024)
    time.sleep(5)
