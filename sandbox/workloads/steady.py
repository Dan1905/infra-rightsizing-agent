"""web-frontend profile: steady, predictable CPU; small flat memory footprint.

Represents a request-serving tier that has been sized off a long-forgotten
capacity estimate. Real usage sits far below the configured limits.
"""
import math
import time

# ~200 KB of long-lived state, plus small per-request churn.
_cache = bytearray(200 * 1024)

def burn(seconds: float) -> None:
    end = time.monotonic() + seconds
    x = 0.0
    while time.monotonic() < end:
        x += math.sqrt(time.monotonic())
    return None

while True:
    # ~8% of one core: 80ms of work per 1s tick.
    burn(0.08)
    _cache[: 64 * 1024] = bytes(64 * 1024)
    time.sleep(0.92)
