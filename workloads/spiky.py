"""payment-service profile: bursty CPU, sawtooth memory, occasional crash.

Idles most of the time, then spikes hard for a few seconds. Memory grows
during a burst and is released after it. Every so often the process exits
non-zero so Docker's restart policy bumps the restart count -- the signal a
naive "low average utilisation" rule would happily ignore.
"""
import math
import os
import random
import time

CRASH_PROB = float(os.environ.get("CRASH_PROB", "0.05"))

def burn(seconds: float) -> None:
    end = time.monotonic() + seconds
    x = 0.0
    while time.monotonic() < end:
        x += math.sqrt(time.monotonic())

ticks = 0
while True:
    ticks += 1
    if random.random() < 0.12:
        # Burst: heavy CPU for a few seconds and a big transient allocation.
        buf = bytearray(90 * 1024 * 1024)
        for _ in range(random.randint(3, 6)):
            burn(0.85)
            buf[: 8 * 1024 * 1024] = bytes(8 * 1024 * 1024)
            time.sleep(0.15)
        del buf
    else:
        burn(0.02)
        time.sleep(1.5)

    # Simulate the intermittent crash documented in the March incident.
    if ticks > 40 and random.random() < CRASH_PROB:
        print("payment-service: upstream settlement timeout, exiting", flush=True)
        os._exit(1)
