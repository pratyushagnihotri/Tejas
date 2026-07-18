"""Shared energy-integration logic for platform-specific power samplers:
energy_meter.PowerMeter (macOS, powermetrics) and
gpu_energy_meter.GPUPowerMeter (Linux/NVIDIA, nvidia-smi). Both append
(timestamp, watts) tuples to self.samples from a background reader thread;
this base class turns that into Joules for an arbitrary [start, end] window.
"""

import threading
import time


class SampledPowerMeterBase:
    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []  # (timestamp, watts)
        self._lock = threading.Lock()
        self._stop = False

    def _append_sample(self, timestamp, watts):
        with self._lock:
            self.samples.append((timestamp, watts))

    def energy_since(self, start_ts, end_ts=None, max_wait_s=1.0):
        """Trapezoidal-integrate watts over [start_ts, end_ts] -> Joules.

        Brackets the window with the nearest sample immediately before
        start_ts and immediately after end_ts (linearly interpolating watts
        at the exact boundary), rather than requiring 2+ samples strictly
        inside the window. Without this, fast requests (< one sampling
        interval) can fall entirely between two samples and report no
        energy at all.

        A sample is only committed to self.samples once the reader thread
        has fully processed it, so the most recent ~1 interval of real time
        is always "pending" — end_ts (usually ~now) is almost always ahead
        of the last committed sample. Poll briefly for the reader thread to
        catch up before giving up, instead of failing immediately.
        """
        end_ts = end_ts or time.time()
        deadline = time.time() + max_wait_s
        while True:
            with self._lock:
                samples = list(self.samples)
            if samples and samples[-1][0] >= end_ts:
                break
            if time.time() >= deadline:
                break
            time.sleep(self.interval_ms / 1000.0 / 2)

        if len(samples) < 2:
            return None
        # samples are appended in chronological order by the reader thread
        if start_ts < samples[0][0] or end_ts > samples[-1][0]:
            return None  # window isn't actually bracketed by real samples, even after waiting

        def watts_at(ts):
            # last sample at/before ts, and first sample at/after ts
            before = None
            after = None
            for t, w in samples:
                if t <= ts:
                    before = (t, w)
                elif after is None:
                    after = (t, w)
                    break
            if before is None:
                return after[1] if after else None
            if after is None:
                return before[1]
            t0, w0 = before
            t1, w1 = after
            if t1 == t0:
                return w0
            frac = (ts - t0) / (t1 - t0)
            return w0 + (w1 - w0) * frac

        pts = [(t, w) for t, w in samples if start_ts < t < end_ts]
        w_start = watts_at(start_ts)
        w_end = watts_at(end_ts)
        if w_start is None or w_end is None:
            return None
        pts = [(start_ts, w_start)] + pts + [(end_ts, w_end)]

        energy = 0.0
        for (t0, w0), (t1, w1) in zip(pts, pts[1:]):
            energy += (w0 + w1) / 2.0 * (t1 - t0)
        return energy
