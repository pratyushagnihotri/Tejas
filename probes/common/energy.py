"""Energy measurement for the probes: NVML GPU energy counter + Linux RAPL.

Same method as GreenQuery (evaluation/measurement/energy_counters.py), reduced
to what the probes need: read cumulative counters at the edges of a window.

* GPU: nvmlDeviceGetTotalEnergyConsumption (mJ since driver load, Volta+).
* CPU package / DRAM: /sys/class/powercap/*rapl*/energy_uj (whole socket, so
  it includes every process on the node; reported separately and labelled).

Unavailable counters (e.g. on a laptop) are reported as None, never as zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

POWERCAP = Path("/sys/class/powercap")


class _Gpu:
    def __init__(self, indices: Optional[List[int]] = None):
        self.handles = {}
        self.error = None
        try:
            import pynvml

            self.nv = pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            for i in indices if indices is not None else range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                self.handles[i] = h
        except Exception as e:  # no driver / no GPU / no pynvml
            self.error = str(e)

    def read_j(self) -> Optional[float]:
        if not self.handles:
            return None
        return sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles.values()) / 1000.0


class _Rapl:
    def __init__(self):
        self.zones = []  # (energy_uj path, max range)
        if POWERCAP.exists():
            for z in sorted(POWERCAP.glob("intel-rapl:*")):
                if z.name.count(":") != 1:
                    continue  # top-level package zones only (dram is a subzone on most CPUs)
                f, m = z / "energy_uj", z / "max_energy_range_uj"
                try:
                    int(f.read_text())
                    self.zones.append((f, int(m.read_text()) if m.exists() else None))
                except Exception:
                    pass

    def read_uj(self) -> Optional[List[int]]:
        if not self.zones:
            return None
        return [int(f.read_text()) for f, _ in self.zones]

    def delta_j(self, a, b) -> Optional[float]:
        if a is None or b is None:
            return None
        tot = 0
        for (x, y, (_, rng)) in zip(a, b, self.zones):
            d = y - x
            if d < 0 and rng:
                d += rng
            tot += d
        return tot / 1e6


@dataclass
class Window:
    seconds: float
    gpu_j: Optional[float]
    cpu_pkg_j: Optional[float]
    extra: Dict = field(default_factory=dict)

    def to_dict(self):
        return {"seconds": self.seconds, "gpu_j": self.gpu_j, "cpu_pkg_j": self.cpu_pkg_j, **self.extra}


class EnergyMeter:
    """with meter.window() as w: ...  -> w.result (Window)"""

    def __init__(self, gpu_indices: Optional[List[int]] = None):
        self.gpu = _Gpu(gpu_indices)
        self.rapl = _Rapl()

    @property
    def available(self) -> Dict[str, bool]:
        return {"gpu": bool(self.gpu.handles), "rapl": bool(self.rapl.zones)}

    def start(self):
        return (time.perf_counter(), self.gpu.read_j(), self.rapl.read_uj())

    def stop(self, s) -> Window:
        t0, g0, r0 = s
        t1, g1, r1 = time.perf_counter(), self.gpu.read_j(), self.rapl.read_uj()
        gpu = None if g0 is None or g1 is None else g1 - g0
        return Window(t1 - t0, gpu, self.rapl.delta_j(r0, r1))

    def idle_power(self, seconds: float = 30.0) -> Dict[str, Optional[float]]:
        """Average idle power (W) with the model loaded; used to report energy above idle."""
        s = self.start()
        time.sleep(seconds)
        w = self.stop(s)
        return {
            "seconds": w.seconds,
            "gpu_w": None if w.gpu_j is None else w.gpu_j / w.seconds,
            "cpu_pkg_w": None if w.cpu_pkg_j is None else w.cpu_pkg_j / w.seconds,
        }
