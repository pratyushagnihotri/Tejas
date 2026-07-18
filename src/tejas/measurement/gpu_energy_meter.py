"""Background energy sampler for Linux/NVIDIA via `nvidia-smi` — used on the
DGX cluster. No root needed (nvidia-smi is user-readable). Same interface as
energy_meter.PowerMeter (start/energy_since/stop), so run_experiment.py can
pick whichever meter matches the platform without any other code changes.

Only polls the GPU(s) actually allocated to this job. On a shared node,
querying every GPU on the box would mix other users' power draw into our
measurements — nvidia-smi does NOT automatically respect CUDA_VISIBLE_DEVICES
the way CUDA applications do (it's a driver-level tool, not a CUDA-runtime
one), so we resolve and pass an explicit -i filter ourselves. If Slurm's
cgroup device restriction already hides other GPUs from us entirely (some
cluster configs do this), nvidia-smi will already only show our allocation —
we detect that case and skip the redundant filter.
"""

import os
import re
import subprocess
import threading
import time

from tejas.measurement.power_meter_base import SampledPowerMeterBase

_LINE_RE = re.compile(r"^\s*(\d+)\s*,\s*([\d.]+)\s*$")


class GPUPowerMeter(SampledPowerMeterBase):
    def __init__(self, interval_ms=200):
        super().__init__(interval_ms)
        self._proc = None
        self._reader_thread = None
        self._gpu_indices = None
        self._latest_per_gpu = {}

    def _resolve_gpu_indices(self):
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES is not set — refusing to guess which "
                "GPU(s) belong to this job on a shared node. Run inside a "
                "Slurm allocation (srun/sbatch) that requests a GPU."
            )
        requested = [x.strip() for x in cvd.split(",") if x.strip() != ""]

        visible = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()
        visible = [v.strip() for v in visible if v.strip()]

        if len(visible) == len(requested):
            # nvidia-smi is already scoped to just our allocation (cgroup
            # device restriction active) — use every index it shows us.
            return visible
        # nvidia-smi sees the whole node — explicitly filter to our GPU(s)
        # so we never attribute another user's power draw to ourselves.
        return requested

    def start(self):
        self._gpu_indices = self._resolve_gpu_indices()
        self._latest_per_gpu = {idx: None for idx in self._gpu_indices}

        cmd = [
            "nvidia-smi",
            "-i", ",".join(self._gpu_indices),
            "--query-gpu=index,power.draw",
            "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        time.sleep(1.0)
        if not self.samples:
            raise RuntimeError(
                f"No power samples received from nvidia-smi for GPU(s) "
                f"{self._gpu_indices}. Check CUDA_VISIBLE_DEVICES and that "
                "nvidia-smi is on PATH inside the allocation."
            )

    def _read_loop(self):
        for line in self._proc.stdout:
            m = _LINE_RE.match(line)
            if not m:
                continue
            idx, watts = m.group(1), float(m.group(2))
            with self._lock:
                self._latest_per_gpu[idx] = watts
                if all(v is not None for v in self._latest_per_gpu.values()):
                    total = sum(self._latest_per_gpu.values())
                    self.samples.append((time.time(), total))
            if self._stop:
                break

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
