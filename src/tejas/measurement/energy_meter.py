"""Background energy sampler for macOS via `powermetrics` (requires sudo).

Start once for the whole experiment; individual requests query
energy_since(start, end) around their own timestamps to get Joules consumed
during that window, integrated from the continuous power-draw samples.
"""

import re
import subprocess
import threading
import time

from tejas.measurement.power_meter_base import SampledPowerMeterBase

_COMBINED_RE = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
_CPU_RE = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
_GPU_RE = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
_ANE_RE = re.compile(r"^ANE Power:\s*([\d.]+)\s*mW", re.MULTILINE)


class PowerMeter(SampledPowerMeterBase):
    def __init__(self, interval_ms=200):
        super().__init__(interval_ms)
        self._proc = None
        self._reader_thread = None

    def start(self):
        self._proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", str(self.interval_ms), "--format", "text"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        time.sleep(1.5)
        if not self.samples:
            raise RuntimeError(
                "No power samples received yet. Run this script with sudo, "
                "e.g. `sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment`, "
                "or run `sudo -v` first so powermetrics doesn't hang on a "
                "password prompt with no TTY."
            )

    def _read_loop(self):
        buf = []
        for line in self._proc.stdout:
            buf.append(line)
            if line.startswith("*** Sampled system activity") and len(buf) > 1:
                self._parse_block("".join(buf[:-1]))
                buf = [line]
            if self._stop:
                break
        if buf:
            self._parse_block("".join(buf))

    def _parse_block(self, block):
        m = _COMBINED_RE.search(block)
        if m:
            watts = float(m.group(1)) / 1000.0
        else:
            cpu, gpu, ane = _CPU_RE.search(block), _GPU_RE.search(block), _ANE_RE.search(block)
            if not (cpu or gpu):
                return
            watts = sum(float(x.group(1)) for x in (cpu, gpu, ane) if x) / 1000.0
        self._append_sample(time.time(), watts)

    def stop(self):
        self._stop = True
        if self._proc:
            subprocess.run(["sudo", "kill", str(self._proc.pid)])
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
