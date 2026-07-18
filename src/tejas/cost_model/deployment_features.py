"""Grounded, hand-authored deployment metadata — not learned from data.

measurement_scope is the load-bearing field here: verified directly against
energy_meter.py (Mac) and gpu_energy_meter.py (DGX), not assumed.
- Mac's `powermetrics` reports "Combined Power (CPU + GPU + ANE)" — whole-SoC
  package power. Apple Silicon has no separate host/accelerator boundary, so
  there's no narrower reading available.
- DGX's `nvidia-smi --query-gpu=power.draw`, scoped to just the job's
  allocated GPU(s) via CUDA_VISIBLE_DEVICES (gpu_energy_meter.py's
  _resolve_gpu_indices) — GPU silicon only, host CPU/RAM explicitly excluded.

These are different physical measurement boundaries, not just different
scales — energy_model.py refuses to pool absolute-J targets across rows with
different measurement_scope (see ASSERT_NO_CROSS_SCOPE_POOLING there).
"""

import re

DEPLOYMENTS = {
    "mac": {
        "platform": "mac",
        "device_class": "apple_silicon",
        "model_family": "qwen",
        "gpu_count": 0,
        "measurement_scope": "package_cpu_gpu_ane",
    },
    "dgx_qwen": {
        "platform": "dgx",
        "device_class": "nvidia_a100",
        "model_family": "qwen",
        "gpu_count": 1,
        "measurement_scope": "gpu_die_only",
    },
    "dgx_llama": {
        "platform": "dgx",
        "device_class": "nvidia_a100",
        "model_family": "llama",
        "gpu_count": 2,
        "measurement_scope": "gpu_die_only",
    },
}

# Maps raw_results*.csv filenames to a DEPLOYMENTS key, so callers can go
# straight from --input path to deployment metadata.
CSV_TO_DEPLOYMENT = {
    "raw_results.csv": "mac",
    "raw_results_dgx_qwen.csv": "dgx_qwen",
    "raw_results_dgx_llama.csv": "dgx_llama",
}

_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.I)


def param_count_b(model_name):
    """qwen2.5:32b-instruct -> 32.0, llama3.3:70b -> 70.0."""
    m = _PARAM_COUNT_RE.search(model_name)
    if not m:
        raise ValueError(f"couldn't parse a parameter count from model_name={model_name!r}")
    return float(m.group(1))


def deployment_for_csv(csv_path):
    import os
    basename = os.path.basename(csv_path)
    if basename not in CSV_TO_DEPLOYMENT:
        raise ValueError(f"no known deployment for {basename} — add it to CSV_TO_DEPLOYMENT")
    return CSV_TO_DEPLOYMENT[basename]


def _self_check():
    cases = {
        "qwen2.5:3b-instruct": 3.0, "qwen2.5:7b-instruct": 7.0,
        "qwen2.5:14b-instruct": 14.0, "qwen2.5:32b-instruct": 32.0,
        "llama3.2:3b": 3.0, "llama3.1:8b": 8.0, "llama3.3:70b": 70.0,
    }
    for name, expected in cases.items():
        got = param_count_b(name)
        assert got == expected, f"{name}: expected {expected}, got {got}"
    for key, meta in DEPLOYMENTS.items():
        assert meta["measurement_scope"] in ("package_cpu_gpu_ane", "gpu_die_only")
        if key == "mac":
            assert meta["gpu_count"] == 0
    print(f"param_count_b OK for {len(cases)} model names")
    print(f"{len(DEPLOYMENTS)} deployments, measurement_scope grounded per energy_meter.py / gpu_energy_meter.py")


if __name__ == "__main__":
    _self_check()
