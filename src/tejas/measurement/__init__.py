"""Pillar 1 -- Measurement: real per-request energy, not estimated.
`PowerMeter` (Mac, powermetrics) and `GPUPowerMeter` (DGX, nvidia-smi) share
`SampledPowerMeterBase`'s trapezoidal-integration interface.
"""
