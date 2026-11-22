# Grid trace

`de_2025_hourly.csv`: Germany, calendar year 2025, hourly (UTC), averaged from 15-minute data.

- `carbon_g_per_kwh`: CO2-equivalent intensity of electricity generation, Energy-Charts (Fraunhofer ISE),
  API `https://api.energy-charts.info/co2eq?country=de`.
- `price_eur_per_mwh`: day-ahead price, bidding zone DE-LU, Energy-Charts API `/price?bzn=DE-LU`
  (CC BY 4.0, Bundesnetzagentur | SMARD.de).

Raw monthly API responses are kept as `raw_*.json`. Downloaded 2026-11-19.

## cross_engine_qwen7b_q1q5.csv

Subset of the cross-engine benchmark (`greenQueryLab/reports/sembench_measured.csv`, copied 2026-11-21):
six engines run unmodified on SemBench Movie (1,000 reviews), Qwen2.5-7B (4-bit, Ollama), one A100-40GB
per run, queries Q1 and Q5, all repetitions. Energy = `gpu_wh` (NVML, incl. idle) + `cpu_job_wh` (job's
share of RAPL package energy).

## cross_engine_local.csv

All 264 single-model runs of the cross-engine benchmark with the ten local models (copied 2026-11-22 from
`greenQueryLab/reports/sembench_measured.csv`; API and multi-model runs excluded). Correctness uses the
`score` column (F1 for Q1/Q2/Q5, 1 - relative error for Q4; `quality` holds the raw relative error for Q4).

