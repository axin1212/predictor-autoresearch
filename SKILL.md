---
name: predictor-autoresearch
description: Run local offline future-prediction AutoResearch with FDE TabPFN3, configurable horizons such as t+3/t+10/t+30, low-risk identity/trend/window/coverage features, RMSE-improvement candidate ranking versus a train-mean baseline, and an interactive Plotly report. Use when the user explicitly names $predictor-autoresearch or asks to compare future prediction horizons for lag/dead-time analysis.
---

# Predictor AutoResearch

Use this skill only after explicit invocation.

Required user inputs:
- Dataset file path (`.csv` or `.parquet`)
- Target column name

Scope:
- This skill is for future prediction: features available at anchor time `t` predict the target at `t+n`.
- Use it to compare prediction horizons and judge whether a target has actionable lag/dead-time behavior.
- Do not use this skill for same-time soft sensing (`t0`). Use `soft-sensor-autoresearch` for current-point soft sensor tasks.
- The target column is allowed to remain in the feature set, because the current/historical target can be a valid autoregressive input for future prediction.

Before running, check local FDE/model availability. Do not silently fall back to XGBoost if the requested FDE model is unavailable.

Read `references/fde-integration.md` when FDE discovery, TabPFN weights, or environment checks are relevant.
Read `references/search-policy.md` when explaining or modifying the search strategy.

Run:

```bash
python scripts/run_predictor_autoresearch.py <data-file> <target-column>
```

Useful options:
- `--time-budget-minutes <minutes>` controls the search budget; default is 15. Use `0` to remove the time cap and run the full finite candidate list.
- `--num-train-samples <n>` controls the ICL context size; reduce this on memory-limited laptops.
- The search probes context sampling before window features: uniform identity baseline, identity with recent/coverage context, then low-risk trend/window/coverage candidates.
- `--top-features-n <n>` controls how many ranked features enter the model; default is 32.
- `--validation-fraction <fraction>` controls the total target-label fraction held out across robust windows; default is `0.30`.
- `--window-steps <steps>` sets the historical feature window in sample steps; default is `90`, following FDE predictor guidance.
- `--prediction-horizons <list>` sets future steps to search, for example `3,5,8,10,20` or `10:30`. Values are positive integer sample steps.
- `--model-type <tabpfn3|tpt>` selects the FDE model path. Default is `tabpfn3`.
- `--tabpfn-device <cpu|auto|mps|cuda>` controls TabPFN device; default is `auto`, preferring MPS when PyTorch reports it is available. On Apple Silicon, `auto` fails fast when PyTorch is built with MPS but the runtime cannot see Metal devices; pass `cpu` explicitly only when CPU fallback is intended.
- `--tabpfn-fit-mode <mode>` controls TabPFN preprocessing/cache mode; default is `fit_preprocessors` for stable local MPS validation.
- `--tabpfn-n-estimators <n>` controls TabPFN ensemble size; default is `1` for stable local MPS validation. The upstream TabPFN default is larger and may crash on this local FDE/MPS stack.
- `--tpt-device <cpu|auto|mps|cuda>` controls TPT_tab device; default is `mps`.
- `--tpt-fit-mode <mode>` controls TPT_tab preprocessing/cache mode; default is `fit_preprocessors`.
- `--tpt-n-estimators <n>` controls TPT_tab ensemble size; default is `1` for fast laptop validation.
- `--fde-root <path>` points to a local FDE or benchmark checkout.
- `--output-dir <path>` overrides the output directory; default is next to the dataset.
- Resource usage logging is enabled by default and writes `resource_usage.csv` next to `report.html`.
- `--resource-log-interval-seconds <seconds>` controls process-tree CPU/RSS sampling; default is `2.0`.
- `--no-resource-log` disables the default resource log.
- `--include-frequency-candidate` enables the tsfresh/frequency candidate. It is off by default because it can expand to tens of thousands of features and dominate long runs.

Negative-R² triage:
- Treat a strongly negative R² from all low-risk candidates as a diagnostic failure, not as a prompt to add synthetic formula features.
- First audit preprocessing: remove leakage columns and duplicate/synchronized tags, check missing-value handling, confirm the natural sampling level, and compare raw/downsampled aggregations before expanding feature families.
- Read the per-holdout RMSE, MAE, target standard deviation, and RMSE/std in the report. If target variance is tiny, R² can look extreme even when absolute moisture error is small.
- Before expanding the feature search, compare holdout target distributions, batch coverage, and whether one holdout is an out-of-distribution batch/time segment.
- Prefer raw identity, recent/coverage context, trend/rolling, and optionally frequency features.

Model weights:
- `--model-type tabpfn3` uses FDE foundation TabPFN3 regressor weights under `weights/tabpfn3/*regressor*.ckpt`.
- `--model-type tpt` uses FDE `TPTTabRegressor` with `$FDE_TPT_WEIGHTS_DIR/TPT_tab/model.ckpt`.

MPS execution:
- Prefer TabPFN3 on MPS for Apple Silicon runs.
- In Codex, Metal devices may be hidden inside the normal sandbox even when the Mac has a supported Apple GPU. If `--tabpfn-device auto` or `mps` reports MPS unavailable on Apple Silicon, verify with an escalated MPS smoke test and rerun the AutoResearch command with `sandbox_permissions="require_escalated"`.
- Do not silently fall back to CPU for Apple Silicon MPS runs. Use `--tabpfn-device cpu` only when the user explicitly wants CPU.

TPT_tab runs in an isolated child process. This avoids Metal/TPT runtime crashes caused by fitting TPT in the same Python process that just performed FDE feature extraction.

Resource logging records process-tree CPU percent and RSS memory. MPS runs may also append `mps_event` rows with PyTorch MPS allocated/driver memory at fit/predict stages. Apple GPU utilization and power require sudo `powermetrics`, so do not describe MPS memory rows as true GPU utilization percent.

Primary metric:

```text
RMSE improvement (%) = (naive_rmse - model_rmse) / naive_rmse * 100
```

The naive baseline predicts the holdout target using the training-label mean for that holdout. Positive values mean the model beats the constant baseline; negative values mean the model is worse than the constant baseline. R² is still shown in the report as a secondary fit metric.

Report expectations:
- The report starts with the target tag and run parameters: model type, horizons, window size, sampling interval, ICL/train sample count, top features, and validation fraction.
- The top chart shows best RMSE improvement by prediction horizon.
- The candidate table is sorted by mean RMSE improvement percentage.
- Actual-vs-predicted plots include a 45-degree reference line and concise subplot titles to avoid overlap.

Return the final `report.html` and `resource_usage.csv` paths and summarize the best RMSE improvement by horizon.
