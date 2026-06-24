# FDE Integration

Discover local FDE in this order:

1. Explicit `--fde-root`
2. `FDE_SOURCE_PATH`
3. The current working directory and parents
4. Sibling `FDE` or `benchmark` directories

Accept either a full FDE checkout with `packages/kernels`, or the benchmark snapshot with `vendor/fde_packages` and `contestants/2_scoPe_regressor`.

Required behavior:

- Fail if TabPFN3 or its local weights are unavailable.
- Do not silently switch to XGBoost as the prediction model.
- XGBoost is allowed only for feature screening and Top-32 ranking.
- Prefer FDE's existing trend/frequency window feature builders when importable.

Local FDE runtime checks:

- Prefer the FDE-managed Python when the system Python cannot import `tabpfn`; current FDE checkouts usually provide `.venv/bin/python` and require Python `>=3.11,<3.12`.
- For local TabPFN3 weights, ensure `FDE_TPT_WEIGHTS_DIR` points at the parent directory containing `tabpfn3/`, for example `FDE/packages/kernels/kernels/weights`, not at the `tabpfn3` directory itself.
- For FDE window features from a full checkout, make `algorithms/2_scoPe_regressor` importable as well as `packages/kernels` and `packages/icl_utils`; otherwise the feature builder can fail with `No module named 'pipeline'`.
- If the FDE venv lacks the report-only `plotly` dependency but another local environment has it, append that environment's `site-packages` to `sys.path` after the FDE venv paths. Do not prepend a whole foreign `site-packages`, because it can override compiled dependencies such as `numpy` or `pandas`.
- On Apple Silicon, run MPS smoke tests and TabPFN3 jobs with escalated permissions when the sandbox reports `torch.backends.mps.is_available() == False`; this can be a sandbox visibility issue rather than a hardware or PyTorch issue.

Outputs are offline fitability validation artifacts, not online deployment backtests.
