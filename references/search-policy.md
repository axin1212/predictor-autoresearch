# Search Policy

The search is rule-driven AutoResearch for quick local future-prediction validation.

Core rules:

- Use three target-label-based holdout intervals when enough labels exist.
- Degrade to two or one holdout when label count is limited; fail below eight non-null target labels.
- Align each sample as `(features at t, target at t+horizon_step)` and drop pairs with missing target or invalid timestamps.
- Treat horizon `0` as mandatory in every run. If the user supplies only future horizons, prepend `t+0` before running search so the report has a current-point reference.
- Exclude training pairs when either the anchor time or the future target time falls inside the active holdout interval.
- Keep model input fairness by selecting exactly Top-32 features for every candidate when enough columns exist.
- Include raw identity, recent/coverage context, FDE trend/window, and optional frequency features in the same Top-32 competition.
- Keep the target column in candidate features for horizons greater than zero unless the user explicitly requests otherwise; current/historical target values are valid autoregressive predictors for future targets.
- For `t+0`, exclude the target column from model inputs. Otherwise the model can learn an identity mapping from the current target to itself and the horizon reference is invalid.
- Treat ICL context sampling as a first-class search dimension. After the uniform identity baseline, probe identity features with recent and coverage context before adding trend/window features.
- Keep ICL context sample count fixed by `--num-train-samples` for the whole run; the sampler caps to available labels when fewer are available.
- Run baseline on all holdouts, then use the worst baseline holdout as quick-screen for low-risk candidates.
- Do not early stop. Run until the requested time budget is essentially exhausted.
- If two rounds do not improve clearly, backtrack to a prior high-value node and explore another path.
- If all low-risk candidates have strongly negative RMSE improvement or worst-holdout R-squared, audit preprocessing first: leakage columns, synchronized duplicate tags, missing-value handling, natural sampling level, downsampling/aggregation, and holdout target distribution shift.

Score candidates by direct mean RMSE improvement percentage across completed holdout runs:

```text
mean(completed_holdout_rmse_improvement_pct_values)
```

The improvement is computed against a train-mean constant baseline:

```text
(naive_rmse - model_rmse) / naive_rmse * 100
```

Do not subtract stability, floor, or missing-window penalties from the ranking score. Keep per-holdout R-squared, RMSE, and naive RMSE visible in the report so robustness remains inspectable without making the score opaque.

HTML report:

- Sort by mean RMSE improvement percentage.
- Show best RMSE improvement percentage and best mean R² by prediction horizon, including `t+0`.
- Show actual values on x-axis and predictions on y-axis.
- Draw a 45-degree reference line in every subplot.
- Display each subplot's prediction horizon and RMSE improvement in the title.
- Include failures with reason text in the candidate index.
