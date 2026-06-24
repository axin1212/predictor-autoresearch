from __future__ import annotations

import numpy as np
import pandas as pd

from predictor_autoresearch.feature_pool import (
    WindowFeatureRequest,
    WindowFeatureResult,
    build_window_feature_pool,
    select_top_features_xgboost,
)


class FakeFdeModules:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        if extraction == "trend":
            return pd.DataFrame({"a": [1.0, 2.0, 3.0], "dup": [0.0, 0.0, 0.0]})
        return pd.DataFrame({"b": [3.0, 2.0, 1.0], "dup": [9.0, 9.0, 9.0]})


def test_build_window_feature_pool_merges_families_and_resolves_duplicates():
    request = WindowFeatureRequest(
        data=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3)}),
        time_column="timestamp",
        feature_columns=["x"],
        target_times=np.array(pd.date_range("2026-01-01", periods=3)),
        window_minutes=30,
        include_frequency=True,
    )

    result = build_window_feature_pool(request, FakeFdeModules())

    assert isinstance(result, WindowFeatureResult)
    assert result.resolved_families == ["trend", "frequency"]
    assert list(result.features.columns) == ["a", "dup", "b", "frequency__dup"]


def test_select_top_features_xgboost_returns_k_features():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "strong": np.arange(30, dtype=float),
            "weak": rng.normal(size=30),
            "noise": rng.normal(size=30),
        }
    )
    y = x["strong"] * 2

    selected, gains = select_top_features_xgboost(x, y, k=2, random_state=0)

    assert len(selected) == 2
    assert selected[0] == "strong"
    assert "strong" in gains
