from __future__ import annotations

import numpy as np
import pandas as pd

from predictor_autoresearch.data_contracts import ColumnContract
from predictor_autoresearch.feature_pool import WindowFeatureRequest
from predictor_autoresearch.holdout import HoldoutInterval
from predictor_autoresearch.model_runner import CandidateConfig, run_candidate_holdout


class FakeFdeBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        times = pd.to_datetime(request.target_times)
        values = np.arange(len(times), dtype=float)
        return pd.DataFrame({"f0": values, "f1": values * 2})


class FailingFdeBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        raise AssertionError("identity mode must not call the FDE window builder")


class FakePredictor:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


class RecordingPredictor(FakePredictor):
    fit_shapes: list[tuple[int, int]] = []

    def fit(self, x, y):
        self.fit_shapes.append(x.shape)
        return super().fit(x, y)


def test_run_candidate_holdout_returns_metrics_and_predictions():
    rows = 30
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": [float(i) if i % 2 == 0 else None for i in range(rows)],
            "x": np.arange(rows, dtype=float),
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[20, "timestamp"],
        end_time=frame.loc[24, "timestamp"],
        label_indices=[20, 22, 24],
    )
    config = CandidateConfig(
        candidate_id="c1",
        window_minutes=30,
        context_policy="uniform",
        horizon_step=2,
        num_train_samples=5,
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["x"]),
        holdout,
        config,
        FakeFdeBuilder(),
        predictor_factory=FakePredictor,
    )

    assert result.status == "ok"
    assert result.predictions.shape == (3,)
    assert result.actual.shape == (3,)
    assert result.selected_features
    assert result.r2 <= 1.0
    assert result.horizon_step == 2
    assert np.isfinite(result.naive_rmse)


def test_identity_feature_mode_uses_existing_columns_without_window_builder():
    rows = 30
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": [float(i) if i % 2 == 0 else None for i in range(rows)],
            "x": np.arange(rows, dtype=float),
            "x_lag": np.arange(rows, dtype=float) * 2,
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[20, "timestamp"],
        end_time=frame.loc[24, "timestamp"],
        label_indices=[20, 22, 24],
    )
    config = CandidateConfig(
        candidate_id="identity",
        window_minutes=30,
        context_policy="uniform",
        horizon_step=2,
        num_train_samples=5,
        feature_mode="identity",
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["x", "x_lag"]),
        holdout,
        config,
        FailingFdeBuilder(),
        predictor_factory=FakePredictor,
    )

    assert result.status == "ok"
    assert result.selected_features


def test_t0_identity_mode_excludes_target_feature_to_avoid_self_leakage():
    RecordingPredictor.fit_shapes = []
    rows = 30
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": np.arange(rows, dtype=float),
            "x": np.arange(rows, dtype=float) * 10,
        }
    )
    holdout = HoldoutInterval(
        name="h1",
        start_time=frame.loc[20, "timestamp"],
        end_time=frame.loc[24, "timestamp"],
        label_indices=[20, 21, 22, 23, 24],
    )
    config = CandidateConfig(
        candidate_id="identity_h+0",
        window_minutes=30,
        context_policy="uniform",
        horizon_step=0,
        num_train_samples=5,
        feature_mode="identity",
    )

    result = run_candidate_holdout(
        frame,
        ColumnContract("timestamp", "target", ["target", "x"]),
        holdout,
        config,
        FailingFdeBuilder(),
        predictor_factory=RecordingPredictor,
    )

    assert result.status == "ok"
    assert result.horizon_step == 0
    assert result.actual.shape == (5,)
    assert "target" not in result.selected_features
    assert RecordingPredictor.fit_shapes[-1][1] == 1
