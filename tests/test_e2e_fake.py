from __future__ import annotations

import numpy as np
import pandas as pd

from predictor_autoresearch.cli import run_autoresearch
from predictor_autoresearch.feature_pool import WindowFeatureRequest


class FakeFdeBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        target_times = pd.to_datetime(request.target_times)
        minutes = (target_times - target_times.min()).total_seconds() / 60.0
        frame = pd.DataFrame({"time_index": minutes, "bias": 1.0})
        if extraction == "frequency":
            frame["frequency_energy"] = minutes**2
        return frame


class MeanPredictor:
    def fit(self, x, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


def test_run_autoresearch_fake_e2e_generates_report(tmp_path):
    rows = 80
    target = [float(i) if i % 3 == 0 else None for i in range(rows)]
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": target,
            "x": np.linspace(0, 1, rows),
        }
    )
    path = tmp_path / "data.parquet"
    data.to_parquet(path)

    report = run_autoresearch(
        path,
        "target",
        time_budget_minutes=0.01,
        output_dir=tmp_path,
        fde_builder=FakeFdeBuilder(),
        predictor_factory=MeanPredictor,
    )

    assert report.exists()
    resource_log = report.parent / "resource_usage.csv"
    assert resource_log.exists()
    resource_text = resource_log.read_text()
    assert "cpu_percent_sum" in resource_text
    assert "mps_current_allocated_mb" in resource_text
    html = report.read_text()
    assert "Predictor AutoResearch" in html
    assert "R²" in html
