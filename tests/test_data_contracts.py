from __future__ import annotations

import pandas as pd
import pytest

from predictor_autoresearch.data_contracts import infer_columns, load_dataset


def test_infer_columns_prefers_first_timestamp_column(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="min"),
            "target": [1.0, None, 2.0],
            "u1": [10.0, 11.0, 12.0],
            "label": ["a", "b", "c"],
        }
    ).to_csv(path, index=False)

    df = load_dataset(path)
    cols = infer_columns(df, "target")

    assert cols.time_column == "timestamp"
    assert cols.target_column == "target"
    assert cols.feature_columns == ["target", "u1"]


def test_target_must_exist(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "u1": [1.0]}).to_csv(path, index=False)
    df = load_dataset(path)

    with pytest.raises(ValueError, match="target column"):
        infer_columns(df, "missing")


def test_parquet_loading_round_trips(tmp_path):
    path = tmp_path / "data.parquet"
    expected = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2), "target": [1.0, 2.0]})
    expected.to_parquet(path)

    loaded = load_dataset(path)

    assert list(loaded.columns) == ["timestamp", "target"]
