from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ColumnContract:
    time_column: str
    target_column: str
    feature_columns: list[str]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported data format: {suffix}; expected .csv or .parquet")


def infer_columns(df: pd.DataFrame, target_column: str) -> ColumnContract:
    if target_column not in df.columns:
        raise ValueError(f"target column not found: {target_column}")

    time_column = _infer_time_column(df)
    feature_columns = [
        str(col)
        for col in df.select_dtypes(include="number").columns
        if str(col) != time_column
    ]
    if not feature_columns:
        raise ValueError("could not infer numeric feature columns; please specify them")
    return ColumnContract(time_column=time_column, target_column=target_column, feature_columns=feature_columns)


def _infer_time_column(df: pd.DataFrame) -> str:
    if len(df.columns) == 0:
        raise ValueError("dataset has no columns")

    first = str(df.columns[0])
    if _can_parse_time(df[first]):
        return first

    matches = [str(col) for col in df.columns if _looks_like_time(str(col))]
    for col in matches:
        if _can_parse_time(df[col]):
            return col
    raise ValueError("could not infer time column; please specify one")


def _looks_like_time(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("time", "timestamp", "date"))


def _can_parse_time(series: pd.Series) -> bool:
    return pd.to_datetime(series, errors="coerce").notna().any()
