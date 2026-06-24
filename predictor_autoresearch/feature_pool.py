from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class WindowFeatureRequest:
    data: pd.DataFrame
    time_column: str
    feature_columns: list[str]
    target_times: np.ndarray
    window_minutes: int
    include_frequency: bool


@dataclass(frozen=True)
class WindowFeatureResult:
    features: pd.DataFrame
    resolved_families: list[str]


class FdeFeatureBuilder(Protocol):
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        ...


def build_window_feature_pool(request: WindowFeatureRequest, fde_modules: FdeFeatureBuilder) -> WindowFeatureResult:
    families = ["trend"]
    if request.include_frequency:
        families.append("frequency")

    frames: list[pd.DataFrame] = []
    for family in families:
        frame = fde_modules.build_feature_matrix(request, family).reset_index(drop=True)
        frame = _prefix_duplicates(frame, family, existing={col for prior in frames for col in prior.columns})
        frames.append(frame)
    return WindowFeatureResult(features=pd.concat(frames, axis=1), resolved_families=families)


def select_top_features_xgboost(
    x: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 32,
    random_state: int = 42,
) -> tuple[list[str], dict[str, float]]:
    if x.empty or k <= 0:
        return [], {}
    clean_x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(clean_x, np.asarray(y, dtype=float))
    raw_scores = model.get_booster().get_score(importance_type="gain")
    gains = {col: float(raw_scores.get(col, 0.0)) for col in clean_x.columns}
    ranked = sorted(clean_x.columns, key=lambda col: (-gains[col], col))
    return list(ranked[: min(k, len(ranked))]), gains


def _prefix_duplicates(frame: pd.DataFrame, family: str, existing: set[str]) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    seen = set(existing)
    for col in frame.columns:
        new_col = str(col)
        if new_col in seen:
            new_col = f"{family}__{new_col}"
        seen.add(new_col)
        renamed[str(col)] = new_col
    return frame.rename(columns=renamed)
