from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np
import pandas as pd

from predictor_autoresearch.context_sampling import sample_context_indices
from predictor_autoresearch.data_contracts import ColumnContract
from predictor_autoresearch.feature_pool import (
    FdeFeatureBuilder,
    WindowFeatureRequest,
    build_window_feature_pool,
    select_top_features_xgboost,
)
from predictor_autoresearch.holdout import HoldoutInterval
from predictor_autoresearch.scoring import r2_score_np, rmse_np


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    window_minutes: int
    context_policy: str
    horizon_step: int
    num_train_samples: int = 400
    include_frequency: bool = False
    random_state: int = 42
    top_features_n: int = 32
    feature_mode: str = "trend"


@dataclass(frozen=True)
class HoldoutRunResult:
    candidate_id: str
    holdout_name: str
    status: str
    actual: np.ndarray
    predictions: np.ndarray
    r2: float
    rmse: float
    naive_rmse: float
    rmse_improvement_pct: float
    horizon_step: int
    selected_features: list[str]
    error: str | None = None


PredictorFactory = Callable[[], object]


def run_candidate_holdout(
    df: pd.DataFrame,
    columns: ColumnContract,
    holdout: HoldoutInterval,
    config: CandidateConfig,
    fde_builder: FdeFeatureBuilder,
    predictor_factory: PredictorFactory,
) -> HoldoutRunResult:
    _progress(f"holdout_start candidate={config.candidate_id} holdout={holdout.name} horizon={config.horizon_step}")
    pairs = _aligned_forecast_pairs(df, columns, config.horizon_step)
    train_pool = pairs[
        ~(
            ((pairs["target_time"] >= holdout.start_time) & (pairs["target_time"] <= holdout.end_time))
            | ((pairs["anchor_time"] >= holdout.start_time) & (pairs["anchor_time"] <= holdout.end_time))
        )
    ]
    holdout_set = set(holdout.label_indices)
    test_pool = pairs[pairs["target_index"].isin(holdout_set)]
    if train_pool.empty:
        raise ValueError(f"no horizon-aligned training labels for holdout={holdout.name} horizon={config.horizon_step}")
    if test_pool.empty:
        raise ValueError(f"no horizon-aligned holdout labels for holdout={holdout.name} horizon={config.horizon_step}")
    sampled_positions = sample_context_indices(
        pd.to_datetime(train_pool["target_time"]).reset_index(drop=True),
        holdout,
        policy=config.context_policy,
        n=config.num_train_samples,
        random_state=config.random_state,
    )
    train_pairs = train_pool.reset_index(drop=True).iloc[sampled_positions]
    test_pairs = test_pool.reset_index(drop=True)
    train_rows = df.iloc[train_pairs["anchor_position"].to_numpy(dtype=int)].copy()
    test_rows = df.iloc[test_pairs["anchor_position"].to_numpy(dtype=int)].copy()

    _progress(f"features_train_start candidate={config.candidate_id} holdout={holdout.name} rows={len(train_rows)}")
    train_features = _build_features(df, columns, train_rows, config, fde_builder)
    _progress(f"features_train_end candidate={config.candidate_id} holdout={holdout.name} shape={train_features.shape}")
    _progress(f"features_test_start candidate={config.candidate_id} holdout={holdout.name} rows={len(test_rows)}")
    test_features = _build_features(df, columns, test_rows, config, fde_builder)
    _progress(f"features_test_end candidate={config.candidate_id} holdout={holdout.name} shape={test_features.shape}")
    y_train = pd.Series(train_pairs["target_value"].to_numpy(dtype=float))
    y_test = test_pairs["target_value"].to_numpy(dtype=float)

    _progress(f"feature_select_start candidate={config.candidate_id} holdout={holdout.name} k={config.top_features_n}")
    selected, _ = select_top_features_xgboost(
        train_features,
        y_train,
        k=config.top_features_n,
        random_state=config.random_state,
    )
    if not selected:
        selected = list(train_features.columns[: min(config.top_features_n, len(train_features.columns))])

    x_train_model, x_test_model = _standardize_features(
        train_features[selected].fillna(0.0),
        test_features[selected].fillna(0.0),
    )
    y_train_model, y_center, y_scale = _standardize_target(y_train.to_numpy(dtype=float))

    _progress(f"feature_select_end candidate={config.candidate_id} holdout={holdout.name} selected={len(selected)}")
    _progress(f"predictor_create_start candidate={config.candidate_id} holdout={holdout.name}")
    predictor = predictor_factory()
    _progress(f"predictor_create_end candidate={config.candidate_id} holdout={holdout.name}")
    _progress(f"predictor_fit_start candidate={config.candidate_id} holdout={holdout.name} train_shape={x_train_model.shape}")
    predictor.fit(x_train_model, y_train_model)
    _progress(f"predictor_fit_end candidate={config.candidate_id} holdout={holdout.name}")
    _progress(f"predictor_predict_start candidate={config.candidate_id} holdout={holdout.name} test_shape={x_test_model.shape}")
    raw_predictions = predictor.predict(x_test_model)
    _progress(f"predictor_predict_end candidate={config.candidate_id} holdout={holdout.name}")
    predictions = _prediction_array(raw_predictions) * y_scale + y_center
    rmse = rmse_np(y_test, predictions)
    naive_prediction = np.full_like(y_test, fill_value=float(np.nanmean(y_train)), dtype=float)
    naive_rmse = rmse_np(y_test, naive_prediction)
    improvement = _rmse_improvement_pct(rmse, naive_rmse)

    return HoldoutRunResult(
        candidate_id=config.candidate_id,
        holdout_name=holdout.name,
        status="ok",
        actual=y_test,
        predictions=predictions,
        r2=r2_score_np(y_test, predictions),
        rmse=rmse,
        naive_rmse=naive_rmse,
        rmse_improvement_pct=improvement,
        horizon_step=config.horizon_step,
        selected_features=selected,
    )


def _progress(message: str) -> None:
    if os.environ.get("PREDICTOR_AUTORESEARCH_PROGRESS", os.environ.get("SOFT_SENSOR_PROGRESS", "1")) == "0":
        return
    print(f"[predictor-autoresearch] {message}", flush=True)


def _build_features(
    df: pd.DataFrame,
    columns: ColumnContract,
    target_rows: pd.DataFrame,
    config: CandidateConfig,
    fde_builder: FdeFeatureBuilder,
) -> pd.DataFrame:
    feature_columns = _feature_columns_for_horizon(columns, config)
    if config.feature_mode == "identity":
        return target_rows[feature_columns].reset_index(drop=True)
    if config.feature_mode != "trend":
        raise ValueError(f"unsupported feature_mode: {config.feature_mode}")
    request = WindowFeatureRequest(
        data=df,
        time_column=columns.time_column,
        feature_columns=feature_columns,
        target_times=pd.to_datetime(target_rows[columns.time_column]).to_numpy(),
        window_minutes=config.window_minutes,
        include_frequency=config.include_frequency,
    )
    return build_window_feature_pool(request, fde_builder).features


def _feature_columns_for_horizon(columns: ColumnContract, config: CandidateConfig) -> list[str]:
    if config.horizon_step != 0:
        return columns.feature_columns
    feature_columns = [column for column in columns.feature_columns if column != columns.target_column]
    if not feature_columns:
        raise ValueError("t+0 predictor reference cannot use the target column as an input feature")
    return feature_columns


def _prediction_array(raw_predictions: object) -> np.ndarray:
    if hasattr(raw_predictions, "mean"):
        mean = getattr(raw_predictions, "mean")
        if callable(mean):
            return np.asarray(raw_predictions, dtype=float)
        return np.asarray(mean, dtype=float)
    return np.asarray(raw_predictions, dtype=float)


def _standardize_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_arr = train.to_numpy(dtype=float)
    test_arr = test.to_numpy(dtype=float)
    center = np.nanmean(train_arr, axis=0)
    scale = np.nanstd(train_arr, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    center[~np.isfinite(center)] = 0.0
    return (
        ((train_arr - center) / scale).astype("float32"),
        ((test_arr - center) / scale).astype("float32"),
    )


def _standardize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.nanmean(y))
    scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return ((y - center) / scale).astype("float32"), center, scale


def _aligned_forecast_pairs(
    df: pd.DataFrame,
    columns: ColumnContract,
    horizon_step: int,
) -> pd.DataFrame:
    if horizon_step < 0:
        raise ValueError("horizon_step must be non-negative for predictor AutoResearch")
    work = df.reset_index(drop=False)
    original_index_column = str(work.columns[0])
    target = pd.to_numeric(work[columns.target_column], errors="coerce")
    times = pd.to_datetime(work[columns.time_column], errors="coerce")
    anchor_positions = np.arange(0, max(0, len(work) - horizon_step), dtype=int)
    target_positions = anchor_positions + horizon_step
    mask = (
        target.iloc[target_positions].notna().to_numpy()
        & times.iloc[target_positions].notna().to_numpy()
        & times.iloc[anchor_positions].notna().to_numpy()
    )
    anchor_positions = anchor_positions[mask]
    target_positions = target_positions[mask]
    return pd.DataFrame(
        {
            "anchor_position": anchor_positions,
            "target_position": target_positions,
            "target_index": work.iloc[target_positions][original_index_column].to_numpy(),
            "anchor_time": times.iloc[anchor_positions].to_numpy(),
            "target_time": times.iloc[target_positions].to_numpy(),
            "target_value": target.iloc[target_positions].to_numpy(dtype=float),
        }
    )


def _rmse_improvement_pct(rmse: float, naive_rmse: float) -> float:
    if not np.isfinite(rmse) or not np.isfinite(naive_rmse) or naive_rmse <= 0:
        return float("nan")
    return float((naive_rmse - rmse) / naive_rmse * 100.0)
