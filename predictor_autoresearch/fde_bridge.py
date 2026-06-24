from __future__ import annotations

from pathlib import Path
import os
import platform
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd

from predictor_autoresearch.feature_pool import WindowFeatureRequest


def resolve_preferred_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and torch.backends.mps.is_built()
        ):
            raise RuntimeError(
                "TabPFN device auto expected MPS on Apple Silicon, but PyTorch reports "
                "MPS unavailable. In Codex this usually means the command is running "
                "inside the sandbox where Metal devices are hidden. Re-run the "
                "AutoResearch command with escalated permissions, or pass "
                "--tabpfn-device cpu explicitly if CPU fallback is intended."
            )
    except Exception:  # noqa: BLE001
        if "torch" in sys.modules:
            raise
    return "cpu"


def find_fde_root(start: Path, explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("FDE_SOURCE_PATH")
    if env:
        candidates.append(Path(env))
    for parent in [start, *start.parents]:
        candidates.extend([parent, parent / "FDE", parent / "benchmark"])
    candidates.append(start.parent / "FDE")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except FileNotFoundError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_fde_or_benchmark(resolved):
            return resolved
    return None


def add_fde_to_path(root: Path) -> None:
    paths = [
        root,
        root / "packages" / "kernels",
        root / "packages" / "icl_utils",
        root / "vendor" / "fde_packages" / "kernels",
        root / "vendor" / "fde_packages" / "icl_utils",
        root / "contestants" / "2_scoPe_regressor",
    ]
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_tabpfn3_predictor_factory(device: str = "auto", fit_mode: str = "fit_preprocessors", n_estimators: int = 1):
    resolved_device = resolve_preferred_device(device)

    def factory():
        return IsolatedTabPFN3Predictor(device=resolved_device, fit_mode=fit_mode, n_estimators=n_estimators)

    return factory


def load_tpt_predictor_factory(device: str = "mps", fit_mode: str = "fit_preprocessors", n_estimators: int = 1):
    def factory():
        return IsolatedTPTTabPredictor(device=device, fit_mode=fit_mode, n_estimators=n_estimators)

    return factory


class IsolatedTabPFN3Predictor:
    """Run TabPFN3 fit/predict in a clean child process.

    The local FDE stack can hang when TabPFN3 MPS fitting happens after the
    window feature pipeline has imported its runtime dependencies. Isolating
    model execution keeps the feature builder and foundation model runtime
    from sharing one Python process.
    """

    def __init__(
        self,
        *,
        device: str,
        fit_mode: str,
        n_estimators: int,
    ) -> None:
        self.device = device
        self.fit_mode = fit_mode
        self.n_estimators = int(n_estimators)
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x, y) -> "IsolatedTabPFN3Predictor":
        self._x_train = np.asarray(x, dtype=np.float32)
        self._y_train = np.asarray(y, dtype=np.float32)
        return self

    def predict(self, x) -> np.ndarray:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("TabPFN3 predictor has not been fitted")
        x_test = np.asarray(x, dtype=np.float32)
        return _run_tabpfn3_child(
            self._x_train,
            self._y_train,
            x_test,
            device=self.device,
            fit_mode=self.fit_mode,
            n_estimators=self.n_estimators,
        )


class IsolatedTPTTabPredictor:
    """Run TPT_tab fit/predict in a clean child process.

    FDE feature extraction imports enough numerical/runtime libraries that the
    official TPT-Tab runtime can segfault on Apple Metal when fit in the same
    process. Keeping the TPT call in a child process preserves MPS acceleration
    while making the local validation path repeatable.
    """

    def __init__(
        self,
        *,
        device: str,
        fit_mode: str,
        n_estimators: int,
        jitter_scale: float = 1e-4,
    ) -> None:
        self.device = device
        self.fit_mode = fit_mode
        self.n_estimators = int(n_estimators)
        self.jitter_scale = float(jitter_scale)
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x, y) -> "IsolatedTPTTabPredictor":
        x_train = np.asarray(x, dtype=np.float32)
        y_train = np.asarray(y, dtype=np.float32)
        if self.jitter_scale > 0 and x_train.size:
            noise = np.random.default_rng(42).normal(scale=self.jitter_scale, size=x_train.shape)
            x_train = (x_train + noise).astype(np.float32)
        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self, x) -> np.ndarray:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("TPT predictor has not been fitted")
        x_test = np.asarray(x, dtype=np.float32)
        return _run_tpt_child(
            self._x_train,
            self._y_train,
            x_test,
            device=self.device,
            fit_mode=self.fit_mode,
            n_estimators=self.n_estimators,
        )


def _run_tabpfn3_child(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    device: str,
    fit_mode: str,
    n_estimators: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="predictor-tabpfn3-") as tmp:
        tmp_path = Path(tmp)
        train_x_path = tmp_path / "x_train.npy"
        train_y_path = tmp_path / "y_train.npy"
        test_x_path = tmp_path / "x_test.npy"
        pred_path = tmp_path / "pred.npy"
        np.save(train_x_path, x_train.astype(np.float32, copy=False))
        np.save(train_y_path, y_train.astype(np.float32, copy=False))
        np.save(test_x_path, x_test.astype(np.float32, copy=False))

        env = os.environ.copy()
        path_entries = [entry for entry in sys.path if entry]
        existing = env.get("PYTHONPATH")
        if existing:
            path_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path_entries))

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _tabpfn3_child_code(),
                str(train_x_path),
                str(train_y_path),
                str(test_x_path),
                str(pred_path),
                device,
                fit_mode,
                str(n_estimators),
            ],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "TabPFN3 child process failed "
                f"(code={completed.returncode})\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return np.load(pred_path).astype(float)


def _tabpfn3_child_code() -> str:
    return textwrap.dedent(
        """
        from pathlib import Path
        from datetime import datetime, timezone
        import csv
        import os
        import sys
        import time
        import numpy as np
        import torch
        from kernels.foundation.weights import resolve_checkpoint_file
        from tabpfn import TabPFNRegressor

        x_train = np.load(sys.argv[1])
        y_train = np.load(sys.argv[2])
        x_test = np.load(sys.argv[3])
        pred_path = Path(sys.argv[4])
        device = sys.argv[5]
        fit_mode = sys.argv[6]
        n_estimators = int(sys.argv[7])
        resource_log_path = os.environ.get("SOFT_SENSOR_RESOURCE_LOG_PATH")
        resource_start = float(os.environ.get("SOFT_SENSOR_RESOURCE_LOG_START_EPOCH", time.time()))

        def _mps_memory_mb(name):
            try:
                if device != "mps" or not torch.backends.mps.is_available():
                    return ""
                value = getattr(torch.mps, name)()
                return f"{float(value) / 1024 / 1024:.2f}"
            except Exception:
                return ""

        def _log_resource_event(note):
            if not resource_log_path:
                return
            now = time.time()
            row = {
                "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                "elapsed_s": f"{now - resource_start:.3f}",
                "kind": "mps_event",
                "root_pid": str(os.getpid()),
                "pid_count": "1",
                "cpu_percent_sum": "",
                "rss_mb_sum": "",
                "gpu_backend": device,
                "mps_current_allocated_mb": _mps_memory_mb("current_allocated_memory"),
                "mps_driver_allocated_mb": _mps_memory_mb("driver_allocated_memory"),
                "note": note,
            }
            with open(resource_log_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writerow(row)

        model_path = resolve_checkpoint_file("tabpfn3", pattern="*regressor*.ckpt")
        model = TabPFNRegressor(
            device=device,
            model_path=model_path,
            categorical_features_indices=[],
            fit_mode=fit_mode,
            n_estimators=n_estimators,
        )
        _log_resource_event("tabpfn3_fit_start")
        model.fit(x_train, y_train)
        _log_resource_event("tabpfn3_fit_end")
        _log_resource_event("tabpfn3_predict_start")
        result = model.predict(x_test)
        _log_resource_event("tabpfn3_predict_end")
        mean = getattr(result, "mean", None)
        if mean is not None and not callable(mean):
            result = mean
        np.save(pred_path, np.asarray(result, dtype=np.float32))
        """
    )


def _run_tpt_child(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    device: str,
    fit_mode: str,
    n_estimators: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="predictor-tpt-") as tmp:
        tmp_path = Path(tmp)
        train_x_path = tmp_path / "x_train.npy"
        train_y_path = tmp_path / "y_train.npy"
        test_x_path = tmp_path / "x_test.npy"
        pred_path = tmp_path / "pred.npy"
        np.save(train_x_path, x_train.astype(np.float32, copy=False))
        np.save(train_y_path, y_train.astype(np.float32, copy=False))
        np.save(test_x_path, x_test.astype(np.float32, copy=False))

        env = os.environ.copy()
        path_entries = [entry for entry in sys.path if entry]
        existing = env.get("PYTHONPATH")
        if existing:
            path_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path_entries))

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _tpt_child_code(),
                str(train_x_path),
                str(train_y_path),
                str(test_x_path),
                str(pred_path),
                device,
                fit_mode,
                str(n_estimators),
            ],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "TPT child process failed "
                f"(code={completed.returncode})\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return np.load(pred_path).astype(float)


def _tpt_child_code() -> str:
    return textwrap.dedent(
        """
        from pathlib import Path
        from datetime import datetime, timezone
        import csv
        import os
        import sys
        import time
        import numpy as np
        import torch
        import tpt_tab.utils as tpt_utils
        from kernels.predictors.TPT_tab import TPTTabRegressor

        x_train = np.load(sys.argv[1])
        y_train = np.load(sys.argv[2])
        x_test = np.load(sys.argv[3])
        pred_path = Path(sys.argv[4])
        device = sys.argv[5]
        fit_mode = sys.argv[6]
        n_estimators = int(sys.argv[7])
        resource_log_path = os.environ.get("SOFT_SENSOR_RESOURCE_LOG_PATH")
        resource_start = float(os.environ.get("SOFT_SENSOR_RESOURCE_LOG_START_EPOCH", time.time()))

        def _mps_memory_mb(name):
            try:
                if device != "mps" or not torch.backends.mps.is_available():
                    return ""
                value = getattr(torch.mps, name)()
                return f"{float(value) / 1024 / 1024:.2f}"
            except Exception:
                return ""

        def _log_resource_event(note):
            if not resource_log_path:
                return
            now = time.time()
            row = {
                "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                "elapsed_s": f"{now - resource_start:.3f}",
                "kind": "mps_event",
                "root_pid": str(os.getpid()),
                "pid_count": "1",
                "cpu_percent_sum": "",
                "rss_mb_sum": "",
                "gpu_backend": device,
                "mps_current_allocated_mb": _mps_memory_mb("current_allocated_memory"),
                "mps_driver_allocated_mb": _mps_memory_mb("driver_allocated_memory"),
                "note": note,
            }
            with open(resource_log_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writerow(row)

        if device == "mps" and torch.backends.mps.is_available():
            tpt_utils._is_mps_supported = lambda: True

        model = TPTTabRegressor(
            n_estimators=n_estimators,
            device=device,
            fit_mode=fit_mode,
        )
        _log_resource_event("tpt_fit_start")
        model.fit(x_train, y_train)
        _log_resource_event("tpt_fit_end")
        result = model.predict(x_test)
        _log_resource_event("tpt_predict_end")
        np.save(pred_path, np.asarray(result.mean, dtype=np.float32))
        """
    )


class FdeWindowFeatureBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        try:
            from pipeline.feature_engine import align_feature_matrix_to_samples, build_feature_matrix
            from pipeline.window_policy import WindowGateConfig, extract_windows
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not import FDE feature/window pipeline: {exc}") from exc

        data = request.data.rename(columns={request.time_column: "Timestamp"}).copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.sort_values("Timestamp")
        feature_columns = [col for col in request.feature_columns if col in data.columns]
        if feature_columns:
            data[feature_columns] = data[feature_columns].ffill().bfill().fillna(0)
        target_times = pd.to_datetime(request.target_times).to_numpy(dtype="datetime64[ns]")
        extracted = extract_windows(
            data,
            target_times,
            int(request.window_minutes),
            feature_columns,
            WindowGateConfig(),
        )
        if not extracted.window_dfs:
            return pd.DataFrame(index=range(len(target_times)))
        matrix = build_feature_matrix(
            extracted.window_dfs,
            feature_columns,
            resolved_extraction=extraction,
            target_points=10,
            raw_n_points=20,
            n_jobs=1,
        )
        return align_feature_matrix_to_samples(
            matrix,
            target_times,
            resolved_extraction=extraction,
        ).reset_index(drop=True)


def _looks_like_fde_or_benchmark(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        (path / "packages" / "kernels").exists()
        or (path / "vendor" / "fde_packages" / "kernels").exists()
        or (path / "contestants" / "2_scoPe_regressor").exists()
    )
