from __future__ import annotations

from types import SimpleNamespace

import pytest

from predictor_autoresearch.fde_bridge import (
    IsolatedTabPFN3Predictor,
    _tabpfn3_child_code,
    _tpt_child_code,
    load_tabpfn3_predictor_factory,
    resolve_preferred_device,
)


def test_resolve_preferred_device_auto_prefers_mps(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    assert resolve_preferred_device("auto") == "mps"


def test_resolve_preferred_device_auto_falls_back_to_cpu(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False, is_built=lambda: False),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    assert resolve_preferred_device("auto") == "cpu"


def test_resolve_preferred_device_auto_fails_fast_when_mps_hidden(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False, is_built=lambda: True),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    with pytest.raises(RuntimeError, match="Metal devices are hidden"):
        resolve_preferred_device("auto")


def test_tpt_child_code_patches_mps_version_check():
    code = _tpt_child_code()

    assert "_is_mps_supported" in code
    assert "torch.backends.mps.is_available()" in code


def test_tpt_child_code_logs_mps_resource_events():
    code = _tpt_child_code()

    assert "SOFT_SENSOR_RESOURCE_LOG_PATH" in code
    assert "mps_current_allocated_mb" in code
    assert "tpt_fit_start" in code
    assert "tpt_predict_end" in code


def test_tabpfn3_factory_uses_isolated_child(monkeypatch):
    monkeypatch.setattr("predictor_autoresearch.fde_bridge.resolve_preferred_device", lambda device: "mps")

    factory = load_tabpfn3_predictor_factory(device="auto", fit_mode="fit_preprocessors", n_estimators=1)

    predictor = factory()
    assert isinstance(predictor, IsolatedTabPFN3Predictor)
    assert predictor.device == "mps"
    assert predictor.fit_mode == "fit_preprocessors"
    assert predictor.n_estimators == 1


def test_tabpfn3_child_code_logs_mps_resource_events():
    code = _tabpfn3_child_code()

    assert "resolve_checkpoint_file" in code
    assert "TabPFNRegressor" in code
    assert "SOFT_SENSOR_RESOURCE_LOG_PATH" in code
    assert "tabpfn3_fit_start" in code
    assert "tabpfn3_predict_end" in code
