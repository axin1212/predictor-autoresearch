from __future__ import annotations

from predictor_autoresearch.env_check import DependencyStatus, EnvironmentReport
from predictor_autoresearch.fde_bridge import add_fde_to_path, find_fde_root


def test_environment_report_has_failure_summary():
    report = EnvironmentReport(
        python="3.11",
        cwd="/tmp/x",
        fde_root=None,
        dependencies=[DependencyStatus("tabpfn", False, "missing")],
        weight_status="not checked",
    )

    assert not report.ok
    assert "tabpfn" in report.to_text()
    assert "missing" in report.to_text()


def test_find_fde_root_accepts_benchmark_vendor_layout(tmp_path):
    root = tmp_path / "benchmark"
    (root / "vendor" / "fde_packages" / "kernels").mkdir(parents=True)

    assert find_fde_root(tmp_path, explicit=root) == root.resolve()


def test_add_fde_to_path_includes_existing_vendor_paths(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    kernels = root / "vendor" / "fde_packages" / "kernels"
    kernels.mkdir(parents=True)
    monkeypatch.setattr("sys.path", [])

    add_fde_to_path(root)

    assert str(kernels) in __import__("sys").path
