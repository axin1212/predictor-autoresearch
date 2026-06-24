from __future__ import annotations

from predictor_autoresearch.artifacts import RunArtifacts


def test_run_artifacts_creates_expected_paths(tmp_path):
    artifacts = RunArtifacts.create(tmp_path, timestamp="20260101_000000")

    assert artifacts.run_dir.exists()
    assert artifacts.report_path.name == "report.html"
    assert artifacts.resource_usage_path.name == "resource_usage.csv"
    assert artifacts.candidate_dir("c1").exists()
