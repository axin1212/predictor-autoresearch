from __future__ import annotations

from pathlib import Path

from predictor_autoresearch import cli


def test_cli_smoke(monkeypatch, capsys):
    def fake_run_autoresearch(**kwargs):
        assert kwargs["data_file"] == Path("data.parquet")
        assert kwargs["target_column"] == "target"
        assert kwargs["model_type"] == "tabpfn3"
        return Path("/tmp/report.html")

    monkeypatch.setattr(cli, "run_autoresearch", fake_run_autoresearch)

    rc = cli.main(["data.parquet", "target"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "report.html: /tmp/report.html" in output
    assert "resource_usage.csv: /tmp/resource_usage.csv" in output


def test_zero_time_budget_stays_unlimited():
    assert cli._time_budget_seconds(0) == 0
    assert cli._time_budget_seconds(0.0) == 0
    assert cli._time_budget_seconds(0.5) == 30.0
