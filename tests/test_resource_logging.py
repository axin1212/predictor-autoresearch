from __future__ import annotations

import csv

from predictor_autoresearch.resource_logging import ResourceMonitor


def test_resource_monitor_samples_process_tree(tmp_path):
    ps_output = "\n".join(
        [
            "  PID  PPID  %CPU   RSS COMM",
            "   10     1   5.0 10000 python",
            "   11    10  50.0 20000 python",
            "   12    11  10.5  5000 python",
            "   99     1  99.0 99999 other",
        ]
    )
    path = tmp_path / "resource_usage.csv"
    monitor = ResourceMonitor(
        path,
        root_pid=10,
        interval_seconds=1.0,
        ps_output_provider=lambda: ps_output,
        start_epoch=1000.0,
    )

    monitor.initialize()
    monitor.sample_once(now_epoch=1002.5)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["kind"] == "process_tree"
    assert rows[0]["pid_count"] == "3"
    assert rows[0]["cpu_percent_sum"] == "65.50"
    assert rows[0]["rss_mb_sum"] == "34.18"
