from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import threading
import time
from collections.abc import Callable


RESOURCE_LOG_COLUMNS = [
    "timestamp",
    "elapsed_s",
    "kind",
    "root_pid",
    "pid_count",
    "cpu_percent_sum",
    "rss_mb_sum",
    "gpu_backend",
    "mps_current_allocated_mb",
    "mps_driver_allocated_mb",
    "note",
]


@dataclass(frozen=True)
class ProcessSample:
    pid_count: int
    cpu_percent_sum: float
    rss_mb_sum: float


class ResourceMonitor:
    def __init__(
        self,
        path: Path,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 2.0,
        ps_output_provider: Callable[[], str] | None = None,
        start_epoch: float | None = None,
    ) -> None:
        self.path = path
        self.root_pid = int(root_pid or os.getpid())
        self.interval_seconds = max(0.2, float(interval_seconds))
        self.ps_output_provider = ps_output_provider or _ps_output
        self.start_epoch = float(start_epoch or time.time())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RESOURCE_LOG_COLUMNS).writeheader()

    def sample_once(self, now_epoch: float | None = None) -> None:
        now = float(now_epoch or time.time())
        try:
            sample = _sample_process_tree(self.root_pid, self.ps_output_provider())
        except Exception as exc:  # noqa: BLE001
            self._append_monitor_error(now, f"process sampling failed: {exc!r}")
            return
        self._append_row(
            {
                "timestamp": _iso_timestamp(now),
                "elapsed_s": f"{now - self.start_epoch:.3f}",
                "kind": "process_tree",
                "root_pid": str(self.root_pid),
                "pid_count": str(sample.pid_count),
                "cpu_percent_sum": f"{sample.cpu_percent_sum:.2f}",
                "rss_mb_sum": f"{sample.rss_mb_sum:.2f}",
                "gpu_backend": "",
                "mps_current_allocated_mb": "",
                "mps_driver_allocated_mb": "",
                "note": "",
            }
        )

    def start(self) -> "ResourceMonitor":
        self.initialize()
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="predictor-resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        try:
            self.sample_once()
        except Exception as exc:  # noqa: BLE001
            self._append_monitor_error(time.time(), f"final sample failed: {exc!r}")

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001
                self._append_monitor_error(time.time(), "resource monitor sample failed")

    def _append_monitor_error(self, now: float, note: str) -> None:
        self._append_row(
            {
                "timestamp": _iso_timestamp(now),
                "elapsed_s": f"{now - self.start_epoch:.3f}",
                "kind": "monitor_error",
                "root_pid": str(self.root_pid),
                "pid_count": "",
                "cpu_percent_sum": "",
                "rss_mb_sum": "",
                "gpu_backend": "",
                "mps_current_allocated_mb": "",
                "mps_driver_allocated_mb": "",
                "note": note,
            }
        )

    def _append_row(self, row: dict[str, str]) -> None:
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=RESOURCE_LOG_COLUMNS).writerow(row)


def _sample_process_tree(root_pid: int, ps_output: str) -> ProcessSample:
    processes = _parse_ps_output(ps_output)
    children_by_parent: dict[int, list[int]] = {}
    for pid, process in processes.items():
        children_by_parent.setdefault(process["ppid"], []).append(pid)

    selected: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in selected:
            continue
        selected.add(pid)
        stack.extend(children_by_parent.get(pid, []))

    cpu = 0.0
    rss_kb = 0.0
    present = 0
    for pid in selected:
        process = processes.get(pid)
        if process is None:
            continue
        present += 1
        cpu += float(process["cpu"])
        rss_kb += float(process["rss"])
    return ProcessSample(pid_count=present, cpu_percent_sum=cpu, rss_mb_sum=rss_kb / 1024.0)


def _parse_ps_output(output: str) -> dict[int, dict[str, float | int]]:
    processes: dict[int, dict[str, float | int]] = {}
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4 or not parts[0].strip().isdigit():
            continue
        pid = int(parts[0])
        processes[pid] = {
            "ppid": int(parts[1]),
            "cpu": float(parts[2]),
            "rss": float(parts[3]),
        }
    return processes


def _ps_output() -> str:
    return subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,ppid=,pcpu=,rss=,comm="],
        text=True,
    )


def _iso_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
