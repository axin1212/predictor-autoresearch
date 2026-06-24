from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path

    @classmethod
    def create(cls, working_dir: Path, timestamp: str | None = None) -> "RunArtifacts":
        stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = working_dir / f"autoresearch_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_dir=run_dir)

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.html"

    @property
    def resource_usage_path(self) -> Path:
        return self.run_dir / "resource_usage.csv"

    def candidate_dir(self, candidate_id: str) -> Path:
        path = self.run_dir / "candidates" / _safe_name(candidate_id)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
