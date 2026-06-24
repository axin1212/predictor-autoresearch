from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
import sys


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    cwd: str
    fde_root: Path | None
    dependencies: list[DependencyStatus]
    weight_status: str

    @property
    def ok(self) -> bool:
        required_ok = all(dep.ok for dep in self.dependencies if dep.name != "tsfresh")
        return required_ok and self.fde_root is not None

    def to_text(self) -> str:
        lines = [
            f"Python: {self.python}",
            f"CWD: {self.cwd}",
            f"FDE root: {self.fde_root or 'not found'}",
            f"Model weights: {self.weight_status}",
        ]
        for dep in self.dependencies:
            mark = "OK" if dep.ok else "FAIL"
            optional = " optional" if dep.name == "tsfresh" else ""
            lines.append(f"{mark}{optional} {dep.name}: {dep.detail}")
        return "\n".join(lines)


def check_import(name: str) -> DependencyStatus:
    spec = importlib.util.find_spec(name)
    return DependencyStatus(name=name, ok=spec is not None, detail="importable" if spec else "missing")


def build_environment_report(fde_root: Path | None, model_type: str = "tabpfn3") -> EnvironmentReport:
    required = ["xgboost", "plotly", "pyarrow"]
    if model_type == "tabpfn3":
        required.insert(0, "tabpfn")
        weight_status = "deferred to FDE TabPFN resolver"
    elif model_type == "tpt":
        required.insert(0, "tpt_tab")
        weight_status = "deferred to FDE TPT_tab resolver"
    else:
        required.insert(0, model_type)
        weight_status = "not checked"
    deps = [check_import(name) for name in required]
    deps.append(check_import("tsfresh"))
    return EnvironmentReport(
        python=sys.version.split()[0],
        cwd=os.getcwd(),
        fde_root=fde_root,
        dependencies=deps,
        weight_status=weight_status,
    )
