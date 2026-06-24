from __future__ import annotations

from pathlib import Path
import sys

SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from predictor_autoresearch.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
