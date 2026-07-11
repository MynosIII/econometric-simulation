from __future__ import annotations

"""Executable loader for the Economic Mesh V2 private-baseline model.

The implementation is split across ``src/`` only to keep the large simulation
readable. Each source section is executed in the same global namespace, so the
behavior is equivalent to the original single-file version.
"""

from pathlib import Path

SOURCE_FILES = (
    "01_core.py",
    "02_agents.py",
    "03_equity_and_credit.py",
    "04_bankruptcy_labor_consumption.py",
    "05_simulation_engine.py",
    "06_dashboard_and_cli.py",
)

base_dir = Path(__file__).resolve().parent / "src"
namespace = globals()

for filename in SOURCE_FILES:
    path = base_dir / filename
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace, namespace)
