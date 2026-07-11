from __future__ import annotations

"""Run the Economic Mesh V2 private-baseline simulation.

The complete implementation is stored in ordered source fragments under
``source_parts/`` and compiled as one Python module at runtime.
"""

from pathlib import Path

base_dir = Path(__file__).resolve().parent
parts = sorted((base_dir / "source_parts").glob("part_*.pyfrag"))
if not parts:
    raise FileNotFoundError("No V2 source fragments were found.")

source = "".join(path.read_text(encoding="utf-8") for path in parts)
exec(
    compile(source, str(base_dir / "economic_mesh_v2_private_full.py"), "exec"),
    globals(),
    globals(),
)
