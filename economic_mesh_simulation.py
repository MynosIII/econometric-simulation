"""Executable entry point for the spatial agent-based economic simulation.

The model is divided into ordered source sections under ``src/``. They are
executed in one shared namespace so the original single-file simulation remains
functionally identical while being easier to browse on GitHub.
"""

from pathlib import Path


def _load_model_sections() -> None:
    namespace = globals()
    source_directory = Path(__file__).resolve().parent / "src"

    sections = sorted(source_directory.glob("[0-9][0-9]_*.py"))
    if not sections:
        raise FileNotFoundError(f"No simulation source sections found in {source_directory}")

    for section in sections:
        source = section.read_text(encoding="utf-8")
        code = compile(source, str(section), "exec")
        exec(code, namespace, namespace)


_load_model_sections()
