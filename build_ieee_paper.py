"""Compatibility entry point for the advisor draft with IEEE-style references."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_advisor_draft.py")), run_name="__main__")
