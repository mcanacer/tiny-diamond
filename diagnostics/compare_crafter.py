"""Older name for diagnostics/compare_wm.py (kept so earlier instructions still work)."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("compare_wm.py")), run_name="__main__")
