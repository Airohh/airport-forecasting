import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "src" / "airport_forecast" / "dashboard.py"),
    run_name="__main__",
)
