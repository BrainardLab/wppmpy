"""Cross-platform pytest runner for pre-commit hook."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
for python in [
    root / "analysis.venv" / "Scripts" / "python.exe",  # Windows
    root / "analysis.venv" / "bin" / "python",  # Mac/Linux
]:
    if python.exists():
        sys.exit(subprocess.call([str(python), "-m", "pytest"]))

sys.exit(subprocess.call([sys.executable, "-m", "pytest"]))
