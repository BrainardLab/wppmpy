"""Pre-commit hook: machine_config.py must be byte-for-byte identical in
wppmpy_public and wppmpy_private.  Assumes the two repos are siblings on disk
(both under the same parent directory), which matches the standard lab layout.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # wppmpy root
THIS_COPY = HERE / "aepsych/aepsych_dconfig/machine_config.py"
OTHER_COPY = HERE.parent / "wppmpy_private/aepsych/aepsych_dconfig/machine_config.py"

if not OTHER_COPY.exists():
    print(f"note: {OTHER_COPY} not found; skipping sync check.")
    sys.exit(0)

if THIS_COPY.read_text() != OTHER_COPY.read_text():
    print("ERROR: machine_config.py has diverged between repos.")
    print(f"  this copy (wppmpy_public):   {THIS_COPY}")
    print(f"  other copy (wppmpy_private): {OTHER_COPY}")
    print("Make both files identical before committing.")
    sys.exit(1)

sys.exit(0)
