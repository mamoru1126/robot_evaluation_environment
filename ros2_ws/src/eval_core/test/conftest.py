import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2] if len(PKG.parents) > 2 else PKG

# Allow running the tests from a source checkout without installing the package
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

# Use the repository config when it exists (source checkout); in the Docker
# image EVAL_CONFIG_DIR already points to /workspace/config.
if "EVAL_CONFIG_DIR" not in os.environ and (REPO / "config" / "scenarios").is_dir():
    os.environ["EVAL_CONFIG_DIR"] = str(REPO / "config")
