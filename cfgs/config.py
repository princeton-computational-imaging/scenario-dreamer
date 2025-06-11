from pathlib import Path
import os

# 1.  Pull from user’s shell if it exists  ──────────────────────────
#    $ export PROJECT_ROOT=/my/cloned/scenario-dreamer
#    $ export CONFIG_PATH=/my/custom/location/cfgs   # optional override
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_PATH  = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "cfgs")).as_posix()
