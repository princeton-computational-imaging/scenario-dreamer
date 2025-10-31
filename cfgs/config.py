from pathlib import Path
import os

# 1.  Pull from user’s shell if it exists  ──────────────────────────
#    $ export PROJECT_ROOT=/my/cloned/scenario-dreamer
#    $ export CONFIG_PATH=/my/custom/location/cfgs   # optional override
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_PATH  = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "cfgs")).as_posix()

### CONSTANTS

# lg_type
NON_PARTITIONED = 0
PARTITIONED = 1 

# partition mask
AFTER_PARTITION = 0
BEFORE_PARTITION = 1

# Waymo connection type
LANE_CONNECTION_TYPES_WAYMO = {
    "none": 0,
    "pred": 1,
    "succ": 2,
    "left": 3,
    "right": 4,
    "self": 5
}

# NuPlan connection type
LANE_CONNECTION_TYPES_NUPLAN = {
    "none": 0,
    "pred": 1,
    "succ": 2,
    "self": 3,
}

# proportion of nocturne compatible scenes in the Waymo dataset
PROPORTION_NOCTURNE_COMPATIBLE = 0.38
NOCTURNE_COMPATIBLE = 1

# object type
NUPLAN_VEHICLE = 0
NUPLAN_PEDESTRIAN = 1
NUPLAN_STATIC_OBJECT = 2

# unified format for computing metrics
# [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
UNIFIED_FORMAT_INDICES = {
    'pos_x': 0,
    'pos_y': 1,
    'speed': 2,
    'cos_heading': 3,
    'sin_heading': 4,
    'length': 5,
    'width': 6
}

