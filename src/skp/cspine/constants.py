import os
from pathlib import Path

LEVELS = tuple(f"C{index}" for index in range(1, 8))
TARGET_COLUMNS = ("patient_overall", *LEVELS)

WINDOW_CENTER = 400.0
WINDOW_WIDTH = 1800.0
WINDOW_LOWER = WINDOW_CENTER - WINDOW_WIDTH / 2
WINDOW_UPPER = WINDOW_CENTER + WINDOW_WIDTH / 2

DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "CSPINE_DATA_ROOT",
        "data/rsna-2022-cervical-spine-fracture-detection",
    )
)
