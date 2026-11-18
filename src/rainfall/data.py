"""Load the monthly rainfall series."""
from pathlib import Path

import pandas as pd

TARGETS = {
    "total": ("rainfall-monthly-total.csv", "total_rainfall"),
    "max_daily": ("rainfall-monthly-highest-daily-total.csv", "maximum_rainfall_in_a_day"),
}


def load_series(target: str, data_dir: str | Path = "data") -> pd.Series:
    """Return one target as a monthly series (month-start index), validated for gaps and nulls."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGETS)}")
    filename, column = TARGETS[target]
    raw = pd.read_csv(Path(data_dir) / filename)
    raw["month"] = pd.to_datetime(raw["month"], format="%Y-%m")
    series = raw.set_index("month")[column].sort_index().asfreq("MS")
    if series.isna().any():
        raise ValueError(f"{target}: series has gaps or nulls")
    series.name = column
    return series
