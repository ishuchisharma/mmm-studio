"""Dataset configuration, loading and design-matrix preparation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FILE = DATA_DIR / "robyn_simulated_weekly.csv"

# Prior ranges for weekly carryover. Offline media (TV, outdoor, print) is
# remembered for weeks; clicks from digital mostly act within a week or two.
DECAY_RANGES = {"slow": (0.10, 0.80), "fast": (0.00, 0.40)}


@dataclass
class DataConfig:
    date_col: str = "DATE"
    target: str = "revenue"
    # display name -> spend column
    media: dict[str, str] = field(default_factory=lambda: {
        "TV": "tv_S",
        "Outdoor": "ooh_S",
        "Print": "print_S",
        "Facebook": "facebook_S",
        "Search": "search_S",
    })
    # display name -> "slow" | "fast"
    decay: dict[str, str] = field(default_factory=lambda: {
        "TV": "slow", "Outdoor": "slow", "Print": "slow",
        "Facebook": "fast", "Search": "fast",
    })
    controls: list[str] = field(default_factory=lambda: ["competitor_sales_B", "newsletter"])
    event_col: str | None = "events"
    event_baseline: str = "na"
    trend: bool = True
    seasonality_terms: int = 2  # Fourier pairs on a 52-week cycle
    currency: str = "₹"
    dataset_name: str = "Robyn simulated weekly (Meta open source)"

    def key(self) -> str:
        """Stable string used as a cache key."""
        return repr(sorted(asdict(self).items()))


def load_default() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_FILE)


def validate(df: pd.DataFrame, cfg: DataConfig) -> list[str]:
    """Return a list of problems that would stop the model from fitting."""
    problems = []
    needed = [cfg.date_col, cfg.target, *cfg.media.values(), *cfg.controls]
    if cfg.event_col:
        needed.append(cfg.event_col)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        problems.append(f"Columns not found in the file: {', '.join(missing)}.")
        return problems
    try:
        pd.to_datetime(df[cfg.date_col])
    except Exception:
        problems.append(f"Dates in '{cfg.date_col}' could not be read. Use a format like 2024-01-29.")
    for c in [cfg.target, *cfg.media.values(), *cfg.controls]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"'{c}' must be numeric.")
    if len(cfg.media) < 2:
        problems.append("Pick at least two media channels.")
    if len(df) < 52:
        problems.append(f"The file has {len(df)} weeks. A media mix model needs at least 52, ideally 104 or more.")
    for name, col in cfg.media.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
            problems.append(f"Spend for {name} has negative values.")
    return problems


@dataclass
class Prepared:
    dates: pd.Series
    y: np.ndarray
    spend: pd.DataFrame          # columns = channel display names
    controls: pd.DataFrame       # trend, seasonality, events, other controls
    control_groups: dict[str, str]  # control column -> group label
    channels: list[str]


def prepare(df: pd.DataFrame, cfg: DataConfig) -> Prepared:
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d = d.sort_values(cfg.date_col).reset_index(drop=True)
    n = len(d)

    spend = pd.DataFrame({name: d[col].astype(float).fillna(0.0) for name, col in cfg.media.items()})

    ctrl = {}
    groups = {}
    if cfg.trend:
        ctrl["trend"] = np.arange(n, dtype=float)
        groups["trend"] = "Trend"
    t = np.arange(n)
    for k in range(1, cfg.seasonality_terms + 1):
        ctrl[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52.18)
        ctrl[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.18)
        groups[f"sin{k}"] = groups[f"cos{k}"] = "Seasonality"
    for c in cfg.controls:
        ctrl[c] = d[c].astype(float).ffill().bfill().fillna(0.0).values
        groups[c] = c
    if cfg.event_col:
        ev = d[cfg.event_col].astype(str).fillna(cfg.event_baseline)
        for level in sorted(set(ev) - {cfg.event_baseline, "nan"}):
            ctrl[f"event_{level}"] = (ev == level).astype(float).values
            groups[f"event_{level}"] = "Events"

    return Prepared(
        dates=d[cfg.date_col],
        y=d[cfg.target].astype(float).values,
        spend=spend,
        controls=pd.DataFrame(ctrl, index=range(n)),
        control_groups=groups,
        channels=list(cfg.media.keys()),
    )


def vif(frame: pd.DataFrame) -> pd.Series:
    """Variance inflation factor for each column (multicollinearity check)."""
    X = frame.astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    out = {}
    for i, col in enumerate(frame.columns):
        others = np.delete(X, i, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        beta, *_ = np.linalg.lstsq(A, X[:, i], rcond=None)
        resid = X[:, i] - A @ beta
        r2 = 1 - resid.var() / max(X[:, i].var(), 1e-12)
        out[col] = 1 / max(1 - r2, 1e-9)
    return pd.Series(out)
