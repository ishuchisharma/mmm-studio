"""Session state: the active dataset, its configuration and the cached model fit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mmm.data import DataConfig, Prepared, load_default, prepare
from mmm.model import Ensemble, FitResult, fit

DEFAULT_SETTINGS = dict(n_trials=1500, holdout=26, rssd_weight=0.3, seed=7, ref_weeks=52, n_alternatives=25)


def init():
    ss = st.session_state
    if "cfg" not in ss:
        ss.cfg = DataConfig()
        ss.df = load_default()
        ss.number_style = "K/M"
        ss.settings = dict(DEFAULT_SETTINGS)
        ss.scenarios = []


@st.cache_data(show_spinner=False, max_entries=8)
def _fit_cached(df: pd.DataFrame, cfg_key: str, settings_key: tuple, _cfg: DataConfig) -> tuple[FitResult, Prepared]:
    s = dict(settings_key)
    prep = prepare(df, _cfg)
    res = fit(prep, _cfg, n_trials=s["n_trials"], holdout=s["holdout"],
              rssd_weight=s["rssd_weight"], seed=s["seed"], ref_weeks=s["ref_weeks"],
              n_alternatives=s["n_alternatives"])
    return res, prep


def get_fit() -> tuple[FitResult, Prepared]:
    init()
    ss = st.session_state
    key = tuple(sorted(ss.settings.items()))
    with st.spinner("Fitting the model: searching carryover and saturation settings for each channel…"):
        return _fit_cached(ss.df, ss.cfg.key(), key, ss.cfg)


def get_models() -> tuple[FitResult, Ensemble, Prepared]:
    """Best single fit (for diagnostics), the averaged ensemble (for planning), and the prepared data."""
    res, prep = get_fit()
    return res, Ensemble(res), prep


def reset_downstream():
    """Call after the data or settings change, so plans built on the old model are cleared."""
    st.session_state.scenarios = []
    st.session_state.pop("allocation", None)
