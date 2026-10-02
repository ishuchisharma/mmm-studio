"""Media mix model: hyperparameter search, bounded ridge fit, decomposition, bootstrap."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .data import DECAY_RANGES, DataConfig, Prepared
from .transforms import geometric_adstock, hill, hill_derivative

ALPHA_RANGE = (0.5, 3.0)
GAMMA_RANGE = (0.3, 1.0)   # half-saturation as a share of the channel's peak adstocked spend
LOG_LAMBDA_RANGE = (-4.0, 1.0)


@dataclass
class ChannelParams:
    theta: float
    alpha: float
    gamma_frac: float
    gamma: float = 0.0  # absolute half-saturation point, in adstocked spend units


@dataclass
class FitResult:
    channels: list[str]
    params: dict[str, ChannelParams]
    lam: float
    coef_media: pd.Series
    coef_ctrl: pd.Series
    intercept: float
    dates: pd.Series
    y: np.ndarray
    fitted: np.ndarray
    spend: pd.DataFrame
    saturated: pd.DataFrame
    contrib: pd.DataFrame          # Baseline + one column per channel, per week
    contrib_controls: pd.DataFrame  # intercept and each control group, per week
    metrics: dict
    holdout_weeks: int
    holdout_pred: np.ndarray        # predictions from the train-only model, full length
    search_log: pd.DataFrame
    boot_coef: np.ndarray | None = None  # (n_boot, n_channels)
    ref_weeks: int = 52
    notes: list[str] = field(default_factory=list)
    alternatives: list["FitResult"] = field(default_factory=list)

    # ---------- response curves ----------
    def weekly_response(self, ch: str, weekly_spend):
        """Steady-state weekly revenue from holding spend constant at `weekly_spend`."""
        p = self.params[ch]
        stock = np.asarray(weekly_spend, dtype=float) / (1 - p.theta)
        return self.coef_media[ch] * hill(stock, p.alpha, p.gamma)

    def marginal_roi(self, ch: str, weekly_spend):
        """Revenue from the next unit of weekly spend, at a steady state."""
        p = self.params[ch]
        stock = np.asarray(weekly_spend, dtype=float) / (1 - p.theta)
        return self.coef_media[ch] * hill_derivative(stock, p.alpha, p.gamma) / (1 - p.theta)

    def reference_spend(self) -> pd.Series:
        """Average weekly spend over the most recent `ref_weeks` weeks."""
        return self.spend.tail(self.ref_weeks).mean()

    # ---------- summary tables ----------
    def channel_table(self) -> pd.DataFrame:
        spend_tot = self.spend.sum()
        contrib_tot = self.contrib[self.channels].sum()
        ref = self.reference_spend()
        rows = []
        for ch in self.channels:
            p = self.params[ch]
            roi = contrib_tot[ch] / spend_tot[ch] if spend_tot[ch] > 0 else np.nan
            row = dict(
                channel=ch,
                spend=spend_tot[ch],
                contribution=contrib_tot[ch],
                spend_share=spend_tot[ch] / spend_tot.sum(),
                effect_share=contrib_tot[ch] / max(contrib_tot.sum(), 1e-9),
                roi=roi,
                mroi=float(self.marginal_roi(ch, ref[ch])),
                ref_weekly_spend=ref[ch],
                theta=p.theta,
                alpha=p.alpha,
                gamma=p.gamma,
            )
            if self.boot_coef is not None and spend_tot[ch] > 0:
                j = self.channels.index(ch)
                rois = self.boot_coef[:, j] * self.saturated[ch].sum() / spend_tot[ch]
                row["roi_lo"], row["roi_hi"] = np.percentile(rois, [5, 95])
            rows.append(row)
        out = pd.DataFrame(rows).set_index("channel")
        if self.alternatives:
            alts = [self, *self.alternatives]
            roi = pd.DataFrame([a.contrib[a.channels].sum() / a.spend.sum() for a in alts])
            ref_all = self.reference_spend()
            mroi = pd.DataFrame([{ch: float(a.marginal_roi(ch, ref_all[ch])) for ch in a.channels} for a in alts])
            out["roi_alt_lo"], out["roi_alt_hi"] = roi.quantile(0.1), roi.quantile(0.9)
            out["mroi_alt_lo"], out["mroi_alt_hi"] = mroi.quantile(0.1), mroi.quantile(0.9)
        return out


# ---------------------------------------------------------------- fitting


def _media_matrix(spend: pd.DataFrame, params: dict[str, ChannelParams]) -> pd.DataFrame:
    cols = {}
    for ch in spend.columns:
        p = params[ch]
        ad = geometric_adstock(spend[ch].values, p.theta)
        p.gamma = p.gamma_frac * max(ad.max(), 1e-9)
        cols[ch] = hill(ad, p.alpha, p.gamma)
    return pd.DataFrame(cols, index=spend.index)


def _ridge_bounded(X: np.ndarray, y: np.ndarray, lam: float, n_media: int):
    """Ridge regression with media coefficients forced to be >= 0.

    Columns are standardised, the penalty is added as extra rows, and the
    bounded least-squares problem is solved directly. Returns raw-scale
    coefficients and the intercept.
    """
    mu = X.mean(0)
    sd = X.std(0)
    keep = sd > 1e-12
    Xs = np.zeros_like(X)
    Xs[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]
    ybar, ysd = y.mean(), max(y.std(), 1e-12)
    yc = (y - ybar) / ysd
    p = X.shape[1]
    A = np.vstack([Xs, np.sqrt(lam * len(y)) * np.eye(p)])
    b = np.concatenate([yc, np.zeros(p)])
    lb = np.full(p, -np.inf)
    lb[:n_media] = 0.0
    sol = lsq_linear(A, b, bounds=(lb, np.full(p, np.inf)), method="bvls" if p < 40 else "trf")
    c = np.zeros(p)
    c[keep] = sol.x[keep] * ysd / sd[keep]
    intercept = ybar - float(mu @ c)
    return c, intercept


def _nrmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / max(y.max() - y.min(), 1e-12))


def _r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def _mape(y, yhat):
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])))


def _rssd(coef_media, sat: pd.DataFrame, spend: pd.DataFrame) -> float:
    """Distance between each channel's share of effect and share of spend."""
    eff = coef_media * sat.sum().values
    eff_share = eff / max(eff.sum(), 1e-12)
    sp = spend.sum().values
    sp_share = sp / max(sp.sum(), 1e-12)
    return float(np.sqrt(np.sum((eff_share - sp_share) ** 2)))


def _sample(rng, channels, decay_map):
    params = {}
    for ch in channels:
        lo, hi = DECAY_RANGES[decay_map.get(ch, "slow")]
        params[ch] = ChannelParams(
            theta=rng.uniform(lo, hi),
            alpha=rng.uniform(*ALPHA_RANGE),
            gamma_frac=rng.uniform(*GAMMA_RANGE),
        )
    return params, 10 ** rng.uniform(*LOG_LAMBDA_RANGE)


def _perturb(rng, params, lam, decay_map, scale):
    new = {}
    for ch, p in params.items():
        lo, hi = DECAY_RANGES[decay_map.get(ch, "slow")]
        new[ch] = ChannelParams(
            theta=float(np.clip(p.theta + rng.normal(0, scale * (hi - lo)), lo, hi)),
            alpha=float(np.clip(p.alpha + rng.normal(0, scale * 2.5), *ALPHA_RANGE)),
            gamma_frac=float(np.clip(p.gamma_frac + rng.normal(0, scale * 0.7), *GAMMA_RANGE)),
        )
    loglam = np.clip(np.log10(lam) + rng.normal(0, scale * 4), *LOG_LAMBDA_RANGE)
    return new, 10 ** loglam


def _evaluate(prep: Prepared, params, lam, holdout: int, rssd_weight: float):
    sat = _media_matrix(prep.spend, params)
    X = np.column_stack([sat.values, prep.controls.values])
    n_media = sat.shape[1]
    ntr = len(prep.y) - holdout
    c, b0 = _ridge_bounded(X[:ntr], prep.y[:ntr], lam, n_media)
    pred = b0 + X @ c
    nrmse_val = _nrmse(prep.y[ntr:], pred[ntr:]) if holdout > 0 else _nrmse(prep.y, pred)
    rssd = _rssd(c[:n_media], sat.iloc[:ntr], prep.spend.iloc[:ntr])
    return nrmse_val + rssd_weight * rssd, nrmse_val, rssd


def fit(
    prep: Prepared,
    cfg: DataConfig,
    n_trials: int = 1500,
    holdout: int = 26,
    rssd_weight: float = 0.3,
    seed: int = 7,
    ref_weeks: int = 52,
    progress: Callable[[float], None] | None = None,
    n_alternatives: int = 25,
) -> FitResult:
    rng = np.random.default_rng(seed)
    channels = prep.channels
    log = []
    explore = []
    best = (np.inf, None, None)

    n_global = int(n_trials * 0.7)
    n_local = n_trials - n_global
    for i in range(n_trials):
        if i < n_global or best[1] is None:
            params, lam = _sample(rng, channels, cfg.decay)
            stage = "explore"
        else:
            scale = 0.15 * (1 - (i - n_global) / max(n_local, 1)) + 0.02
            params, lam = _perturb(rng, best[1], best[2], cfg.decay, scale)
            stage = "refine"
        obj, nrmse_val, rssd = _evaluate(prep, params, lam, holdout, rssd_weight)
        log.append((i, stage, obj, nrmse_val, rssd))
        if stage == "explore":
            explore.append((obj, params, lam))
        if obj < best[0]:
            best = (obj, params, lam)
        if progress and i % 25 == 0:
            progress(i / n_trials)
    if progress:
        progress(1.0)

    _, params, lam = best
    main = _build(prep, params, lam, holdout, n_trials, ref_weeks)
    main.search_log = pd.DataFrame(log, columns=["trial", "stage", "objective", "nrmse", "rssd"])

    # near-best alternatives: the top independent random draws, refit on all data
    explore.sort(key=lambda t: t[0])
    main.alternatives = [_build(prep, p_, l_, holdout, n_trials, ref_weeks) for _, p_, l_ in explore[:n_alternatives]]
    return main


def _build(prep: Prepared, params, lam, holdout, n_trials, ref_weeks) -> FitResult:
    channels = prep.channels
    params = {ch: ChannelParams(p.theta, p.alpha, p.gamma_frac) for ch, p in params.items()}
    sat = _media_matrix(prep.spend, params)
    X = np.column_stack([sat.values, prep.controls.values])
    n_media = len(channels)
    ntr = len(prep.y) - holdout

    # honest out-of-sample check: model trained without the holdout weeks
    c_tr, b_tr = _ridge_bounded(X[:ntr], prep.y[:ntr], lam, n_media)
    holdout_pred = b_tr + X @ c_tr

    # final model on all weeks
    c, b0 = _ridge_bounded(X, prep.y, lam, n_media)
    fitted = b0 + X @ c
    coef_media = pd.Series(c[:n_media], index=channels)
    coef_ctrl = pd.Series(c[n_media:], index=prep.controls.columns)

    contrib = pd.DataFrame({ch: coef_media[ch] * sat[ch].values for ch in channels})
    ctrl_parts = {"Intercept": np.full(len(prep.y), b0)}
    for col in prep.controls.columns:
        g = prep.control_groups[col]
        ctrl_parts[g] = ctrl_parts.get(g, 0) + coef_ctrl[col] * prep.controls[col].values
    contrib_controls = pd.DataFrame(ctrl_parts)
    contrib.insert(0, "Baseline", contrib_controls.sum(axis=1).values)

    resid = prep.y - fitted
    dw = float(np.sum(np.diff(resid) ** 2) / max(np.sum(resid**2), 1e-12))
    metrics = dict(
        r2_full=_r2(prep.y, fitted),
        nrmse_full=_nrmse(prep.y, fitted),
        mape_full=_mape(prep.y, fitted),
        r2_train=_r2(prep.y[:ntr], holdout_pred[:ntr]),
        r2_holdout=_r2(prep.y[ntr:], holdout_pred[ntr:]) if holdout else np.nan,
        nrmse_holdout=_nrmse(prep.y[ntr:], holdout_pred[ntr:]) if holdout else np.nan,
        mape_holdout=_mape(prep.y[ntr:], holdout_pred[ntr:]) if holdout else np.nan,
        rssd=_rssd(c[:n_media], sat, prep.spend),
        durbin_watson=dw,
        lambda_=lam,
        trials=n_trials,
    )

    notes = []
    zero = [ch for ch in channels if coef_media[ch] <= 1e-9]
    if zero:
        notes.append(
            f"The model found no measurable effect for {', '.join(zero)}. "
            "That can mean the channel truly does little, or that its spend moves too "
            "closely with other channels for the data to separate them."
        )
    if metrics["r2_holdout"] < 0.5 and holdout:
        notes.append("Holdout R² is below 0.5, so treat channel-level numbers as directional.")
    if dw < 1.2 or dw > 2.8:
        notes.append(
            f"Durbin–Watson is {dw:.2f}: residuals are autocorrelated, so uncertainty "
            "bands are likely too narrow."
        )

    return FitResult(
        channels=channels, params=params, lam=lam,
        coef_media=coef_media, coef_ctrl=coef_ctrl, intercept=b0,
        dates=prep.dates, y=prep.y, fitted=fitted, spend=prep.spend,
        saturated=sat, contrib=contrib, contrib_controls=contrib_controls,
        metrics=metrics, holdout_weeks=holdout, holdout_pred=holdout_pred,
        search_log=pd.DataFrame(),
        ref_weeks=ref_weeks, notes=notes,
    )


def bootstrap(res: FitResult, prep: Prepared, n_boot: int = 200, block: int = 4, seed: int = 11) -> np.ndarray:
    """Moving-block residual bootstrap of media coefficients, hyperparameters held fixed."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([res.saturated.values, prep.controls.values])
    resid = res.y - res.fitted
    n = len(resid)
    n_media = len(res.channels)
    out = np.empty((n_boot, n_media))
    starts_max = n - block
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, starts_max + 1, n // block + 1)])[:n]
        y_star = res.fitted + resid[idx]
        c, _ = _ridge_bounded(X, y_star, res.lam, n_media)
        out[b] = c[:n_media]
    return out


class Ensemble:
    """Averages response curves across the best model and its near-best alternatives.

    Any single fit can sit on a lucky corner of a flat search landscape. Averaging the curves of
    the top-scoring candidates (a form of model averaging) gives planning numbers that move much
    less when the random seed or trial count changes.
    """

    def __init__(self, res: FitResult):
        self.best = res
        self.models = [res, *res.alternatives]
        self.channels = res.channels
        self.ref_weeks = res.ref_weeks
        self.spend = res.spend

    def weekly_response(self, ch, weekly_spend):
        return np.mean([m.weekly_response(ch, weekly_spend) for m in self.models], axis=0)

    def marginal_roi(self, ch, weekly_spend):
        return np.mean([m.marginal_roi(ch, weekly_spend) for m in self.models], axis=0)

    def reference_spend(self) -> pd.Series:
        return self.best.reference_spend()

    def channel_table(self) -> pd.DataFrame:
        """Ensemble ROI and next-unit return, with the spread across member models."""
        ref = self.reference_spend()
        spend_tot = self.spend.sum()
        roi = pd.DataFrame([m.contrib[self.channels].sum() / spend_tot for m in self.models])
        contrib = pd.DataFrame([m.contrib[self.channels].sum() for m in self.models])
        mroi = pd.DataFrame([{ch: float(m.marginal_roi(ch, ref[ch])) for ch in self.channels} for m in self.models])
        out = pd.DataFrame(index=pd.Index(self.channels, name="channel"))
        out["spend"] = spend_tot
        out["contribution"] = contrib.mean()
        out["spend_share"] = spend_tot / spend_tot.sum()
        out["effect_share"] = out.contribution / out.contribution.sum()
        out["roi"] = roi.mean()
        out["roi_lo"], out["roi_hi"] = roi.quantile(0.1), roi.quantile(0.9)
        out["mroi"] = [float(self.marginal_roi(ch, ref[ch])) for ch in self.channels]
        out["mroi_lo"], out["mroi_hi"] = mroi.quantile(0.1), mroi.quantile(0.9)
        out["ref_weekly_spend"] = ref
        return out
