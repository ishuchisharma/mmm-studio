"""Budget allocation on top of a fitted model's response curves."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import FitResult


@dataclass
class Allocation:
    budget: float
    current: pd.Series
    optimal: pd.Series
    current_revenue: float
    optimal_revenue: float
    lower: pd.Series
    upper: pd.Series
    success: bool
    message: str

    @property
    def uplift(self) -> float:
        return self.optimal_revenue - self.current_revenue

    def table(self, res: FitResult) -> pd.DataFrame:
        rows = []
        for ch in self.optimal.index:
            rows.append(dict(
                channel=ch,
                current=self.current[ch],
                optimal=self.optimal[ch],
                change=self.optimal[ch] - self.current[ch],
                change_pct=(self.optimal[ch] / self.current[ch] - 1) if self.current[ch] > 0 else np.nan,
                revenue_current=float(res.weekly_response(ch, self.current[ch])),
                revenue_optimal=float(res.weekly_response(ch, self.optimal[ch])),
                mroi_current=float(res.marginal_roi(ch, self.current[ch])),
                mroi_optimal=float(res.marginal_roi(ch, self.optimal[ch])),
                at_bound=("floor" if np.isclose(self.optimal[ch], self.lower[ch], rtol=1e-3)
                          else "ceiling" if np.isclose(self.optimal[ch], self.upper[ch], rtol=1e-3)
                          else ""),
            ))
        return pd.DataFrame(rows).set_index("channel")


def total_response(res: FitResult, spend: pd.Series) -> float:
    return float(sum(res.weekly_response(ch, spend[ch]) for ch in spend.index))


def optimise(
    res: FitResult,
    budget: float,
    current: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    starts: int = 8,
    seed: int = 3,
) -> Allocation:
    """Maximise weekly media-driven revenue for a fixed weekly budget.

    Spend per channel is held between `lower` and `upper`. Several starting
    points are tried because S-shaped curves make the problem non-convex.
    """
    chans = list(current.index)
    lo = lower[chans].values.astype(float)
    hi = upper[chans].values.astype(float)
    if lo.sum() > budget + 1e-6:
        raise ValueError("The channel floors add up to more than the budget. Lower the floors or raise the budget.")
    if hi.sum() < budget - 1e-6:
        raise ValueError("The channel ceilings add up to less than the budget. Raise the ceilings or lower the budget.")

    scale = max(budget, 1.0)

    def neg(x):
        return -sum(float(res.weekly_response(ch, x[i] * scale)) for i, ch in enumerate(chans)) / scale

    def neg_grad(x):
        return -np.array([float(res.marginal_roi(ch, x[i] * scale)) for i, ch in enumerate(chans)])

    cons = [{"type": "eq", "fun": lambda x: x.sum() - budget / scale, "jac": lambda x: np.ones_like(x)}]
    bounds = list(zip(lo / scale, hi / scale))

    rng = np.random.default_rng(seed)
    x_cur = current[chans].values / max(current.sum(), 1e-9) * budget
    candidates = [np.clip(x_cur, lo, hi), np.clip(np.full(len(chans), budget / len(chans)), lo, hi)]
    for _ in range(starts):
        w = rng.dirichlet(np.ones(len(chans)))
        candidates.append(np.clip(w * budget, lo, hi))

    best = None
    for x0 in candidates:
        r = minimize(neg, x0 / scale, jac=neg_grad, bounds=bounds, constraints=cons,
                     method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        if best is None or (r.fun < best.fun and abs(r.x.sum() - budget / scale) < 1e-4):
            best = r

    opt = pd.Series(best.x * scale, index=chans)
    return Allocation(
        budget=budget,
        current=current[chans],
        optimal=opt,
        current_revenue=total_response(res, current[chans]),
        optimal_revenue=total_response(res, opt),
        lower=lower[chans],
        upper=upper[chans],
        success=bool(best.success),
        message=str(best.message),
    )


def frontier(res: FitResult, current: pd.Series, lower_pct: float, upper_pct: float,
             multipliers=np.linspace(0.5, 2.0, 16)) -> pd.DataFrame:
    """Best achievable revenue at each total budget, with bounds scaled to that budget."""
    rows = []
    base = current.sum()
    for m in multipliers:
        b = base * m
        lower = current * lower_pct
        upper = current * upper_pct
        try:
            a = optimise(res, b, current, lower, upper, starts=3)
            opt_rev = a.optimal_revenue
        except ValueError:
            opt_rev = np.nan
        rows.append(dict(
            multiplier=m, budget=b,
            optimal_revenue=opt_rev,
            same_mix_revenue=total_response(res, current * m),
        ))
    return pd.DataFrame(rows)


def robustness(res: FitResult, budget: float, current: pd.Series, lower: pd.Series, upper: pd.Series,
               tol: float = 0.05) -> pd.DataFrame:
    """Re-run the optimiser on each near-best model and count which way each channel moves.

    A channel 'increases' if its optimal spend is more than `tol` above current, 'decreases' if more
    than `tol` below. Agreement across models shows how much the advice depends on one particular fit.
    """
    models = [res, *res.alternatives]
    moves = []
    for m in models:
        try:
            a = optimise(m, budget, current, lower, upper, starts=3)
        except ValueError:
            continue
        moves.append(a.optimal / current.replace(0, np.nan) - 1)
    mv = pd.DataFrame(moves)
    n = len(mv)
    return pd.DataFrame({
        "models": n,
        "up": (mv > tol).sum(),
        "down": (mv < -tol).sum(),
        "median_change": mv.median(),
    })
