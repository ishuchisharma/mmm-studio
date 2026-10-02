import numpy as np
import pandas as pd

from mmm.data import DataConfig, load_default, prepare
from mmm.model import ChannelParams, FitResult, _ridge_bounded, fit
from mmm.optimiser import optimise
from mmm.transforms import geometric_adstock, half_life, hill, hill_derivative


def test_adstock_matches_loop():
    x = np.array([10, 0, 0, 5, 0], float)
    out = geometric_adstock(x, 0.5)
    expect = [10, 5, 2.5, 6.25, 3.125]
    assert np.allclose(out, expect)


def test_hill_half_point_and_derivative():
    assert np.isclose(hill(50.0, 2.0, 50.0), 0.5)
    x, h = 30.0, 1e-4
    num = (hill(x + h, 1.7, 40) - hill(x - h, 1.7, 40)) / (2 * h)
    assert np.isclose(num, hill_derivative(x, 1.7, 40), rtol=1e-4)


def test_half_life():
    assert np.isclose(half_life(0.5), 1.0)


def test_ridge_keeps_media_non_negative():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = -2 * X[:, 0] + 3 * X[:, 2] + rng.normal(size=200)
    c, _ = _ridge_bounded(X, y, 1e-4, n_media=2)
    assert c[0] >= 0 and c[1] >= 0
    assert np.isclose(c[2], 3, atol=0.3)


def test_recovers_known_effect():
    rng = np.random.default_rng(1)
    n = 156
    spend = pd.DataFrame({"A": rng.gamma(2, 50, n) * (rng.random(n) > 0.4),
                          "B": rng.gamma(2, 30, n) * (rng.random(n) > 0.4)})
    sa = hill(geometric_adstock(spend.A.values, 0.5), 1.0, 0.5 * geometric_adstock(spend.A.values, 0.5).max())
    sb = hill(geometric_adstock(spend.B.values, 0.1), 1.0, 0.5 * geometric_adstock(spend.B.values, 0.1).max())
    y = 1000 + 400 * sa + 150 * sb + rng.normal(0, 10, n)
    df = pd.DataFrame({"DATE": pd.date_range("2022-01-03", periods=n, freq="W-MON"), "rev": y, **spend})
    cfg = DataConfig(date_col="DATE", target="rev", media={"A": "A", "B": "B"},
                     decay={"A": "slow", "B": "fast"}, controls=[], event_col=None, trend=False,
                     seasonality_terms=0)
    res = fit(prepare(df, cfg), cfg, n_trials=600, holdout=20, rssd_weight=0.0)
    share = res.contrib[["A", "B"]].sum()
    true = pd.Series({"A": (400 * sa).sum(), "B": (150 * sb).sum()})
    assert np.allclose(share / share.sum(), true / true.sum(), atol=0.08)
    assert res.metrics["r2_holdout"] > 0.9


def test_optimiser_respects_budget_and_equalises_mroi():
    cfg = DataConfig()
    prep = prepare(load_default(), cfg)
    res = fit(prep, cfg, n_trials=300)
    cur = res.reference_spend()
    a = optimise(res, cur.sum(), cur, cur * 0.1, cur * 5)
    assert np.isclose(a.optimal.sum(), cur.sum(), rtol=1e-4)
    assert a.optimal_revenue >= a.current_revenue - 1e-6
    t = a.table(res)
    free = t[t.at_bound == ""]
    if len(free) > 1:
        assert free.mroi_optimal.std() / free.mroi_optimal.mean() < 0.02
