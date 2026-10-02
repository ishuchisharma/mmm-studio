"""Media transforms: carryover (geometric adstock) and diminishing returns (Hill)."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Carry a share `theta` of last week's effective spend into this week.

    a_t = x_t + theta * a_{t-1}. Implemented as an IIR filter, so it is O(n).
    """
    x = np.asarray(x, dtype=float)
    if theta <= 0:
        return x.copy()
    return lfilter([1.0], [1.0, -theta], x)


def hill(x: np.ndarray | float, alpha: float, gamma: float) -> np.ndarray | float:
    """Hill saturation in [0, 1). `gamma` is the half-saturation point, `alpha` the shape.

    alpha > 1 gives an S-curve (a threshold before spend starts working);
    alpha <= 1 gives a concave curve (every extra rupee works a little less).
    """
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    gamma = max(gamma, 1e-12)
    xa = np.power(x, alpha)
    return xa / (xa + gamma**alpha)


def hill_derivative(x: np.ndarray | float, alpha: float, gamma: float) -> np.ndarray | float:
    """d/dx of the Hill curve, used for marginal ROI."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    ga = gamma**alpha
    xa = np.power(x, alpha)
    return alpha * ga * np.power(x, alpha - 1) / (xa + ga) ** 2


def decay_weights(theta: float, weeks: int = 12) -> np.ndarray:
    """Share of a one-off week of spend that is still working k weeks later."""
    return theta ** np.arange(weeks)


def half_life(theta: float) -> float:
    """Weeks until a burst of spend has lost half its effect."""
    if theta <= 0:
        return 0.0
    return float(np.log(0.5) / np.log(theta))
