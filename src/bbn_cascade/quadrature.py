"""Composite Gauss rules resolving the thermal peak without global high order."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=None)
def gauss_rule(order):
    nodes, weights = leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def thermal_quadrature(lower, upper, temperature, order, breakpoints=()):
    """Nodes and d-epsilon weights on panels in x=epsilon/T.

    Keep the actual kinematic endpoints; resolve the Rayleigh--Jeans region,
    Planck peak and Wien tail separately. No thermal support is discarded.
    """
    if not 0 < lower < upper:
        return np.empty(0), np.empty(0)
    lo, hi = lower/temperature, upper/temperature
    landmarks = np.array([1e-8, 1e-6, 1e-4, .01, .1, .3, 1., 3., 10., 30., 100., 200.])
    landmarks = np.r_[landmarks, np.asarray(breakpoints)/temperature]
    cuts = np.unique(np.r_[lo, landmarks[(landmarks > lo) & (landmarks < hi)], hi])
    a, b = np.log(cuts[:-1]), np.log(cuts[1:])
    nodes, weights = gauss_rule(order)
    x = np.exp((b-a)[:,None]*nodes/2+(a+b)[:,None]/2)
    return (temperature*x).ravel(), (temperature*x*(b-a)[:,None]/2*weights).ravel()


def thermal_integral(function, lower, upper, temperature, order):
    nodes, weights = thermal_quadrature(lower, upper, temperature, order)
    return float(np.dot(weights, function(nodes))) if len(nodes) else 0.0
