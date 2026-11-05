"""Decay normalization and conservative photonuclear network primitives.

Y_X is the undecayed n_X / total standard-background entropy density.
Reaction 11 is deliberately excluded: its inclusive daughter model is unresolved.
"""
from dataclasses import dataclass
import numpy as np

HBAR = 6.582119569e-25
SPECIES = ('n', 'p', 'D', 'T', 'He3', 'He4', 'Li6', 'Li7', 'Be7')
MASS_NUMBERS = np.array([1, 1, 2, 3, 3, 4, 6, 7, 7])
CHARGES = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4])
# Repeated daughters encode multiplicities, including free nucleons.
CHANNELS = {
 1: ('D', ('n','p')), 2: ('T', ('n','D')), 3: ('T', ('n','n','p')),
 4: ('He3', ('p','D')), 5: ('He3', ('n','p','p')),
 6: ('He4', ('p','T')), 7: ('He4', ('n','He3')),
 8: ('He4', ('D','D')), 9: ('He4', ('n','p','D')),
 10: ('Li6', ('n','p','He4')), 12: ('Li7', ('T','He4')),
 13: ('Li7', ('n','Li6')), 14: ('Li7', ('n','n','p','He4')),
 15: ('Be7', ('He3','He4')), 16: ('Be7', ('p','Li6')),
 17: ('Be7', ('p','p','n','He4')),
}

@dataclass(frozen=True)
class PhotonDecay:
    mass_GeV: float
    lifetime_s: float
    yield_X: float

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.mass_GeV, self.lifetime_s, self.yield_X)):
            raise ValueError('Decay parameters must be finite')
        if self.mass_GeV <= 0 or self.lifetime_s <= 0 or self.yield_X < 0:
            raise ValueError('Positive mass/lifetime and nonnegative yield required')

    def injection_GeV5(self, time_s, entropy_GeV3):
        t, s = np.broadcast_arrays(time_s, entropy_GeV3)
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(s)) or np.any(t < 0) or np.any(s < 0):
            raise ValueError('Time and entropy must be finite and nonnegative')
        return self.mass_GeV * self.yield_X * s * np.exp(-t/self.lifetime_s) * HBAR/self.lifetime_s


def photonuclear_matrix(rates_per_s):
    """Return M for dy/dt=M@y; rates ordered as published reactions 1..17.

    Reaction 11 is omitted from both destruction and production.
    """
    rates = np.asarray(rates_per_s, dtype=float)
    if rates.shape != (17,) or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError('Expected 17 finite nonnegative per-target rates')
    matrix = np.zeros((9, 9))
    for rid, (target, daughters) in CHANNELS.items():
        j = SPECIES.index(target)
        matrix[j, j] -= rates[rid-1]
        for daughter in daughters:
            matrix[SPECIES.index(daughter), j] += rates[rid-1]
    return matrix
