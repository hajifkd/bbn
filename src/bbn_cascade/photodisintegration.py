"""Finite-volume photonuclear convolution; rates are per target nucleus."""
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from .constants import HBAR_GEV_S, MEV
from .photonuclear import REACTIONS, cross_section


@dataclass(frozen=True)
class PhotodisintegrationRate:
    reaction_id: int
    reaction: str
    threshold_GeV: float
    upper_GeV: float
    rate_GeV: float
    rate_per_s: float
    response_GeV_minus4: float
    quadrature_error_GeV: float
    above_100MeV_rate_GeV: float


def calculate_rates(spectrum, *, maximum_energy=1.0, rtol=1e-7):
    """Integrate cell-constant dn/dE up to min(1 GeV, spectrum upper edge).

    No extra injection-line term is added: the solver includes that population.
    100 MeV is a tail reporting boundary, not a validated fit limit.
    """
    if not np.isfinite(rtol) or rtol <= 0 or not 0 < maximum_energy <= 1:
        raise ValueError("rtol must be positive and 0 < maximum_energy <= 1 GeV")
    edges = np.asarray(spectrum.boundaries, dtype=float)
    photons = np.asarray(spectrum.photon, dtype=float)
    if photons.ndim != 1 or edges.shape != (len(photons)+1,) or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("explicit, finite, increasing cell boundaries are required")
    if np.any(~np.isfinite(photons)) or np.any(photons < 0):
        raise ValueError("photon density must be finite and nonnegative")
    qinj = spectrum.diagnostics.get("injected_energy_rate")
    if qinj is None or not np.isfinite(qinj) or qinj <= 0:
        raise ValueError("positive injected_energy_rate in GeV^5 is required")
    rows = []
    limit = min(edges[-1], maximum_energy)
    for reaction in REACTIONS:
        threshold = reaction.threshold_MeV * MEV
        if edges[0] > min(threshold, limit):
            raise ValueError("spectrum lower boundary leaves part of the integral uncovered")
        total, error, tail = 0., 0., 0.
        # Gaussian centers, narrow Li7 resonance, Li7 branch switch, tail boundary.
        knots = [.019, .030, .043, .007249962+.00746, .003467032, .1]
        for lo, hi, density in zip(edges[:-1], edges[1:], photons):
            a, b = max(lo, threshold), min(hi, limit)
            if b <= a or density == 0:
                continue
            cuts = [a] + sorted(k for k in knots if a < k < b) + [b]
            for left, right in zip(cuts[:-1], cuts[1:]):
                value, err = quad(lambda e: cross_section(reaction.id, e), left, right,
                                  epsabs=1e-30, epsrel=rtol, limit=150)
                total += density * value
                error += density * err
                if left >= .1:
                    tail += density * value
        rows.append(PhotodisintegrationRate(
            reaction.id, reaction.name, threshold, limit, total, total/HBAR_GEV_S,
            total/qinj, error, tail))
    return rows


def write_rates(output, rows, metadata):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'rates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    record = dict(metadata, reactions=[asdict(r) for r in REACTIONS],
                  references=['https://arxiv.org/abs/astro-ph/0211258',
                              'https://arxiv.org/abs/1808.09324', 'https://arxiv.org/abs/1403.5995'],
                  normalization='dn_gamma/dE: GeV^2; Q: GeV^5; rate per target; K=rate/Q: GeV^-4',
                  extrapolation='1 GeV cutoff; 100 MeV tail is a diagnostic, not an experimental validity boundary',
                  reaction13='Printed Ecm in resonance interpreted as E_gamma-Q')
    (out/'metadata.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
