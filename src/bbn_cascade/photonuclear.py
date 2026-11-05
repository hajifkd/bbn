"""Independent transcription of astro-ph/0211258 Appendix A, reactions 1--17.

No external cascade implementation is used. Energies at the API are GeV.
The published expressions internally use MeV and mb. Reaction 7 includes
1808.09324 section 3.2's 20.7 mb normalization. Reactions 12 and 15 use
1403.5995 equations III.5/III.7 and III.8, respectively.
"""
from dataclasses import dataclass

import numpy as np

from .constants import MEV, MB_TO_GEV_MINUS2


@dataclass(frozen=True)
class PhotonuclearReaction:
    id: int
    name: str
    threshold_MeV: float
    references: str
    note: str = ""


_NAMES = (
    "D(gamma,n)p", "T(gamma,n)D", "T(gamma,np)n", "He3(gamma,p)D",
    "He3(gamma,np)p", "He4(gamma,p)T", "He4(gamma,n)He3",
    "He4(gamma,D)D", "He4(gamma,np)D", "Li6(gamma,np)He4",
    "Li6(gamma,X)A3", "Li7(gamma,T)He4", "Li7(gamma,n)Li6",
    "Li7(gamma,2np)He4", "Be7(gamma,He3)He4", "Be7(gamma,p)Li6",
    "Be7(gamma,2pn)He4",
)
_THRESHOLDS = (2.224573, 6.257248, 8.481821, 5.493485, 7.718058,
               19.813852, 20.577615, 23.846527, 26.0711, 3.698892,
               15.794685, 2.467032, 7.249962, 10.948850, 1.586627,
               5.605794, 9.30468)
REACTIONS = tuple(
    PhotonuclearReaction(i, name, q, "astro-ph/0211258 Appendix A reaction " + str(i)
                         + ({7: "; 1808.09324 section 3.2", 12: "; replaced by 1403.5995 III.5/III.7", 15: "; replaced by 1403.5995 III.8"}.get(i, "")),
                         {11: "Inclusive A=3 production; daughter multiplicities unspecified.",
                          13: "Ecm interpreted literally as E_gamma-Q in the printed resonance term.",
                          15: "s+d wave fit; E_G=(5.19)^2 MeV consistent with rounded Coulomb exponent; high-energy extrapolation.",
                          17: "Isospin-symmetry estimate, not measured Be7 data."}.get(i, ""))
    for i, (name, q) in enumerate(zip(_NAMES, _THRESHOLDS), 1)
)
_POWER = {2: (9.8, 1.95, 1.65), 3: (26., 2.6, 2.3),
          4: (8.88, 1.75, 1.65), 5: (16.7, 1.95, 2.3),
          6: (19.5, 3.5, 1.), 7: (20.7, 3.5, 1.),
          8: (10.7, 10.2, 3.4), 9: (21.7, 4., 3.),
          10: (104., 2.3, 4.7), 14: (122., 4., 3.), 17: (133., 4., 3.)}


def cross_section(reaction_id, energy_GeV):
    """Cross section in GeV^-2 for 0 <= energy <= 1 GeV.

    The 1 GeV upper bound is the chosen integration cutoff, not a statement
    of experimental validity. The fits extrapolate beyond measured energies.
    """
    if reaction_id not in range(1, 18):
        raise ValueError("reaction_id must be between 1 and 17")
    energy = np.asarray(energy_GeV, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("energy must be finite and nonnegative")
    if np.any(energy > 1.0):
        raise ValueError("photonuclear fit evaluation is limited to 1 GeV")
    q = _THRESHOLDS[reaction_id - 1]
    output = np.zeros_like(energy)
    mask = energy > q * MEV
    e = energy[mask] / MEV
    x = e - q

    def power(c, a, b):
        # Ratios avoid overflow at TeV energies and are algebraically identical.
        return c * (q/e)**a * (x/e)**b

    if reaction_id in _POWER:
        value = power(*_POWER[reaction_id])
    elif reaction_id == 1:
        z = np.sqrt(q*x)/e
        value = 18.75*(z**3 + .007947*z**2*(np.sqrt(q)-np.sqrt(.037))**2/(x+.037))
    elif reaction_id == 11:
        value = power(38.1, 3., 2.) * sum(
            c*np.exp(-.5*((e-mu)/s)**2)
            for c, mu, s in ((3.7, 19., 3.5), (2.75, 30., 3.), (2.2, 43., 5.)))
    elif reaction_id == 12:
        low = 80.3/e**2*np.exp(-2.60/np.sqrt(x)-2.056*x) * (
            1+2.2875*x**2-1.1798*x**3+2.5279*x**4)
        high = 48.1/e**2*np.exp(-2.60/np.sqrt(x))
        value = np.where(x <= 1., low, high)
    elif reaction_id == 13:
        value = power(.176, 1.51, .49) + power(1205., 5.5, 5.) + .06/(1+((x-7.46)/.188)**2)
    elif reaction_id == 15:
        gamow = 5.19**2  # MeV, matching the printed rounded exponent
        d_wave = (1+4*np.pi**2*x/gamow)*(1+16*np.pi**2*x/gamow)
        sfactor = sum(qi/(x+qi)*(s0*(1+a*x)**2+s2*d_wave)
                      for qi, s0, s2, a in ((q, .406, .007, -.207),
                                            (1.1570, .163, .004, -.134)))
        value = 801/e**2*np.exp(-5.19/np.sqrt(x))*sfactor
    elif reaction_id == 16:
        value = power(32.6, 10., 2.) + power(2.27e6, 8.8335, 13.)
    output[mask] = value * MB_TO_GEV_MINUS2
    return float(output) if output.ndim == 0 else output
