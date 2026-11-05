"""Natural-unit constants; all energies are in GeV."""

import numpy as np

ALPHA = 1.0 / 137.035999084
M_E = 0.51099895069e-3
R_E = ALPHA / M_E
ZETA3 = 1.202056903159594
PI = np.pi
EV = 1.0e-9
MEV = 1.0e-3
TEV = 1.0e3
HBAR_GEV_S = 6.582119569e-25
MB_TO_GEV_MINUS2 = 1.0 / 0.3893793721


def photon_number_density(temperature: float) -> float:
    return 2.0 * ZETA3 * temperature**3 / PI**2

