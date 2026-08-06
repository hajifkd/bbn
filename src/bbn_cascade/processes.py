"""Cross sections/rates used by the cascade solver (natural units)."""

import numpy as np

from .constants import ALPHA, M_E, PI, R_E, photon_number_density


def klein_nishina_total(energy):
    """Total Compton cross section on an electron at rest [GeV^-2]."""
    e = np.asarray(energy, dtype=float)
    x = 2.0 * e / M_E
    out = np.empty_like(x)
    small = x < 1e-3
    out[small] = 8 * PI * R_E**2 / 3 * (1 - x[small])
    z = x[~small]
    out[~small] = 2 * PI * R_E**2 / z * (
        (1 - 4 / z - 8 / z**2) * np.log1p(z)
        + 0.5 + 8 / z - 1 / (2 * (1 + z) ** 2)
    )
    return np.maximum(out, 0.0)


def nuclear_pair_total(energy, charge=1.0):
    """Maximon cross section, paper Eqs. (34)--(36), through k^-6."""
    k = np.asarray(energy, dtype=float) / M_E
    value = np.zeros_like(k)
    near = (k >= 2) & (k < 4)
    kn = k[near]
    rho = (2 * kn - 4) / (kn + 2 + 2 * np.sqrt(2 * kn))
    value[near] = (2 * PI / 3) * ((kn - 2) / kn) ** 3 * (
        1 + rho / 2 + 23 * rho**2 / 40 + 11 * rho**3 / 60 + 29 * rho**4 / 960
    )
    high = k >= 4
    kh = k[high]
    log2k = np.log(2 * kh)
    value[high] = 28 * log2k / 9 - 218 / 27
    value[high] += (2 / kh) ** 2 * (
        2 * log2k**3 / 3 - log2k**2 + (6 - PI**2 / 3) * log2k
        + 2 * 1.202056903159594 + PI**2 / 6 - 3.5
    )
    value[high] -= (2 / kh) ** 4 * (3 * log2k / 16 + 1 / 8)
    value[high] -= (2 / kh) ** 6 * (29 * log2k / 2304 - 77 / 13824)
    return charge**2 * ALPHA * R_E**2 * np.maximum(value, 0)


def photon_photon_scattering_rate(energy, temperature):
    """Photon-photon scattering loss rate based on paper Eq. (31)."""
    e = np.asarray(energy, dtype=float)
    # int eps^3 f_BB(eps) d eps = 8 pi^4 T^6/(63 pi^2)
    moment = 8 * PI**4 * temperature**6 / 63
    rate = 1946 / (50625 * PI) * ALPHA**2 * R_E**2 * M_E**-6 * e**3 * moment
    return np.where(e <= M_E**2 / temperature, rate, 0)


def double_pair_rate_approx(energy, temperature):
    """Smooth thermal gamma-gamma pair rate with the paper's E_c."""
    e = np.asarray(energy, dtype=float)
    critical = M_E**2 / (22 * temperature)
    turn_on = 1 / (1 + np.exp(np.clip(8 * (critical / e - 1), -700, 700)))
    return photon_number_density(temperature) * 0.2 * PI * R_E**2 * turn_on


def background_target_densities(temperature, eta_baryon, helium_mass_fraction=0.0):
    nb = eta_baryon * photon_number_density(temperature)
    n_h = (1 - helium_mass_fraction) * nb
    n_he = helium_mass_fraction * nb / 4
    return n_h + 2 * n_he, n_h + 4 * n_he


def loss_rates(energy, temperature, eta_baryon, helium_mass_fraction=0.0):
    n_e, n_z2 = background_target_densities(
        temperature, eta_baryon, helium_mass_fraction
    )
    return {
        "double_pair": double_pair_rate_approx(energy, temperature),
        "photon_photon": photon_photon_scattering_rate(energy, temperature),
        "compton": n_e * klein_nishina_total(energy),
        "nuclear_pair": n_z2 * nuclear_pair_total(energy),
    }

