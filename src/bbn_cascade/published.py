"""Eqs. (15)-(16) and Table 1 of astro-ph/9412055."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import EV, M_E

TEMPERATURES_EV = np.array([1.0, 10.0, 100.0])
ENERGIES_GEV = np.array([10.0, 100.0, 1e3, 1e4])
P_LOW = np.array([[-1.56, -1.56, -1.56, -1.57], [-1.33, -1.33, -1.34, -1.34], [-1.22, -1.22, -1.22, -1.22]])
N_LOW = np.array([[1.4e8, 1.4e8, 1.4e8, 1.6e8], [4.5e8, 4.7e8, 4.9e8, 5.4e8], [1.3e9, 1.3e9, 1.4e9, 1.7e9]])
P_PP = np.array([[-5.01, -5.01, -5.07, -5.10], [-5.12, -5.15, -5.17, -5.20], [-4.77, -4.74, -4.79, -4.84]])
N_PP = np.array([[5.7e-18, 5.7e-18, 6.2e-18, 6.9e-18], [5.5e-18, 5.3e-18, 5.5e-18, 6.0e-18], [9.6e-18, 1.1e-17, 1.0e-17, 1.1e-17]])


def table_parameters(temperature, injection_energy):
    axes = (np.log10(TEMPERATURES_EV), np.log10(ENERGIES_GEV))
    point = np.log10([temperature / EV, injection_energy])
    def get(values):
        value = RegularGridInterpolator(
            axes, values, bounds_error=False, fill_value=None
        )(point)
        return np.asarray(value).item()
    return get(P_LOW), get(N_LOW), get(P_PP), get(N_PP)


def published_spectrum(energy, temperature, injection_energy, injection_rate=1.0):
    e = np.asarray(energy)
    pl, nl, pp, npp = table_parameters(temperature, injection_energy)
    break_energy = M_E**2 / (80 * temperature)
    cutoff = M_E**2 / (22 * temperature)
    low = nl * injection_rate * temperature**-3 * e**pl
    high = npp * injection_rate * temperature**-6 * e**pp
    spectrum = np.where(e < break_energy, low, high)
    return spectrum * np.exp(-20 * np.maximum(e / cutoff - 1, 0))
