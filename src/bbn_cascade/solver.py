from dataclasses import dataclass

import numpy as np

from .config import CascadeConfig
from .constants import M_E, PI, R_E, photon_number_density
from .processes import loss_rates
from .published import published_spectrum
from .sources import MonochromaticPhotonSource


@dataclass(frozen=True)
class CascadeSpectrum:
    energy: np.ndarray
    photon: np.ndarray
    electron: np.ndarray
    rates: dict[str, np.ndarray]
    diagnostics: dict[str, float]

    def to_csv(self, path) -> None:
        names = list(self.rates)
        data = np.column_stack([self.energy, self.photon, self.electron] + [self.rates[n] for n in names])
        np.savetxt(path, data, delimiter=",", header=",".join(["energy_GeV", "photon_GeV2", "electron_GeV2"] + names), comments="")


class CascadeSolver:
    """Stationary, downward-in-energy cascade solver.

    ``transport`` follows particle/energy flow with paper reaction rates.
    ``published-fit`` evaluates the paper's tabulated fits and is the exact
    regression reference for Figs. 1--3.
    """

    def __init__(self, config: CascadeConfig):
        self.config = config

    def solve(self, source: MonochromaticPhotonSource | None = None) -> CascadeSpectrum:
        source = source or MonochromaticPhotonSource(self.config.maximum_energy)
        if not np.isclose(source.energy, self.config.maximum_energy):
            raise ValueError("source energy must equal config.maximum_energy")
        edges = np.geomspace(self.config.minimum_energy, self.config.maximum_energy, self.config.grid_size + 1)
        energy = np.sqrt(edges[:-1] * edges[1:])
        widths = np.diff(edges)
        rates = loss_rates(energy, self.config.temperature, self.config.eta_baryon, self.config.helium_mass_fraction)
        if self.config.mode == "published-fit":
            photon = published_spectrum(energy, self.config.temperature, source.energy, source.energy_injection_rate)
            return CascadeSpectrum(energy, photon, np.zeros_like(photon), rates, {"injected_energy_rate": source.energy_injection_rate, "deposited_energy_rate": np.nan, "energy_residual_fraction": np.nan})
        return self._transport(energy, edges, widths, rates, source)

    def _transport(self, energy, edges, widths, rates, source):
        n = len(energy)
        photon_in = np.zeros(n)
        electron_in = np.zeros(n)
        photon = np.zeros(n)
        electron = np.zeros(n)
        deposited = 0.0
        photon_in[-1] = source.number_injection_rate
        total_loss = sum(rates.values())
        total_loss = np.maximum(total_loss, np.finfo(float).tiny)

        def deposit(array, daughter_energy, number, max_index):
            nonlocal deposited
            if number <= 0 or daughter_energy <= 0:
                return
            if daughter_energy < edges[0]:
                deposited += daughter_energy * number
                return
            # Strictly downward flow makes the triangular solve deterministic.
            idx = min(np.searchsorted(edges, daughter_energy, side="right") - 1, max_index)
            if idx < 0:
                deposited += daughter_energy * number
            else:
                array[idx] += number

        n_gamma = photon_number_density(self.config.temperature)
        sigma_t = 8 * PI * R_E**2 / 3
        for j in range(n - 1, -1, -1):
            e = energy[j]
            incoming = photon_in[j]
            photon[j] = incoming / (total_loss[j] * widths[j])
            if incoming:
                for name, rate in rates.items():
                    events = incoming * rate[j] / total_loss[j]
                    if name in {"double_pair", "nuclear_pair"}:
                        kinetic = max((e - 2 * M_E) / 2, 0)
                        deposited += min(e, 2 * M_E) * events
                        deposit(electron_in, kinetic, 2 * events, j - 1)
                    elif name == "photon_photon":
                        deposit(photon_in, 0.75 * e, events, j - 1)
                        deposit(photon_in, 0.25 * e, events, j - 1)
                    else:  # Compton recoil, using the mean-angle kinematics.
                        scattered = e / (1 + e / M_E)
                        deposit(photon_in, scattered, events, j - 1)
                        deposit(electron_in, e - scattered, events, j - 1)

            ein = electron_in[j]
            if not ein:
                continue
            gamma = 1 + e / M_E
            # KN-suppressed inverse-Compton residence rate.
            ic_rate = n_gamma * sigma_t / (1 + 4 * gamma * self.config.temperature / M_E) ** 1.5
            electron[j] = ein / (max(ic_rate, np.finfo(float).tiny) * widths[j])
            emitted = min((4 / 3) * gamma**2 * 2.701 * self.config.temperature, e)
            if emitted <= 0:
                deposited += e * ein
            else:
                deposit(photon_in, emitted, ein * e / emitted, j - 1)

        residual = (source.energy_injection_rate - deposited) / source.energy_injection_rate
        diagnostics = {"injected_energy_rate": source.energy_injection_rate, "deposited_energy_rate": deposited, "energy_residual_fraction": residual}
        return CascadeSpectrum(energy, photon, electron, rates, diagnostics)
