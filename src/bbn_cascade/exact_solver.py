"""Stationary solver using the full differential collision kernels."""

from dataclasses import dataclass

import numpy as np

from .config import CascadeConfig
from .kernels import CollisionIntegrals
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
        columns = [self.energy, self.photon, self.electron] + [self.rates[name] for name in names]
        header = ["energy_GeV", "photon_GeV2", "electron_GeV2"] + names
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


class CascadeSolver:
    """Solve the coupled photon/electron stationary Boltzmann equations."""

    def __init__(self, config: CascadeConfig):
        self.config = config

    def solve(self, source: MonochromaticPhotonSource | None = None) -> CascadeSpectrum:
        source = source or MonochromaticPhotonSource(self.config.maximum_energy)
        if not np.isclose(source.energy, self.config.maximum_energy):
            raise ValueError("source energy must equal config.maximum_energy")
        energy = np.geomspace(self.config.minimum_energy, self.config.maximum_energy, self.config.grid_size)
        if self.config.mode == "published-fit":
            rates = loss_rates(energy, self.config.temperature, self.config.eta_baryon, self.config.helium_mass_fraction)
            photon = published_spectrum(energy, self.config.temperature, source.energy, source.energy_injection_rate)
            diagnostics = {"injected_energy_rate": source.energy_injection_rate, "published_fit": 1.0}
            return CascadeSpectrum(energy, photon, np.zeros_like(photon), rates, diagnostics)

        system = CollisionIntegrals(
            self.config.temperature,
            self.config.eta_baryon,
            self.config.helium_mass_fraction,
            self.config.quadrature_order,
        ).build(energy)
        return self._solve_stationary(energy, system, source)

    def _solve_stationary(self, energy, system, source):
        """Paper Eqs. (8)--(14) with trapezoidal endpoint corrections."""
        n = len(energy)
        dy = np.log(energy[-1] / energy[0]) / (n - 1)
        kernels = system.kernels
        losses = np.maximum(
            np.vstack((system.photon_loss, system.electron_loss)),
            np.finfo(float).tiny,
        )
        spectra = np.zeros((2, n))
        mono = np.array([source.number_injection_rate, 0.0])

        spectra[:, -1] = mono / losses[:, -1]
        spectra[1, -1] += (
            kernels[1, 0, -1, -1]
            * mono[0]
            / (losses[0, -1] * losses[1, -1])
        )

        for i in range(n - 2, -1, -1):
            matrix = np.diag(losses[:, i])
            matrix -= 0.5 * dy * energy[i] * kernels[:, :, i, i]
            incoming = kernels[:, :, i, -1] @ (mono / losses[:, -1])
            incoming += (
                0.5
                * dy
                * energy[-1]
                * (kernels[:, :, i, -1] @ spectra[:, -1])
            )
            if i + 1 < n - 1:
                weighted = spectra[:, i + 1 : n - 1] * energy[i + 1 : n - 1]
                incoming += dy * np.einsum(
                    "abj,bj->a", kernels[:, :, i, i + 1 : n - 1], weighted
                )
            spectra[:, i] = np.linalg.solve(matrix, incoming)

        negative = float(np.count_nonzero(spectra < 0))
        spectra = np.maximum(spectra, 0.0)
        diagnostics = {
            "injected_energy_rate": source.energy_injection_rate,
            "kernel_quadrature_order": float(self.config.quadrature_order),
            "negative_bins_clipped": negative,
        }
        return CascadeSpectrum(
            energy, spectra[0], spectra[1], dict(system.components), diagnostics
        )
