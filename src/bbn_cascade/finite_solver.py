"""Stationary conservative finite-volume cascade solver."""

from dataclasses import dataclass

import numpy as np

from .config import CascadeConfig
from .finite_volume import FiniteVolumeCollision
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
        data = np.column_stack(
            [self.energy, self.photon, self.electron]
            + [self.rates[name] for name in names]
        )
        np.savetxt(
            path,
            data,
            delimiter=",",
            header=",".join(["energy_GeV", "photon_GeV2", "electron_GeV2"] + names),
            comments="",
        )


class CascadeSolver:
    """Solve the coupled photon/electron Boltzmann equation by finite volumes."""

    def __init__(self, config: CascadeConfig):
        self.config = config

    def solve(self, source=None):
        source = source or MonochromaticPhotonSource(self.config.maximum_energy)
        if not np.isclose(source.energy, self.config.maximum_energy):
            raise ValueError("source energy must equal config.maximum_energy")
        energy = np.geomspace(
            self.config.minimum_energy,
            self.config.maximum_energy,
            self.config.grid_size,
        )
        if self.config.mode == "published-fit":
            rates = loss_rates(
                energy,
                self.config.temperature,
                self.config.eta_baryon,
                self.config.helium_mass_fraction,
            )
            photon = published_spectrum(
                energy,
                self.config.temperature,
                source.energy,
                source.energy_injection_rate,
            )
            return CascadeSpectrum(
                energy, photon, np.zeros_like(photon), rates, {"published_fit": 1.0}
            )

        system = FiniteVolumeCollision(
            self.config.temperature,
            self.config.eta_baryon,
            self.config.helium_mass_fraction,
            self.config.quadrature_order,
        ).build_finite_volume(energy)
        return self._solve_system(energy, system, source)

    def _solve_system(self, energy, system, source):
        n = len(energy)
        loss = np.maximum(
            np.vstack((system.photon_loss, system.electron_loss)),
            np.finfo(float).tiny,
        )
        populations = np.zeros((2, n))
        mono = np.array([source.number_injection_rate, 0.0])
        for i in range(n - 1, -1, -1):
            matrix = np.diag(loss[:, i]) - system.transitions[:, :, i, i]
            incoming = mono.copy() if i == n - 1 else np.zeros(2)
            if i + 1 < n:
                incoming += np.einsum(
                    "abj,bj->a",
                    system.transitions[:, :, i, i + 1 :],
                    populations[:, i + 1 :],
                )
            populations[:, i] = np.linalg.solve(matrix, incoming)

        widths = np.diff(system.boundaries)
        spectra = populations / widths
        negative = float(np.count_nonzero(spectra < 0))
        spectra = np.maximum(spectra, 0)
        diagnostics = {
            "injected_energy_rate": source.energy_injection_rate,
            "kernel_quadrature_order": float(self.config.quadrature_order),
            "negative_bins_clipped": negative,
            "finite_volume": 1.0,
        }
        return CascadeSpectrum(
            energy, spectra[0], spectra[1], dict(system.components), diagnostics
        )
