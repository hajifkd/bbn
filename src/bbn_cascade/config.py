from dataclasses import dataclass

from .constants import MEV


@dataclass(frozen=True)
class CascadeConfig:
    temperature: float
    maximum_energy: float
    minimum_energy: float = MEV
    eta_baryon: float = 3.0e-10
    helium_mass_fraction: float = 0.0
    grid_size: int = 400
    quadrature_order: int = 20
    inverse_compton_scheme: str = "hybrid-fp"
    soft_fraction: float = 0.05
    include_energy_diffusion: bool = True
    integration_rtol: float = 1e-4
    background_cutoff_over_temperature: float = 200.0
    mode: str = "transport"

    def __post_init__(self) -> None:
        if self.temperature <= 0 or not 0 < self.minimum_energy < self.maximum_energy:
            raise ValueError("energies and temperature must be positive, with Emin < Emax")
        if self.grid_size < 40 or self.eta_baryon <= 0:
            raise ValueError("grid_size >= 40 and eta_baryon > 0 are required")
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8")
        if self.inverse_compton_scheme not in {"boltzmann", "hybrid-fp"}:
            raise ValueError("inverse_compton_scheme must be 'boltzmann' or 'hybrid-fp'")
        if not 0 < self.soft_fraction < 1:
            raise ValueError("soft_fraction must be in (0, 1)")
        if self.integration_rtol <= 0:
            raise ValueError("integration_rtol must be positive")
        if self.background_cutoff_over_temperature <= 0:
            raise ValueError("background cutoff must be positive")
        if not 0 <= self.helium_mass_fraction <= 1:
            raise ValueError("helium_mass_fraction must be in [0, 1]")
        if self.mode not in {"transport", "published-fit"}:
            raise ValueError("mode must be 'transport' or 'published-fit'")

