from dataclasses import dataclass
from math import ceil, isfinite, log10

from .constants import MEV

DEFAULT_POINTS_PER_DECADE = 120.0


@dataclass(frozen=True)
class CascadeConfig:
    temperature: float
    maximum_energy: float
    minimum_energy: float = MEV
    eta_baryon: float = 3.0e-10
    helium_mass_fraction: float = 0.0
    grid_size: int | None = None
    quadrature_order: int = 20
    inverse_compton_scheme: str = "hybrid-fp"
    soft_fraction: float = 0.05
    include_energy_diffusion: bool = True
    integration_rtol: float = 1e-4
    background_cutoff_over_temperature: float = 200.0
    mode: str = "transport"
    points_per_decade: float = DEFAULT_POINTS_PER_DECADE

    def __post_init__(self) -> None:
        if self.temperature <= 0 or not 0 < self.minimum_energy < self.maximum_energy:
            raise ValueError("energies and temperature must be positive, with Emin < Emax")
        if not isfinite(self.minimum_energy) or not isfinite(self.maximum_energy):
            raise ValueError("energy bounds must be finite")
        if not isfinite(self.points_per_decade) or self.points_per_decade <= 0:
            raise ValueError("points_per_decade must be finite and positive")
        if self.grid_size is None:
            decades = log10(self.maximum_energy) - log10(self.minimum_energy)
            object.__setattr__(self, "grid_size", max(40, ceil(self.points_per_decade * decades) + 1))
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

