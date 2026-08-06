from dataclasses import dataclass


@dataclass(frozen=True)
class MonochromaticPhotonSource:
    energy: float
    energy_injection_rate: float = 1.0

    @property
    def number_injection_rate(self) -> float:
        return self.energy_injection_rate / self.energy
