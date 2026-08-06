"""Bin-integrated, conservative representation of the collision kernels."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .constants import ALPHA, M_E, PI, R_E
from .kernels import (
    CollisionIntegrals,
    bethe_heitler_differential,
    compton_kernel,
)


@dataclass(frozen=True)
class FiniteVolumeSystem:
    photon_loss: np.ndarray
    electron_loss: np.ndarray
    transitions: np.ndarray
    components: dict[str, np.ndarray]
    boundaries: np.ndarray


class FiniteVolumeCollision(CollisionIntegrals):
    """Integrate K_ab(E,E') over every outgoing energy cell."""

    def _differential(self, outgoing, incoming):
        result = np.zeros((2, 2))
        ratio = outgoing / incoming
        result[0, 0] = (
            1112
            / (10125 * PI)
            * ALPHA**2
            * R_E**2
            * M_E**-6
            * incoming**2
            * (1 - ratio + ratio**2) ** 2
            * self.moment
        )
        result[0, 0] += compton_kernel(outgoing, incoming, self.ne)
        result[0, 1] = self.inverse_compton_photon(outgoing, incoming)
        result[1, 0] = compton_kernel(
            incoming + M_E - outgoing, incoming, self.ne
        )
        result[1, 0] += (
            2 * self.nz2 * bethe_heitler_differential(outgoing, incoming)
        )
        result[1, 0] += self.double_pair_electron(outgoing, incoming)
        result[1, 1] = self.inverse_compton_electron(outgoing, incoming)
        return result

    def build_finite_volume(self, energy):
        energy = np.asarray(energy, dtype=float)
        point_system = super().build(energy)
        n = len(energy)
        boundaries = np.empty(n + 1)
        boundaries[1:-1] = np.sqrt(energy[:-1] * energy[1:])
        boundaries[0], boundaries[-1] = energy[0], energy[-1]
        transitions = np.zeros((2, 2, n, n))
        nodes, weights = leggauss(max(6, self.order // 2))

        for parent, incoming in enumerate(energy):
            for child in range(parent + 1):
                low = boundaries[child]
                high = min(boundaries[child + 1], incoming)
                if high <= low:
                    continue
                a, b = np.log(low), np.log(high)
                points = np.exp((b - a) * nodes / 2 + (a + b) / 2)
                for point, weight in zip(points, weights):
                    transitions[:, :, child, parent] += (
                        weight * point * self._differential(point, incoming)
                    )
                transitions[:, :, child, parent] *= (b - a) / 2

        return FiniteVolumeSystem(
            point_system.photon_loss,
            point_system.electron_loss,
            transitions,
            point_system.components,
            boundaries,
        )
