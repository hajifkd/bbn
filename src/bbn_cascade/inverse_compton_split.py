"""IC moment integration with explicit soft/hard quadrature domains."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .constants import ALPHA, M_E, PI
from .inverse_compton_fp import InverseComptonMoments, _energy_boundaries
from .kernels import jones_function


def build_inverse_compton_moments(
    energy,
    temperature,
    soft_fraction=0.05,
    quadrature_order=20,
    background_cutoff_over_temperature=200.0,
):
    energy = np.asarray(energy, dtype=float)
    n = len(energy)
    boundaries = _energy_boundaries(energy)
    drift = np.zeros(n)
    diffusion = np.zeros(n)
    hard_loss = np.zeros(n)
    total_loss = np.zeros(n)
    soft_energy = np.zeros(n)
    soft_photon = np.zeros((n, n))
    hard_photon = np.zeros((n, n))
    hard_electron = np.zeros((n, n))
    nodes, weights = leggauss(quadrature_order)
    prefactor = 2 * PI * ALPHA**2
    eps_min = temperature * 1e-8
    eps_global_max = background_cutoff_over_temperature * temperature

    for parent, electron in enumerate(energy):
        kinetic = electron - M_E
        if kinetic <= 0:
            continue
        eps_max = min(electron - M_E**2 / (4 * electron), eps_global_max)
        if eps_max <= eps_min:
            continue
        log_min, log_max = np.log(eps_min), np.log(eps_max)
        log_eps = (log_max - log_min) * nodes / 2 + (log_max + log_min) / 2
        eps_values = np.exp(log_eps)
        eps_weights = weights * (log_max - log_min) / 2 * eps_values
        for epsilon, eps_weight in zip(eps_values, eps_weights):
            photon_max = 4 * epsilon * electron**2 / (
                M_E**2 + 4 * epsilon * electron
            )
            if photon_max <= epsilon:
                continue
            split = min(epsilon + soft_fraction * kinetic, photon_max)
            planck = epsilon / (PI**2 * np.expm1(epsilon / temperature))
            for lower, upper, is_soft in (
                (epsilon, split, True),
                (split, photon_max, False),
            ):
                if upper <= lower:
                    continue
                # Integrate separately inside every outgoing-energy cell.
                # Hard events are also split at mapped electron-cell faces.
                cuts = [lower, upper]
                cuts.extend(boundaries[(boundaries > lower) & (boundaries < upper)])
                if not is_soft:
                    electron_cuts = electron + epsilon - boundaries
                    cuts.extend(electron_cuts[(electron_cuts > lower) & (electron_cuts < upper)])
                cuts = np.unique(cuts)
                photon_target = soft_photon if is_soft else hard_photon

                for segment_low, segment_high in zip(cuts[:-1], cuts[1:]):
                    if segment_high <= segment_low:
                        continue
                    photon = (
                        (segment_high - segment_low) * nodes / 2
                        + (segment_high + segment_low) / 2
                    )
                    photon_weight = weights * (segment_high - segment_low) / 2
                    rate_weight = (
                        prefactor
                        / electron**2
                        * eps_weight
                        * photon_weight
                        * planck
                        * jones_function(
                            photon, electron, np.full_like(photon, epsilon)
                        )
                    )
                    event_rate = np.sum(rate_weight)
                    delta = photon - epsilon
                    total_loss[parent] += event_rate
                    if is_soft:
                        first_moment = rate_weight * delta
                        drift[parent] += np.sum(first_moment)
                        diffusion[parent] += np.sum(first_moment * delta)
                        soft_energy[parent] += np.sum(first_moment)
                    else:
                        hard_loss[parent] += event_rate

                    midpoint = 0.5 * (segment_low + segment_high)
                    photon_child = np.searchsorted(
                        boundaries, midpoint, side="right"
                    ) - 1
                    if 0 <= photon_child < n:
                        photon_target[photon_child, parent] += event_rate
                    if not is_soft:
                        outgoing = electron + epsilon - midpoint
                        electron_child = np.searchsorted(
                            boundaries, outgoing, side="right"
                        ) - 1
                        if 0 <= electron_child < n:
                            hard_electron[electron_child, parent] += event_rate

    return InverseComptonMoments(
        drift,
        diffusion,
        hard_loss,
        soft_photon,
        hard_photon,
        hard_electron,
        total_loss,
        soft_energy,
    )
