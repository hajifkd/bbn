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
                photon = (upper - lower) * nodes / 2 + (upper + lower) / 2
                photon_weight = weights * (upper - lower) / 2
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
                delta = photon - epsilon
                total_loss[parent] += np.sum(rate_weight)
                if is_soft:
                    first_moment = rate_weight * delta
                    drift[parent] += np.sum(first_moment)
                    diffusion[parent] += np.sum(first_moment * delta)
                    soft_energy[parent] += np.sum(first_moment)
                    photon_target = soft_photon
                else:
                    hard_loss[parent] += np.sum(rate_weight)
                    photon_target = hard_photon

                for emitted, event_rate in zip(photon, rate_weight):
                    child = np.searchsorted(boundaries, emitted, side="right") - 1
                    if 0 <= child < n:
                        photon_target[child, parent] += event_rate
                if not is_soft:
                    for loss, event_rate in zip(delta, rate_weight):
                        outgoing = electron - loss
                        child = (
                            np.searchsorted(boundaries, outgoing, side="right") - 1
                        )
                        if 0 <= child < n:
                            hard_electron[child, parent] += event_rate

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
