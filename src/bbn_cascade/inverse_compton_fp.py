"""Hybrid Boltzmann/Fokker--Planck inverse-Compton collision integrals."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .constants import ALPHA, M_E, PI
from .kernels import jones_function


@dataclass(frozen=True)
class InverseComptonMoments:
    drift: np.ndarray
    diffusion: np.ndarray
    hard_loss: np.ndarray
    soft_photon_kernel: np.ndarray
    hard_photon_kernel: np.ndarray
    hard_electron_kernel: np.ndarray
    total_loss: np.ndarray
    soft_energy_from_photons: np.ndarray


def _energy_boundaries(energy):
    boundaries = np.empty(len(energy) + 1)
    boundaries[1:-1] = np.sqrt(energy[:-1] * energy[1:])
    boundaries[0], boundaries[-1] = energy[0], energy[-1]
    return boundaries


def build_inverse_compton_moments(
    energy,
    temperature,
    soft_fraction=0.05,
    quadrature_order=20,
    background_cutoff_over_temperature=200.0,
):
    """Integrate Jones' rate and split each event by fractional energy loss.

    Every output quantity is accumulated from the same quadrature points, so
    the soft photon energy moment and the FP drift agree up to roundoff.
    Kernel arrays are bin-integrated transition rates [GeV].
    """
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
    eps_max_global = background_cutoff_over_temperature * temperature
    for parent, electron in enumerate(energy):
        kinetic = electron - M_E
        if kinetic <= 0:
            continue
        eps_max = min(electron - M_E**2 / (4 * electron), eps_max_global)
        if eps_max <= eps_min:
            continue
        a, b = np.log(eps_min), np.log(eps_max)
        log_eps = (b - a) * nodes / 2 + (a + b) / 2
        eps_values = np.exp(log_eps)
        eps_weights = weights * (b - a) / 2 * eps_values
        for epsilon, eps_weight in zip(eps_values, eps_weights):
            photon_max = 4 * epsilon * electron**2 / (
                M_E**2 + 4 * epsilon * electron
            )
            if photon_max <= epsilon:
                continue
            photon = (photon_max - epsilon) * nodes / 2 + (
                photon_max + epsilon
            ) / 2
            photon_weights = weights * (photon_max - epsilon) / 2
            F = jones_function(photon, electron, np.full_like(photon, epsilon))
            planck = epsilon / (PI**2 * np.expm1(epsilon / temperature))
            event_weights = (
                prefactor
                / electron**2
                * eps_weight
                * photon_weights
                * planck
                * F
            )
            delta = photon - epsilon
            fraction = delta / kinetic
            soft = fraction < soft_fraction
            drift[parent] += np.sum(event_weights[soft] * delta[soft])
            diffusion[parent] += np.sum(event_weights[soft] * delta[soft] ** 2)
            soft_energy[parent] += np.sum(event_weights[soft] * delta[soft])
            hard_loss[parent] += np.sum(event_weights[~soft])
            total_loss[parent] += np.sum(event_weights)

            for emitted, weight, is_soft in zip(photon, event_weights, soft):
                child = np.searchsorted(boundaries, emitted, side="right") - 1
                if 0 <= child < n:
                    target = soft_photon if is_soft else hard_photon
                    target[child, parent] += weight
            for loss, weight, is_soft in zip(delta, event_weights, soft):
                if is_soft:
                    continue
                outgoing = electron - loss
                child = np.searchsorted(boundaries, outgoing, side="right") - 1
                if 0 <= child < n:
                    hard_electron[child, parent] += weight

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


def chang_cooper_weight(drift_velocity, diffusion, spacing):
    """Chang--Cooper delta for J=A f-B df/dE, with A=-b and B=D/2."""
    if diffusion <= 0:
        return 1.0 if drift_velocity < 0 else 0.0
    w = drift_velocity * spacing / diffusion
    if abs(w) < 1e-5:
        return 0.5 - w / 12 + w**3 / 720
    if w > 700:
        return 1 / w
    if w < -700:
        return 1 + 1 / w
    return 1 / w - 1 / np.expm1(w)
