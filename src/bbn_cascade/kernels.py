"""Full differential collision kernels of astro-ph/9412055, Eqs. (26)--(50)."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .constants import ALPHA, M_E, PI, R_E
from .quadrature import gauss_rule, thermal_integral
from .processes import background_target_densities, klein_nishina_total, nuclear_pair_total

M2 = M_E**2


def _bose_weight(x, temperature):
    z = x / temperature
    return np.where(z < 700, x / (PI**2 * np.expm1(z)), 0.0)


def _log_integral(function, lower, upper, order):
    if upper <= lower or lower <= 0 or not np.isfinite(lower):
        return 0.0
    nodes, weights = gauss_rule(order)
    a, b = np.log(lower), np.log(upper)
    lx = (b - a) * nodes / 2 + (a + b) / 2
    x = np.exp(lx)
    return (b - a) / 2 * np.sum(weights * function(x) * x)


def jones_function(photon, electron, background):
    """Jones inverse-Compton function, including its kinematic support."""
    bg = np.asarray(background, dtype=float)
    gamma = 4 * bg * electron / M2
    den = gamma * (electron - photon)
    q = np.divide(photon, den, out=np.zeros_like(bg), where=den > 0)
    valid = (photon >= bg) & (q > 0) & (q <= 1)
    out = np.zeros_like(bg)
    qq, gg = q[valid], gamma[valid]
    out[valid] = 2 * qq * np.log(qq) + (1 + 2 * qq) * (1 - qq)
    out[valid] += (gg * qq) ** 2 * (1 - qq) / (2 + 2 * gg * qq)
    return out


def pair_function(electron, photon, background):
    """Agaronyan et al. gamma-gamma pair function (paper Eq. 30)."""
    bg = np.asarray(background, dtype=float)
    positron = photon + bg - electron
    rad = np.maximum(1 - M2 / (photon * bg), 0)
    delta = (photon - bg) * np.sqrt(rad)
    low, high = (photon + bg - delta) / 2, (photon + bg + delta) / 2
    valid = (photon * bg >= M2) & (electron >= low) & (electron <= high) & (positron > M_E)
    out = np.zeros_like(bg)
    if not np.any(valid):
        return out
    x, p = bg[valid], positron[valid]
    total = electron + p
    value = 4 * total**2 * np.log(4 * x * electron * p / (M2 * total)) / (electron * p)
    value += (M2 / (x * total) - 1) * total**4 / (electron**2 * p**2)
    value += 2 * (2 * x * total - M2) * total**2 / (M2 * electron * p)
    value -= 8 * x * total / M2
    out[valid] = np.maximum(value, 0)
    return out


def bethe_heitler_differential(electron, photon):
    """d sigma/(dE Z^2) for nuclear pair creation, paper Eq. (37)."""
    em, ep = electron, photon - electron
    if em <= M_E or ep <= M_E:
        return 0.0
    pm, pp = np.sqrt(em**2 - M2), np.sqrt(ep**2 - M2)
    L = np.log((ep * em + pp * pm + M2) / (ep * em - pp * pm + M2))
    lm, lp = np.log((em + pm) / (em - pm)), np.log((ep + pp) / (ep - pp))
    value = -4 / 3 - 2 * ep * em * (pp**2 + pm**2) / (pp**2 * pm**2)
    value += M2 * (lm * ep / pm**3 + lp * em / pp**3 - lp * lm / (pp * pm))
    value += L * (-8 * ep * em / (3 * pp * pm) + photon**2 * ((ep * em) ** 2 + (pp * pm) ** 2 - M2 * ep * em) / (pp**3 * pm**3))
    value -= L * M2 * photon * (lp * (ep * em - pp**2) / pp**3 + lm * (ep * em - pm**2) / pm**3) / (2 * pp * pm)
    return max(ALPHA * R_E**2 * pp * pm / photon**3 * value, 0.0)


def compton_kernel(out_photon, in_photon, electron_density):
    """Differential Klein-Nishina kernel, paper Eqs. (42),(45)."""
    if out_photon < in_photon / (1 + 2 * in_photon / M_E) or out_photon > in_photon:
        return 0.0
    value = in_photon / out_photon + out_photon / in_photon
    value += (M_E / out_photon - M_E / in_photon) ** 2
    value -= 2 * M_E * (1 / out_photon - 1 / in_photon)
    return max(PI * R_E**2 * M_E / in_photon**2 * electron_density * value, 0.0)


@dataclass(frozen=True)
class KernelSystem:
    photon_loss: np.ndarray
    electron_loss: np.ndarray
    kernels: np.ndarray  # [out species, in species, out E, in E]
    components: dict[str, np.ndarray]


class CollisionIntegrals:
    """Numerical Planck averages and all four redistribution kernels."""

    def __init__(self, temperature, eta_baryon, helium_fraction=0.0, order=20):
        self.T, self.order = temperature, order
        self.tmax = 200 * temperature
        self.ne, self.nz2 = background_target_densities(temperature, eta_baryon, helium_fraction)
        self.moment = 8 * PI**4 * temperature**6 / 63

    def double_pair_rate(self, photon):
        lower = M2 / photon
        if lower >= self.tmax:
            return 0.0
        nodes, weights = gauss_rule(self.order)
        def over_background(bg):
            result = np.zeros_like(bg)
            for k, x in enumerate(bg):
                slo, shi = 4 * M2, 4 * photon * x
                if shi <= slo:
                    continue
                # u=sqrt(log(s/4m_e^2)) removes the square-root threshold.
                # Composite panels also resolve the broad high-energy range.
                umax = np.sqrt(np.log(shi/slo))
                cuts = np.linspace(0., umax, max(1, int(np.ceil(umax/.5)))+1)
                a, b = cuts[:-1,None], cuts[1:,None]
                u = (b-a)*nodes/2+(a+b)/2
                s = slo*np.exp(u*u)
                beta = np.sqrt(-np.expm1(-u*u))
                sigma = PI * R_E**2 / 2 * (1-beta**2) * (
                    (3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))
                result[k] = np.sum((b-a)/2*weights*2*u*s*s*sigma)
            return _bose_weight(bg, self.T) / bg * result
        return thermal_integral(over_background, lower, self.tmax, self.T, self.order) / (8 * photon**2)

    def inverse_compton_photon(self, out_energy, in_energy):
        if in_energy < out_energy + M_E:
            return 0.0
        lo = M2 * out_energy / (4 * in_energy * (in_energy - out_energy))
        hi = min(out_energy, in_energy - M2 / (4 * in_energy), self.tmax)
        integral = thermal_integral(lambda x: jones_function(out_energy, in_energy, x) * _bose_weight(x, self.T), lo, hi, self.T, self.order)
        return 2 * PI * ALPHA**2 * integral / in_energy**2

    def inverse_compton_electron(self, out_energy, in_energy):
        if out_energy >= in_energy:
            return 0.0
        p = M2 / (4 * in_energy) - out_energy
        q = M2 * (in_energy - out_energy) / (4 * in_energy)
        disc = (p / 2) ** 2 - q
        if disc <= 0:
            return 0.0
        lo = -p / 2 - np.sqrt(disc)
        hi = min(-p / 2 + np.sqrt(disc), in_energy - M2 / (4 * in_energy), self.tmax)
        integral = thermal_integral(lambda x: jones_function(in_energy + x - out_energy, in_energy, x) * _bose_weight(x, self.T), lo, hi, self.T, self.order)
        return 2 * PI * ALPHA**2 * integral / in_energy**2

    def double_pair_electron(self, out_energy, in_energy):
        if out_energy <= M_E or in_energy < M2 / (50 * self.T):
            return 0.0
        d = in_energy - out_energy
        if d <= 0:
            return 0.0
        root = np.sqrt(out_energy**2 - M2)
        den = 4 * in_energy * d + M2
        z1 = in_energy * (M2 + 2 * d * M2 / (root + out_energy)) / den
        z2 = in_energy * (M2 + 2 * d * (root + out_energy)) / den
        lo, hi = max(M2 / in_energy, z1), min(z2, in_energy, self.tmax)
        integral = thermal_integral(lambda x: pair_function(out_energy, in_energy, x) / (PI**2 * np.expm1(x / self.T)), lo, hi, self.T, self.order)
        return PI * ALPHA**2 * M2 * integral / (2 * in_energy**3)  # e- plus e+

    def inverse_compton_rate(self, electron):
        lo, hi = self.T * 1e-8, min(electron - M2 / (4 * electron), self.tmax)
        if hi <= lo:
            return 0.0
        nodes, weights = gauss_rule(self.order)
        def over_background(bg):
            result = np.zeros_like(bg)
            for k, x in enumerate(bg):
                ymax = 4 * x * electron**2 / (M2 + 4 * x * electron)
                if ymax <= x:
                    continue
                photons = (ymax - x) * nodes / 2 + (ymax + x) / 2
                result[k] = (ymax - x) / 2 * np.sum(weights * jones_function(photons, electron, np.full_like(photons, x)))
            # Eq. (47): f_BB(x)/x = x/[pi^2(exp(x/T)-1)].
            return result * _bose_weight(bg, self.T)
        return 2 * PI * ALPHA**2 * thermal_integral(over_background, lo, hi, self.T, self.order) / electron**2

    def build(self, energy):
        energy = np.asarray(energy)
        n = energy.size
        pp = 1946 / (50625 * PI) * ALPHA**2 * R_E**2 * M_E**-6 * energy**3 * self.moment
        cs = self.ne * klein_nishina_total(energy)
        bh = self.nz2 * nuclear_pair_total(energy)
        dp = np.array([self.double_pair_rate(e) for e in energy])
        ic = np.array([self.inverse_compton_rate(e) for e in energy])
        kernels = np.zeros((2, 2, n, n))
        for j, incoming in enumerate(energy):
            for i in range(j + 1):
                outgoing = energy[i]
                ratio = outgoing / incoming
                kernels[0, 0, i, j] = 1112 / (10125 * PI) * ALPHA**2 * R_E**2 * M_E**-6 * incoming**2 * (1 - ratio + ratio**2) ** 2 * self.moment
                kernels[0, 0, i, j] += compton_kernel(outgoing, incoming, self.ne)
                kernels[0, 1, i, j] = self.inverse_compton_photon(outgoing, incoming)
                kernels[1, 0, i, j] = compton_kernel(incoming + M_E - outgoing, incoming, self.ne)
                kernels[1, 0, i, j] += 2 * self.nz2 * bethe_heitler_differential(outgoing, incoming)
                kernels[1, 0, i, j] += self.double_pair_electron(outgoing, incoming)
                kernels[1, 1, i, j] = self.inverse_compton_electron(outgoing, incoming)
        components = {"photon_photon": pp, "compton": cs, "nuclear_pair": bh, "double_pair": dp, "inverse_compton": ic}
        return KernelSystem(pp + cs + bh + dp, ic, kernels, components)
