import numpy as np

from bbn_cascade.constants import EV
from bbn_cascade.inverse_compton_fp import (
    build_inverse_compton_moments,
    chang_cooper_weight,
)


def test_ic_moments_are_nonnegative_and_share_soft_energy_integral():
    energy = np.geomspace(1e-3, 1.0, 20)
    moments = build_inverse_compton_moments(
        energy, 100 * EV, quadrature_order=8
    )
    assert np.all(moments.drift >= 0)
    assert np.all(moments.diffusion >= 0)
    assert np.all(moments.hard_loss >= 0)
    assert np.allclose(moments.drift, moments.soft_energy_from_photons)


def test_chang_cooper_has_correct_drift_limits():
    assert chang_cooper_weight(-1.0, 0.0, 1.0) == 1.0
    assert chang_cooper_weight(1.0, 0.0, 1.0) == 0.0
    assert chang_cooper_weight(0.0, 1.0, 1.0) == 0.5
