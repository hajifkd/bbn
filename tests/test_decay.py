import numpy as np
from scipy.linalg import expm
from bbn_cascade.decay import (PhotonDecay, HBAR, photonuclear_matrix,
                               MASS_NUMBERS, CHARGES)


def test_conservation_each_channel():
    for rate in np.eye(17):
        matrix = photonuclear_matrix(rate)
        np.testing.assert_array_equal(MASS_NUMBERS @ matrix, 0)
        np.testing.assert_array_equal(CHARGES @ matrix, 0)


def test_deuterium_analytic():
    rates = np.zeros(17)
    rates[0] = 0.3
    initial = np.zeros(9)
    initial[2] = 1
    final = expm(photonuclear_matrix(rates)*7) @ initial
    np.testing.assert_allclose(final[:3], [1-np.exp(-2.1), 1-np.exp(-2.1), np.exp(-2.1)])


def test_decay_units_and_zero():
    model = PhotonDecay(200, 100, 1e-12)
    np.testing.assert_allclose(model.injection_GeV5(100, 2)/HBAR, 4e-12/np.e)
    assert PhotonDecay(200,100,0).injection_GeV5(100,2) == 0
