import numpy as np

from bbn_cascade.constants import EV, M_E
from bbn_cascade.kernels import (
    CollisionIntegrals,
    bethe_heitler_differential,
    compton_kernel,
    jones_function,
)


def test_differential_kernels_have_physical_support():
    collision = CollisionIntegrals(100 * EV, 3e-10, order=8)
    assert compton_kernel(2.0, 1.0, collision.ne) == 0
    assert bethe_heitler_differential(M_E / 2, 1.0) == 0
    assert np.all(jones_function(1e-4, 1.0, np.array([1e-7])) >= 0)


def test_all_four_kernel_matrices_are_finite_and_nonnegative():
    energy = np.geomspace(1e-3, 1.0, 10)
    system = CollisionIntegrals(100 * EV, 3e-10, order=8).build(energy)
    assert system.kernels.shape == (2, 2, 10, 10)
    assert np.all(np.isfinite(system.kernels))
    assert np.all(system.kernels >= 0)
    assert np.all(system.photon_loss >= 0)
    assert np.all(system.electron_loss >= 0)
