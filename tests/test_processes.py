import numpy as np

from bbn_cascade.constants import EV, M_E
from bbn_cascade.hybrid_collision import HybridBoltzmannCollision
from bbn_cascade.kernels import compton_kernel
from bbn_cascade.processes import klein_nishina_total, nuclear_pair_total


def test_cross_sections_are_nonnegative():
    energy = np.geomspace(1e-6, 1e4, 200)
    assert np.all(klein_nishina_total(energy) >= 0)
    assert np.all(nuclear_pair_total(energy) >= 0)




def test_photon_photon_scattering_is_cut_off_as_in_paper_eq32():
    collision = HybridBoltzmannCollision(100 * EV, 3e-10, order=8)
    cutoff = M_E**2 / collision.T
    outgoing = 0.4 * cutoff

    below = collision._photon_parent_differential(outgoing, 0.8 * cutoff)[0]
    above = collision._photon_parent_differential(outgoing, 1.2 * cutoff)[0]
    compton_below = compton_kernel(outgoing, 0.8 * cutoff, collision.ne)
    compton_above = compton_kernel(outgoing, 1.2 * cutoff, collision.ne)

    assert below > compton_below
def test_nuclear_pair_threshold():
    assert nuclear_pair_total(np.array([1.99 * M_E]))[0] == 0

