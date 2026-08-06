import numpy as np

from bbn_cascade import CascadeConfig, CascadeSolver
from bbn_cascade.constants import EV
from bbn_cascade.published import table_parameters


def test_published_spectrum_is_finite_and_nonnegative():
    result = CascadeSolver(CascadeConfig(EV, 1e3, mode="published-fit", grid_size=100)).solve()
    assert np.all(np.isfinite(result.photon))
    assert np.all(result.photon >= 0)


def test_table1_reference_point():
    assert np.allclose(table_parameters(EV, 1e3), (-1.56, 1.4e8, -5.07, 6.2e-18))


def test_transport_runs_and_has_both_species():
    result = CascadeSolver(CascadeConfig(100 * EV, 100, grid_size=100)).solve()
    assert np.any(result.photon > 0)
    assert np.any(result.electron > 0)
    assert np.all(result.photon >= 0)

