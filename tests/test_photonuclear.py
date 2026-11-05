from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from bbn_cascade import REACTIONS, cross_section, calculate_rates, CascadeSpectrum
from bbn_cascade.constants import MEV, MB_TO_GEV_MINUS2, HBAR_GEV_S


def spectrum(edges, values, injection=1.):
    edges = np.asarray(edges, dtype=float)
    return CascadeSpectrum(np.sqrt(edges[:-1]*edges[1:]), np.array(values, dtype=float),
                           np.zeros(len(values)), {}, {'injected_energy_rate': injection}, edges)


def test_support_units_and_positivity():
    assert MB_TO_GEV_MINUS2 == pytest.approx(2.56819, rel=2e-6)
    for r in REACTIONS:
        assert cross_section(r.id, r.threshold_MeV*MEV) == 0
        energies = np.geomspace(r.threshold_MeV*MEV*(1+1e-9), 1., 500)
        assert np.all(np.isfinite(cross_section(r.id, energies)))
        assert np.all(cross_section(r.id, energies) >= 0)
    with pytest.raises(ValueError):
        cross_section(15, 1.01)


def test_updated_helium_normalization():
    # At E=2Q, Q^3.5 (E-Q)/E^4.5 = 2^-4.5.
    assert cross_section(7, 2*20.577615*MEV)/MB_TO_GEV_MINUS2 == pytest.approx(20.7/2**4.5)


def test_li7_piecewise_and_be7_s_d_wave():
    # III.7: Ecm=4 MeV, hence Coulomb exponent -1.30.
    e = 2.467032+4
    assert cross_section(12, e*MEV)/MB_TO_GEV_MINUS2 == pytest.approx(48.1/e**2*np.exp(-1.30))
    # III.8 evaluated independently at Ecm=1: two final-state contributions.
    d = (1+4*np.pi**2/26.9361)*(1+16*np.pi**2/26.9361)
    s = 1.586627/2.586627*(.406*.793**2+.007*d)+1.157/2.157*(.163*.866**2+.004*d)
    expected = 801/2.586627**2*np.exp(-5.19)*s
    assert cross_section(15, 2.586627*MEV)/MB_TO_GEV_MINUS2 == pytest.approx(expected)
    assert cross_section(15, .006) > 0  # Old polynomial was negative here.


def test_constant_spectrum_threshold_cutoff_and_tail():
    sp = spectrum([.001, .02, .3, 2.], [2., 2., 2.])
    row = calculate_rates(sp)[0]
    expected = 2*quad(lambda e: cross_section(1, e), .002224573, 1., epsabs=1e-14)[0]
    tail = 2*quad(lambda e: cross_section(1, e), .1, 1., epsabs=1e-14)[0]
    assert row.rate_GeV == pytest.approx(expected, rel=1e-7)
    assert row.above_100MeV_rate_GeV == pytest.approx(tail)
    assert row.rate_per_s == pytest.approx(expected/HBAR_GEV_S)
    rows2 = calculate_rates(replace(sp, photon=sp.photon*2, diagnostics={'injected_energy_rate': 2.}))
    assert rows2[0].rate_GeV == pytest.approx(2*row.rate_GeV)
    assert rows2[0].response_GeV_minus4 == pytest.approx(row.response_GeV_minus4)


def test_narrow_line_normalization_and_no_extra_line():
    e, width, number = .01, 1e-7, 3.
    sp = spectrum([.001, e-width/2, e+width/2], [0., number/width])
    row = calculate_rates(sp)[0]
    assert row.rate_GeV == pytest.approx(number*cross_section(1, e), rel=1e-8)
    assert calculate_rates(sp, maximum_energy=.002)[0].rate_GeV == 0


def test_invalid_spectrum_rejected():
    sp = spectrum([.001, .01], [1.])
    with pytest.raises(ValueError):
        calculate_rates(replace(sp, boundaries=None))
    with pytest.raises(ValueError):
        calculate_rates(replace(sp, photon=np.array([-1.])))
    with pytest.raises(ValueError):
        calculate_rates(replace(sp, diagnostics={}))


@pytest.mark.parametrize('rid,c,power', [(2,9.8,3.6),(3,26.,4.9),(4,8.88,3.4),
    (5,16.7,4.25),(6,19.5,4.5),(8,10.7,13.6),(9,21.7,7.),(10,104.,7.),
    (14,122.,7.),(17,133.,7.)])
def test_appendix_power_fits_at_twice_threshold(rid,c,power):
    q=REACTIONS[rid-1].threshold_MeV
    assert cross_section(rid,2*q*MEV)/MB_TO_GEV_MINUS2 == pytest.approx(c/2**power)


def test_li6_gaussians_and_be7_two_component_fit():
    # Appendix A #11 at first Gaussian center, and #16 at E=2Q.
    q=15.794685
    expected=38.1*q**3*(19-q)**2/19**5*(3.7+2.75*np.exp(-.5*(11/3)**2)+2.2*np.exp(-.5*(24/5)**2))
    assert cross_section(11,.019)/MB_TO_GEV_MINUS2 == pytest.approx(expected)
    q=5.605794
    expected=32.6/2**12+2.27e6/2**21.8335
    assert cross_section(16,2*q*MEV)/MB_TO_GEV_MINUS2 == pytest.approx(expected)


def test_rate_cli_output(tmp_path):
    from bbn_cascade.cli import main
    main(['rates','--energy','30MeV','--temperature','100eV','--grid-size','40',
          '--output',str(tmp_path)])
    import csv,json
    with (tmp_path/'rates.csv').open() as h:
        rows=list(csv.DictReader(h))
    assert len(rows)==17
    assert float(rows[0]['upper_GeV'])==pytest.approx(.03)
    meta=json.loads((tmp_path/'metadata.json').read_text())
    assert meta['config']['maximum_energy']==pytest.approx(.03)
    assert (tmp_path/'boundaries_GeV.npy').exists()
