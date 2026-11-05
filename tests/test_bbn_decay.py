import numpy as np
import pytest
from bbn_cascade.bbn_decay import Backend, observables, loginterp


def test_standard_and_late_extension():
    b=Backend()
    early=b.solve(end=0)
    np.testing.assert_allclose(early[:9], [0,.753262345,1.88297590e-5,6.02846894e-8,
        7.71801920e-6,.0616691645,8.27344440e-15,2.20151111e-11,3.76560789e-10],rtol=5e-5)
    late=b.solve()
    np.testing.assert_allclose(observables(late),observables(early),rtol=1e-5)
    table=np.column_stack([np.geomspace(.001,1e11,10),np.ones((10,17))])
    np.testing.assert_allclose(observables(b.solve(table,tau=1e8,yield_X=0)),observables(late),rtol=.002)


def test_late_deuterium_destruction_in_c_network():
    b=Backend(); base=b.solve()
    table=np.zeros((4,18));table[:,0]=[.001,1e7,1e7*(1+1e-6),1e11]
    table[2:,1]=.1
    y=b.solve(table,tau=1e8,yield_X=1)
    expected=np.exp(-.1*(np.exp(-.1)-np.exp(-30)))
    np.testing.assert_allclose(y[2]/base[2],expected,rtol=5e-4)
    weights=np.array([1,1,2,3,3,4,6,7,7,8,8,9,10,11,11,12,12,12,13,13,14,14,14,15,15,16])
    np.testing.assert_allclose(y@weights,base@weights,atol=1e-8)


def test_response_input_validation():
    with pytest.raises(ValueError): loginterp(0.1,np.array([1,2]),np.array([3,4]))
    with pytest.raises(ValueError): Backend().solve(np.ones((3,18)),yield_X=1)


def test_zero_injection_continuity():
    table=np.column_stack([np.geomspace(.001,1e11,10),np.ones((10,17))])
    b=Backend()
    np.testing.assert_array_equal(b.solve(table,tau=100,yield_X=0), b.solve(table,tau=100,yield_X=1e-100))
