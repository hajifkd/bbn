import numpy as np
import pytest
from bbn_cascade.inverse_compton_fp import chang_cooper_flux_coefficients


def test_variable_diffusion_flux_for_constant_density():
    # With b=0 and f=1, J=-(D*f)'/2=-D'/2. The previous implementation
    # returned zero, i.e. discretized D*f' instead of (D*f)'.
    for low, high in [(1.,1.04),(2.,3.),(7.,7.1)]:
        k=3.
        left,right=chang_cooper_flux_coefficients(low,high,0.,0.,k*low**2,k*high**2)
        assert left+right==pytest.approx(-k*(low+high)/2)


def test_constant_diffusion_exponential_zero_flux():
    # For constant b and D, exact zero-flux solution f ~ exp(-2*b*E/D).
    b,D=2.,3.
    low,high=1.,1.2
    left,right=chang_cooper_flux_coefficients(low,high,b,b,D,D)
    assert left*np.exp(-2*b*low/D)+right*np.exp(-2*b*high/D)==pytest.approx(0.,abs=1e-14)


def test_zero_diffusion_cooling_upwind_limit():
    left,right=chang_cooper_flux_coefficients(1.,1.1,2.,4.,0.,0.)
    assert left==0.
    assert right==-3.
