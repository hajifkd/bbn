import numpy as np
from scipy.special import zeta

from bbn_cascade.quadrature import thermal_quadrature
from bbn_cascade.constants import EV
from bbn_cascade.hybrid_collision import HybridBoltzmannCollision
from bbn_cascade.kernels import CollisionIntegrals
from bbn_cascade.inverse_compton_split import build_inverse_compton_moments


def test_thermal_number_and_energy_moments():
    for temperature in [EV,100*EV]:
        eps,weights=thermal_quadrature(1e-8*temperature,200*temperature,temperature,8)
        x=eps/temperature
        number=np.dot(weights/temperature,x*x/np.expm1(x))
        energy=np.dot(weights/temperature,x**3/np.expm1(x))
        assert abs(number/(2*zeta(3))-1)<1e-6
        assert abs(energy/(np.pi**4/15)-1)<1e-6


def test_pair_rate_and_endpoint_energy_balance():
    energy=np.geomspace(.001,1e4,421)
    boundaries=np.r_[energy[0],np.sqrt(energy[:-1]*energy[1:]),energy[-1]]
    c=HybridBoltzmannCollision(100*EV,3e-10,order=8)
    column=c.photon_parent_column(energy,boundaries,len(energy)-1)
    rate=c.double_pair_rate(energy[-1])
    reference=CollisionIntegrals(100*EV,3e-10,order=16).double_pair_rate(energy[-1])
    assert abs(rate/reference-1)<1e-4
    # Pair creation dominates; 1.7e-3 energy residual comes from storing
    # cell averages at grid energies, not from the continuous kernel integral.
    assert abs(column[1].sum()/(2*rate)-1)<2e-4
    assert abs(np.dot(energy,column.sum(axis=0))/(energy[-1]*rate)-1)<2e-3


def test_ic_composite_thermal_order_convergence():
    energy=np.geomspace(.001,100.,40)
    a=build_inverse_compton_moments(energy,100*EV,quadrature_order=8)
    b=build_inverse_compton_moments(energy,100*EV,quadrature_order=12)
    for x,y in [(a.drift,b.drift),(a.diffusion,b.diffusion),(a.hard_loss,b.hard_loss)]:
        mask=y>np.max(y)*1e-8
        assert np.max(abs(x[mask]/y[mask]-1))<.01
