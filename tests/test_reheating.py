import numpy as np
from bbn_cascade.reheating import ReheatingModel,evolve,plasma
from bbn_cascade.bbn_decay import Backend,observables


def test_plasma_limits_and_heat_capacity():
    for T in [.001,.02,.2,1.,10.]:
        rho,p,heat=plasma(T)
        eps=1e-5
        # Avoid stepping outside the upper EOS bound.
        if T<10:
            finite=(plasma(T*(1+eps))[0]-plasma(T*(1-eps))[0])/(2*eps*T)
            np.testing.assert_allclose(heat,finite,rtol=2e-7)
        assert rho>0 and p>0 and heat>0
    rho,p,c=plasma(.001)
    np.testing.assert_allclose([p/rho,c*.001/rho],[1/3,4])


def test_entropy_and_final_eta():
    baseline=evolve(ReheatingModel(yield_X=0),points=400)
    hot=evolve(ReheatingModel(yield_X=1e-8),points=400)
    for h in [baseline,hot]:
        np.testing.assert_allclose(h.metadata['eta_final'],h.model.eta_cmb,rtol=3e-6)
        np.testing.assert_allclose(h.neutrino_temperature_MeV*np.exp(h.loga),h.neutrino_temperature_MeV[0],rtol=2e-7)
        np.testing.assert_allclose(h.baryon_density_MeV3*np.exp(3*h.loga),h.baryon_density_MeV3[0],rtol=2e-14)
    S=baseline.entropy_MeV3*np.exp(3*baseline.loga)
    np.testing.assert_allclose(S/S[0],1,rtol=3e-6)
    assert hot.metadata['eta_initial']>baseline.metadata['eta_initial']
    assert hot.metadata['total_entropy_comoving_growth']>1
    assert hot.metadata['final_Neff']<baseline.metadata['final_Neff']
    np.testing.assert_allclose(hot.metadata['eta_initial']/hot.metadata['photon_comoving_growth'],hot.model.eta_cmb,rtol=3e-6)


def test_energy_continuity():
    h=evolve(ReheatingModel(yield_X=1e-8),points=2500)
    rho_em=np.array([plasma(T)[0] for T in h.temperature_MeV])
    p_em=np.array([plasma(T)[1] for T in h.temperature_MeV])
    rho_nu=7*np.pi**2/40*h.neutrino_temperature_MeV**4
    rhs=-3*h.hubble_per_s*(rho_em+p_em)+h.rho_X_MeV4/h.model.lifetime_s
    # Finite differences on a logarithmic scale factor grid.
    measured=np.gradient(rho_em,h.time_s)
    np.testing.assert_allclose(measured[5:-5]/rhs[5:-5],1,rtol=.003)
    assert np.all(np.diff(h.entropy_MeV3*np.exp(3*h.loga))>=-1e-6*h.entropy_MeV3[0])


def test_alterbbn_accepts_heated_cosmology(tmp_path):
    h=evolve(ReheatingModel(yield_X=0),points=1800);h.save(tmp_path)
    y=Backend().solve_background(tmp_path/'cosmo_file.dat',end=h.metadata['end_time_s'])
    assert .23<4*y[5]<.26
    assert 1e-5<y[2]/y[1]<4e-5
    assert np.all(np.isfinite(observables(y)))


def test_temperature_turnaround_uses_time_endpoint(tmp_path):
    # Synthetic background tests the C interface, not energy conservation.
    h=evolve(ReheatingModel(yield_X=0),points=1600);h.save(tmp_path)
    c=np.loadtxt(tmp_path/'cosmo_file.dat');t=c[:,0]
    c[:,1]*=1+2*np.exp(-.5*(np.log(t/5e5)/.18)**2)
    from bbn_cascade.reheating import HBAR_MEV_S
    c[:,2]=np.gradient(c[:,1],t)*HBAR_MEV_S
    assert np.any(c[:,2]>0)
    np.savetxt(tmp_path/'cosmo_file.dat',c)
    y=Backend().solve_background(tmp_path/'cosmo_file.dat',end=h.metadata['end_time_s'],history=tmp_path/'history.dat')
    history=np.loadtxt(tmp_path/'history.dat')
    assert history[-1,0]>=h.metadata['end_time_s']
    assert np.all(np.isfinite(y))
