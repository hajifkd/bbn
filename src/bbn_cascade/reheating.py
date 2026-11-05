"""Electromagnetic reheating with a final eta boundary condition.

Internal thermodynamic units are MeV; cosmic time is seconds. The EM plasma is an ideal photon plus finite-mass e+- gas.
Neutrinos redshift freely without energy exchange, as requested. X is pressureless.
The photon bath is assumed to thermalize to a blackbody instantaneously.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, quad

from .bbn_decay import ROOT, ETA
from .decay import HBAR

HBAR_MEV_S=HBAR*1000
NGAMMA=0.243587656435
MNUCLEON_MEV=931.494


# Fixed quadrature integrates the finite-mass ideal FD gas without an EOS table.
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
_nodes,_weights=leggauss(128)
_MOMENTA=30*(_nodes+1)
_WEIGHTS=30*_weights


def plasma(T):
    """rho, pressure, d rho/dT of photons + equilibrium e+- in MeV units.

    The heat capacity differentiates the FD occupation at fixed physical
    momentum, avoiding numerical differentiation of tabulated thermodynamics.
    """
    if not np.isfinite(T) or T<=0 or T>10.0001:
        raise ValueError('Reheating EOS supports 0 < T_gamma <= 10 MeV')
    rg=np.pi**2/15*T**4
    x=.51099895/T
    if x>60: return rg,rg/3,4*rg/T
    y=_MOMENTA; E=np.sqrt(y*y+x*x); f=expit(-E)
    factor=2/np.pi**2
    re=factor*T**4*np.dot(_WEIGHTS,y*y*E*f)
    pe=factor/3*T**4*np.dot(_WEIGHTS,y**4/E*f)
    ce=factor*T**3*np.dot(_WEIGHTS,y*y*E*E*f*(1-f))
    return rg+re,rg/3+pe,4*rg/T+ce


def hubble(rho_MeV4):
    return np.sqrt(8*np.pi*rho_MeV4/(3*(1.220890e22)**2))/HBAR_MEV_S


@dataclass(frozen=True)
class ReheatingModel:
    mass_GeV: float=200.
    lifetime_s: float=1e5
    yield_X: float=1e-9
    eta_cmb: float=ETA
    initial_temperature_MeV: float=10.
    stable_dm_to_baryon: float=5.36

    def __post_init__(self):
        values=(self.mass_GeV,self.lifetime_s,self.yield_X,self.eta_cmb,self.initial_temperature_MeV,self.stable_dm_to_baryon)
        if not all(np.isfinite(values)) or self.mass_GeV<=0 or self.lifetime_s<100 or self.yield_X<0 or self.eta_cmb<=0 or not 8.7<=self.initial_temperature_MeV<=10 or self.stable_dm_to_baryon<0:
            raise ValueError('Require finite positive mass/eta, tau>=100 s, YX>=0, 8.7<=T_initial<=10 MeV, nonnegative DM ratio')
        if self.mass_GeV*1000<100*self.initial_temperature_MeV:
            raise ValueError('X must be nonrelativistic at the initial epoch')


@dataclass
class ReheatingHistory:
    model: ReheatingModel
    loga: np.ndarray
    time_s: np.ndarray
    temperature_MeV: np.ndarray
    neutrino_temperature_MeV: np.ndarray
    rho_X_MeV4: np.ndarray
    baryon_density_MeV3: np.ndarray
    hubble_per_s: np.ndarray
    dTdt_MeV_per_s: np.ndarray
    entropy_MeV3: np.ndarray
    metadata: dict

    def save(self, directory):
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        # AlterAlterBBN: t[s], T[MeV], dTdt[MeV^2], Tnu[MeV], H[MeV], nb/eta_final[MeV^3].
        table=np.column_stack([self.time_s,self.temperature_MeV,self.dTdt_MeV_per_s*HBAR_MEV_S,
                               self.neutrino_temperature_MeV,self.hubble_per_s*HBAR_MEV_S,
                               self.baryon_density_MeV3/self.model.eta_cmb])
        np.savetxt(out/'cosmo_file.dat',table,fmt='%.15e')
        np.savez(out/'background.npz',loga=self.loga,time_s=self.time_s,T_MeV=self.temperature_MeV,
                 Tnu_MeV=self.neutrino_temperature_MeV,rho_X_MeV4=self.rho_X_MeV4,
                 nb_MeV3=self.baryon_density_MeV3,H_per_s=self.hubble_per_s,
                 entropy_MeV3=self.entropy_MeV3)
        (out/'background_metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')


def evolve(model, *, end_time=None, rtol=2e-7, points=2200):
    """Shoot the conserved baryon number so eta after all X decays equals eta_cmb.

    YX refers to the initial SM entropy at T_initial, extrapolating away the
    tiny exponential depletion before the start; it is NOT nX/s(t) after heating.
    """
    T0=model.initial_temperature_MeV
    rho0,p0,_=plasma(T0);rnu0=7*np.pi**2/40*T0**4
    s0=(rho0+p0)/T0+4*rnu0/(3*T0)
    amplitude=model.mass_GeV*1000*model.yield_X*s0
    end=float(end_time or max(3e6,40*model.lifetime_s))
    if end<30*model.lifetime_s: raise ValueError('Evolution must cover at least 30 lifetimes')

    def integrate(log_eta_initial):
        nbi=np.exp(log_eta_initial)*NGAMMA*T0**3
        if nbi/T0**3>.01:
            raise ValueError('Initial baryon density requires an electron chemical potential; outside current EOS')
        matter0=amplitude+MNUCLEON_MEV*nbi*(1+model.stable_dm_to_baryon)
        # Initial age in the preceding relativistic-SM + nonrelativistic-matter era.
        ti=quad(lambda a:a/float(hubble(rho0+rnu0+matter0*a)),0,1,epsabs=1e-12)[0]
        def quantities(N,u):
            T,Tnu,t=np.exp(u)
            rho,p,heat=plasma(T)
            rnu=7*np.pi**2/40*Tnu**4
            rx=amplitude*np.exp(-3*N-t/model.lifetime_s)
            nb=nbi*np.exp(-3*N)
            matter=MNUCLEON_MEV*nb*(1+model.stable_dm_to_baryon)
            H=float(hubble(rho+rnu+rx+matter))
            q=0.
            dT=(-3*H*(rho+p)-q+rx/model.lifetime_s)/heat
            dTnu=-H*Tnu+q*Tnu/(4*rnu)
            return H,dT,dTnu,rx,nb,(rho+p)/T+4*rnu/(3*Tnu)
        def rhs(N,u):
            H,dT,dTnu,*_=quantities(N,u)
            T,Tnu,t=np.exp(u)
            return [dT/(T*H),dTnu/(Tnu*H),1/(t*H)]
        def finish(N,u): return u[2]-np.log(end*1.1)
        finish.terminal=True;finish.direction=1
        sol=solve_ivp(rhs,(0,40),np.log([T0,T0,ti]),method='Radau',rtol=rtol,atol=rtol*.02,
                      max_step=.08,events=finish,dense_output=True)
        if not sol.success or not len(sol.t_events[0]): raise RuntimeError(f'Background integration failed: {sol.message}')
        Nf=sol.t[-1];Tf=np.exp(sol.y[0,-1])
        final_eta=nbi*np.exp(-3*Nf)/(NGAMMA*Tf**3)
        return final_eta,sol,quantities

    guess=np.log(model.eta_cmb*2.75)
    shooting=[]
    for iteration in range(12):
        eta,sol,quantities=integrate(guess)
        error=np.log(eta/model.eta_cmb)
        shooting.append(dict(eta_initial=float(np.exp(guess)),eta_final=float(eta)))
        if abs(error)<max(2e-6,rtol*10): break
        guess-=error
    else: raise RuntimeError(f'Final-eta shooting did not converge: {shooting}')
    N=np.linspace(0,sol.t[-1],points);u=sol.sol(N);T,Tnu,t=np.exp(u)
    values=np.array([quantities(x,y) for x,y in zip(N,u.T)])
    H,dT,_,rx,nb,entropy=values.T
    ng=NGAMMA*T**3
    photons_comoving=ng*np.exp(3*N)
    neff=(3*(Tnu[-1]/T[-1])**4)/(4/11)**(4/3)
    # Fraction of the comoving decay energy released below the approximate
    # blackbody-restoration temperature (~470 eV, z~2e6). Diagnostic, not a cut.
    weights=rx*np.exp(3*N)/model.lifetime_s
    total=np.trapezoid(weights,t)
    late=np.trapezoid(np.where(T<4.7e-4,weights,0),t)/total if total>0 else 0.
    meta=dict(mass_GeV=model.mass_GeV,lifetime_s=model.lifetime_s,yield_X=model.yield_X,
              yield_reference='undecayed nX / initial total SM entropy at T_initial',
              T_initial_MeV=T0,eta_initial=float(nb[0]/ng[0]),eta_final=float(nb[-1]/ng[-1]),
              eta_cmb_target=model.eta_cmb,photon_comoving_growth=float(photons_comoving[-1]/photons_comoving[0]),
              total_entropy_comoving_growth=float(entropy[-1]*np.exp(3*N[-1])/entropy[0]),
              final_Neff=float(neff),max_X_to_radiation=float(np.max(rx/np.array([plasma(x)[0]+7*np.pi**2/40*y**4 for x,y in zip(T,Tnu)]))),
              blackbody_assumption=True,late_decay_energy_fraction_below_470eV=float(late),
              blackbody_caution=bool(late>.01),shooting=shooting,end_time_s=end,
              neutrinos='free redshifting FD gas, no energy exchange; initial Tnu=Tgamma',
              plasma_eos='ideal photons plus finite-mass FD electron/positron gas; no QED corrections',
              stable_dm_to_baryon=model.stable_dm_to_baryon,
              assumptions=['instantaneous complete electromagnetic thermalization',
                           'baryon and stable-DM rest mass in H; negligible baryon thermal pressure',
                           'initial age uses relativistic-SM plus matter asymptote before T_initial'])
    return ReheatingHistory(model,N,t,T,Tnu,rx,nb,H,dT,entropy,meta)
