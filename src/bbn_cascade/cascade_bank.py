"""Density-dependent stationary cascade operators on a fixed standard composition.

A(T, eta) = A_radiative(T) + (eta*z2/eta_reference)*A_baryonic(T).
This retains the changed target density exactly at each temperature node;
only the standard nuclear composition is held fixed, as requested.
"""
import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
from scipy.integrate import quad
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import spsolve

from .bbn_decay import ROOT, CHARGES
from .config import CascadeConfig
from .hybrid_solver import CascadeSolver
from .hybrid_collision import HybridBoltzmannCollision
from .inverse_compton_split import build_inverse_compton_moments
from .sources import MonochromaticPhotonSource
from .photonuclear import REACTIONS, cross_section

ETA_REFERENCE=1e-9


class BaryonicCollision(HybridBoltzmannCollision):
    """Same quadrature and kernels, with photon-bath interactions switched off."""
    def __init__(self,T,eta,he,order):
        super().__init__(T,eta,he,order)
        self.moment=0.
    def double_pair_rate(self,photon): return 0.
    def double_pair_electron(self,outgoing,incoming): return 0.


def source_hash():
    h=hashlib.sha256()
    names=['cascade_bank.py','config.py','hybrid_solver.py','hybrid_collision.py',
           'inverse_compton_fp.py','inverse_compton_split.py','kernels.py','processes.py',
           'quadrature.py','sources.py','photonuclear.py','constants.py']
    for name in names: h.update((ROOT/'src/bbn_cascade'/name).read_bytes())
    return h.hexdigest()


def cell_weights(edges):
    W=np.zeros((17,len(edges)-1));knots=[.019,.030,.043,.014709962,.003467032,.1]
    for r in REACTIONS:
        for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            a,b=max(lo,r.threshold_MeV*.001),min(hi,1.)
            if b<=a: continue
            cuts=[a]+sorted(x for x in knots if a<x<b)+[b]
            W[r.id-1,i]=sum(quad(lambda E:cross_section(r.id,E),l,h,epsabs=1e-30,epsrel=1e-7)[0]
                            for l,h in zip(cuts[:-1],cuts[1:]))/(hi-lo)
    return W


def build_node(task):
    directory,index,T,he,mass,ppd,identity=task
    out=Path(directory);marker=out/f'node_{index:03d}.json'
    if marker.exists():
        if json.loads(marker.read_text())['identity']!=identity: raise ValueError('Incompatible operator checkpoint')
        if not all((out/f'{name}_{index:03d}.npz').exists() for name in ['radiative','baryonic']):
            raise ValueError('Incomplete operator checkpoint')
        return index
    cfg=CascadeConfig(T,mass/2,eta_baryon=ETA_REFERENCE,helium_mass_fraction=he,
                      points_per_decade=ppd,quadrature_order=8)
    energy=np.geomspace(cfg.minimum_energy,cfg.maximum_energy,cfg.grid_size)
    solver=CascadeSolver(cfg);source=MonochromaticPhotonSource(mass/2)
    rad=HybridBoltzmannCollision(T,0.,he,8).build_hybrid(energy)
    moments=build_inverse_compton_moments(energy,T,cfg.soft_fraction,8,cfg.background_cutoff_over_temperature)
    transitions=rad.transitions.copy()
    transitions[0,1]=moments.soft_photon_kernel+moments.hard_photon_kernel
    transitions[1,1]=moments.hard_electron_kernel
    A,b,widths=solver._assemble_operator(energy,rad.boundaries,rad.photon_loss,moments.hard_loss,
                                         transitions,moments.drift,moments.diffusion,source)
    baryon=BaryonicCollision(T,ETA_REFERENCE,he,8).build_hybrid(energy)
    zero=np.zeros_like(energy)
    B,_,_=solver._assemble_operator(energy,baryon.boundaries,baryon.photon_loss,zero,
                                    baryon.transitions,zero,zero,source)
    save_npz(out/f'radiative_{index:03d}.npz',A)
    save_npz(out/f'baryonic_{index:03d}.npz',B)
    marker.write_text(json.dumps(dict(identity=identity,config=asdict(cfg)),indent=2)+'\n')
    return index


def prepare_bank(directory, standard_directory, *, mass=200., ppd=120., intervals=10, workers=12):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    standard=Path(standard_directory)
    bg=np.load(standard/'background.npz');hist=np.loadtxt(standard/'thermal_history.dat')
    Th=np.exp(np.interp(np.log(hist[:,0]),np.log(bg['time_s']),np.log(bg['T_MeV']*.001)))
    temperature=np.geomspace(.01,1e-9,7*intervals+1)
    z2=[];he=[]
    for T in temperature:
        y=np.array([np.interp(np.log(T),np.log(Th[::-1]),hist[::-1,j]) for j in range(1,10)])
        z=float(CHARGES@y);zz=float(CHARGES**2@y)
        z2.append(zz);he.append(float(np.clip(2*(1-z/zz),0,1)))
    identity=dict(source_sha256=source_hash(),mass_GeV=mass,points_per_decade=ppd,intervals=intervals,
                  standard_composition_sha256=hashlib.sha256((standard/'thermal_history.dat').read_bytes()).hexdigest())
    record=out/'identity.json'
    if record.exists() and json.loads(record.read_text())!=identity: raise ValueError('Choose a new operator-bank directory')
    record.write_text(json.dumps(identity,indent=2)+'\n')
    cfg=CascadeConfig(.001,mass/2,points_per_decade=ppd)
    energy=np.geomspace(cfg.minimum_energy,cfg.maximum_energy,cfg.grid_size)
    edges=np.r_[energy[0],np.sqrt(energy[:-1]*energy[1:]),energy[-1]]
    np.savez(out/'grid.npz',temperature_GeV=temperature,z2=z2,helium=he,energy_GeV=energy,
             edges_GeV=edges,weights=cell_weights(edges))
    tasks=[(str(out),i,float(T),he[i],mass,ppd,identity) for i,T in enumerate(temperature)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for count,f in enumerate(as_completed([pool.submit(build_node,t) for t in tasks]),1):
            print(f'operator {count}/{len(tasks)} ready (node {f.result()})',flush=True)
    return out


class ResponseBank:
    def __init__(self,directory,density_log_step=.1):
        self.path=Path(directory);self.identity=json.loads((self.path/'identity.json').read_text())
        if self.identity['source_sha256']!=source_hash(): raise ValueError('Operator physics sources have changed')
        if density_log_step<=0: raise ValueError('Density interpolation step must be positive')
        self.grid=np.load(self.path/'grid.npz');self.T=self.grid['temperature_GeV'][::-1]
        self.weights=self.grid['weights'];self.n=self.weights.shape[1]
        self.step=density_log_step;self.operators={};self.responses={}
        self.source=np.zeros(2*self.n);self.source[self.n-1]=2/self.identity['mass_GeV']

    def node(self,index,eta):
        if index not in self.operators:
            self.operators[index]=(load_npz(self.path/f'radiative_{index:03d}.npz'),load_npz(self.path/f'baryonic_{index:03d}.npz'))
        A,B=self.operators[index]
        population=spsolve(A+(eta*self.grid['z2'][index]/ETA_REFERENCE)*B,self.source)
        if np.any(~np.isfinite(population)) or np.min(population)<-1e-10*np.max(abs(population)):
            raise RuntimeError('Invalid stationary cascade populations')
        return self.weights@np.maximum(population[:self.n],0)

    def _density(self,index,eta):
        position=np.log(eta)/self.step;lo=int(np.floor(position));w=position-lo
        key0=(index,lo);key1=(index,lo+1)
        for key in (key0,key1):
            if key not in self.responses: self.responses[key]=self.node(index,np.exp(key[1]*self.step))
        return positive_mix(self.responses[key0],self.responses[key1],w)

    def evaluate(self,T,eta):
        if not np.isfinite(T) or not np.isfinite(eta) or eta<=0 or T<self.T[0]*(1-1e-10) or T>self.T[-1]*(1+1e-10):
            raise ValueError('Response outside bank temperature range or invalid eta')
        T=float(np.clip(T,self.T[0],self.T[-1]));j=min(max(np.searchsorted(self.T,T)-1,0),len(self.T)-2)
        w=np.log(T/self.T[j])/np.log(self.T[j+1]/self.T[j])
        a=self._density(len(self.T)-1-j,eta);b=self._density(len(self.T)-2-j,eta)
        return positive_mix(a,b,w)

    def rate_history(self,background):
        result=np.zeros((len(background.time_s),18));result[:,0]=background.time_s
        for i,(T,nb,rho) in enumerate(zip(background.temperature_MeV,background.baryon_density_MeV3,background.rho_X_MeV4)):
            if rho==0: continue
            eta=nb/(.243587656435*T**3)
            # K [GeV^-4] * rho_X [GeV^4] / tau [s].
            result[i,1:]=self.evaluate(T*.001,eta)*rho*1e-12/background.model.lifetime_s
        return result


def positive_mix(a,b,w):
    positive=(a>0)&(b>0);out=(1-w)*a+w*b
    out[positive]=np.exp((1-w)*np.log(a[positive])+w*np.log(b[positive]))
    return out
