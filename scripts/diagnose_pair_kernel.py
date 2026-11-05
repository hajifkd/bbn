"""Controlled pair-kernel replacement in the CURRENT solver; diagnostic only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import argparse,csv,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import numpy as np
from bbn_cascade.constants import M_E,ALPHA,PI
from bbn_cascade.kernels import CollisionIntegrals
from bbn_cascade.hybrid_collision import HybridBoltzmannCollision
from bbn_cascade.hybrid_solver import CascadeSolver
from bbn_cascade.config import CascadeConfig
from bbn_cascade.photodisintegration import calculate_rates,write_rates


def old_pair(self, out_energy, in_energy, bug=False):
    # dp_t_ele / incoming energy: Fortran table_t integrates in d log E_in.
    # Keep CURRENT m_e, alpha and pi to isolate kernel/sampling effects.
    a,b=out_energy,in_energy-out_energy
    if a<=M_E or b<=M_E: return 0.
    lo=max(self.T*1e-4, M_E**2*in_energy/(4*a*b)); hi=self.T*100
    if lo>=hi: return 0.
    h=np.log(hi/lo)/59
    bg=lo*np.exp(h*np.arange(119)/2)
    prod=a*b; total=in_energy
    value=4*total**2/prod*np.log(4*bg*prod/(M_E**2*total))
    value-=8*bg*total/M_E**2
    value+=2*(2*bg*total-M_E**2)*total**2/(M_E**2*prod)
    value-=(1-M_E**2/(bg*total))*total**4/prod**2
    integrand=value*bg/(PI**2*np.expm1(bg/self.T))
    integrand[-1]=0. # bg_photon's strict 100 T cutoff
    if bug: integrand[2::2]*=np.exp(h/2)
    weights=np.full(119,2.);weights[1::2]=4.;weights[[0,-1]]=1.
    return PI*ALPHA**2*M_E**2/(2*in_energy**3)*h/6*np.dot(weights,integrand)


def old_fixed(self,e,p): return old_pair(self,e,p,False)
def old_bug(self,e,p): return old_pair(self,e,p,True)


def worker(task):
    temperature,mode,root=task
    if mode=='old_fixed': HybridBoltzmannCollision.double_pair_electron=old_fixed
    elif mode=='old_bug': HybridBoltzmannCollision.double_pair_electron=old_bug
    else: raise ValueError(mode)
    start=time.monotonic()
    cfg=CascadeConfig(temperature,100.,points_per_decade=120,quadrature_order=8,eta_baryon=6.1e-10)
    spectrum=CascadeSolver(cfg).solve(); rows=calculate_rates(spectrum)
    out=Path(root)/f'{mode}_T{temperature:.0e}'
    write_rates(out,rows,dict(config=asdict(cfg),variant=mode,diagnostics=spectrum.diagnostics,
        note='Only pair-electron reinjection kernel replaced; current physical constants and all other operators retained'))
    spectrum.to_csv(out/'spectrum.csv');np.save(out/'boundaries_GeV.npy',spectrum.boundaries)
    return mode,temperature,rows[0].rate_GeV,time.monotonic()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--balances-only',action='store_true');args=p.parse_args()
    root=Path('results/high_temperature_diagnosis/current_solver');root.mkdir(parents=True,exist_ok=True)
    nodes,weights=np.polynomial.legendre.leggauss(12)
    rows=[]
    for t in (1e-7,1e-5,1e-3):
        obj=CollisionIntegrals(t,6.1e-10,order=12)
        for incoming in (.003,.01,.1,1.,100.):
            cuts=np.geomspace(M_E,incoming-M_E,150)
            cuts=np.unique(np.r_[cuts,incoming-cuts]); cuts=cuts[(cuts>=M_E)&(cuts<=incoming-M_E)]
            a,b=cuts[:-1,None],cuts[1:,None]
            energies=((b-a)*nodes/2+(a+b)/2).ravel();w=((b-a)*weights/2).ravel()
            loss=obj.double_pair_rate(incoming)
            for name,fn in [('new',obj.double_pair_electron),('old_fixed',lambda e,p:old_pair(obj,e,p)),('old_bug',lambda e,p:old_pair(obj,e,p,True))]:
                values=np.array([fn(e,incoming) for e in energies]);number=np.dot(w,values);energy=np.dot(w,values*energies)
                rows.append(dict(temperature_GeV=t,incoming_GeV=incoming,variant=name,pair_rate=loss,number_over_2loss=number/(2*loss) if loss>0 else None,energy_over_parent_loss=energy/(incoming*loss) if loss>0 else None))
    with (root/'pair_balances.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote pair balances',flush=True)
    if not args.balances_only:
        tasks=[(t,mode,str(root)) for t in (1e-7,1e-5,1e-3) for mode in ('old_fixed','old_bug')]
        with ProcessPoolExecutor(max_workers=6) as pool:
            for f in as_completed([pool.submit(worker,t) for t in tasks]): print(f.result(),flush=True)


if __name__=='__main__':main()
