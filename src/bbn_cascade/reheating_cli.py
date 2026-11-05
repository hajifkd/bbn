"""Entropy-producing X -> gamma gamma: background shooting, BBN and photodisintegration."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from .bbn_decay import ROOT,ETA,Backend,observables,fingerprint
from .reheating import ReheatingModel,evolve
from .cascade_bank import ResponseBank,prepare_bank

DEFAULT_STANDARD=ROOT/'results/reheating_standard'
DEFAULT_BANK=ROOT/'results/reheating_bank_200GeV'


def standard(directory=DEFAULT_STANDARD,eta_cmb=ETA):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    h=evolve(ReheatingModel(yield_X=0,eta_cmb=eta_cmb),end_time=4e9)
    h.save(out)
    y=Backend().solve_background(out/'cosmo_file.dat',eta_cmb=eta_cmb,end=4e9,history=out/'thermal_history.dat')
    np.save(out/'standard_abundances.npy',y)
    return y


def run_point(model,output,bank=None,*,baseline=None,include_photons=True,make_plot=True,save_histories=True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    h=evolve(model);h.save(out)
    backend=Backend()
    thermal=backend.solve_background(out/'cosmo_file.dat',eta_cmb=model.eta_cmb,tau=model.lifetime_s,
                                     end=h.metadata['end_time_s'],history=out/'thermal_history.dat' if save_histories else None)
    if include_photons and model.yield_X>0:
        if bank is None: raise ValueError('A density-dependent response bank is required')
        if not np.isclose(model.mass_GeV,bank.identity['mass_GeV'],rtol=1e-12,atol=0):
            raise ValueError('Decay mass does not match response bank')
        rates=bank.rate_history(h)
        np.save(out/'physical_rates_per_s.npy',rates)
        final=backend.solve_background(out/'cosmo_file.dat',rates,eta_cmb=model.eta_cmb,tau=model.lifetime_s,
                                       end=h.metadata['end_time_s'],history=out/'abundance_history.dat' if save_histories else None)
    else:
        final=thermal
    np.save(out/'final_abundances.npy',final)
    if baseline is None:
        zero=evolve(replace(model,yield_X=0.));zero.save(out/'reference')
        baseline=backend.solve_background(out/'reference/cosmo_file.dat',eta_cmb=model.eta_cmb,
                    tau=model.lifetime_s,end=zero.metadata['end_time_s'])
    obs=observables(final);thermal_obs=observables(thermal);ref=observables(baseline)
    report=dict(model=asdict(model),background=h.metadata,observables=obs.tolist(),
                standard_observables=ref.tolist(),ratios=(obs/ref).tolist(),
                thermal_only_ratios=(thermal_obs/ref).tolist(),
                observables_order=['D/H','(He3+T)/H','He4 mass fraction','(Li7+Be7)/H'],
                photodisintegration_included=include_photons,
                fixed_standard_cascade_composition=True,
                nonthermal_cascade_thermal_pairs=False,
                photonuclear_omitted_reactions=[11],photonuclear_cutoff_GeV=1.,
                response_identity=bank.identity if bank is not None else None)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if make_plot: plot_history(h,out)
    return report


def plot_history(h,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    t=h.time_s
    axes[0,0].loglog(t,h.temperature_MeV,label='T gamma')
    axes[0,0].loglog(t,h.neutrino_temperature_MeV,label='T neutrino')
    axes[0,0].set_ylabel('Temperature [MeV]');axes[0,0].legend()
    eta=h.baryon_density_MeV3/(.243587656435*h.temperature_MeV**3)
    axes[0,1].semilogx(t,eta/h.model.eta_cmb)
    axes[0,1].set_ylabel('eta(t) / eta CMB')
    S=h.entropy_MeV3*np.exp(3*h.loga)
    axes[1,0].semilogx(t,S/S[0]);axes[1,0].set_ylabel('Comoving total SM entropy / initial')
    axes[1,1].loglog(t,h.hubble_per_s);axes[1,1].set_ylabel('H [1/s]')
    for ax in axes.flat: ax.set_xlabel('Time [s]');ax.grid(alpha=.2)
    fig.suptitle(f'X -> gamma gamma: mX={h.model.mass_GeV:g} GeV, tau={h.model.lifetime_s:g} s, YX={h.model.yield_X:g}\nBlackbody thermalization; freely redshifting neutrinos')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(Path(out)/f'background_history.{ext}',dpi=150)
    plt.close(fig)


_WORKER_BANK=None

def scan_task(task):
    global _WORKER_BANK
    model,root,bank_path,baseline=task
    if _WORKER_BANK is None: _WORKER_BANK=ResponseBank(bank_path)
    identity=dict(model=asdict(model),response=_WORKER_BANK.identity,evolution_source=fingerprint(),
                  background_code=hashlib.sha256((ROOT/'src/bbn_cascade/reheating.py').read_bytes()).hexdigest())
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    out=Path(root)/'points'/key
    result=out/'result.json'
    if result.exists() and (out/'run_identity.json').exists() and json.loads((out/'run_identity.json').read_text())==identity:
        return json.loads(result.read_text())
    report=run_point(model,out,_WORKER_BANK,baseline=baseline,make_plot=False,save_histories=False)
    (out/'run_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return report


def scan(output,bank_path,*,mass=200.,eta_cmb=ETA,points=25,ymin=1e-18,ymax=1e-6,workers=8):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if not 0<ymin<ymax or points<2: raise ValueError('Invalid scan grid')
    taus=np.geomspace(1e2,1e8,points);ys=np.geomspace(ymin,ymax,points)
    baselines=[];backend=Backend()
    for i,tau in enumerate(taus):
        ref=out/'references'/f'tau_{i:03d}'
        zero=evolve(ReheatingModel(mass,float(tau),0.,eta_cmb));zero.save(ref)
        baselines.append(backend.solve_background(ref/'cosmo_file.dat',eta_cmb=eta_cmb,tau=float(tau),end=zero.metadata['end_time_s']))
    ratios=np.full((points,points,4),np.nan);dilution=np.zeros((points,points));caution=np.zeros((points,points),bool)
    tasks=[(ReheatingModel(mass,float(tau),float(y),eta_cmb),str(out),str(bank_path),baselines[i]) for y in ys for i,tau in enumerate(taus)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(scan_task,task):index for index,task in enumerate(tasks)}
        for count,f in enumerate(as_completed(futures),1):
            index=futures[f];j,i=divmod(index,points);r=f.result()
            ratios[j,i]=r['ratios'];dilution[j,i]=r['background']['photon_comoving_growth']
            caution[j,i]=r['background']['blackbody_caution']
            print(f'reheating scan {count}/{len(tasks)}',flush=True)
    np.savez(out/'scan.npz',lifetime_s=taus,yield_X=ys,ratios=ratios,photon_growth=dilution,blackbody_caution=caution)
    np.savetxt(out/'scan.csv',[[t,y,*ratios[j,i],dilution[j,i],int(caution[j,i])] for j,y in enumerate(ys) for i,t in enumerate(taus)],
               delimiter=',',header='tau_s,YX,D_ratio,He3_ratio,He4_ratio,Li7_ratio,photon_comoving_growth,blackbody_caution')
    plot_scan(out,mass)


def plot_scan(output,mass):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    out=Path(output);data=np.load(out/'scan.npz');t=data['lifetime_s'];y=data['yield_X']
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    for k,(ax,name) in enumerate(zip(axes.flat,['D/H','(He3+T)/H','He4 mass fraction','(Li7+Be7)/H'])):
        r=data['ratios'][:,:,k]
        mesh=ax.pcolormesh(t,y,np.log10(r),shading='auto',rasterized=True,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1))
        levels=[v for v in [.1,.5,.9,.99,1.01,1.1,2,10] if np.min(r)<v<np.max(r)]
        if levels: ax.clabel(ax.contour(t,y,r,levels=levels,colors='k',linewidths=.7),fmt='%g',fontsize=7)
        if np.any(data['blackbody_caution']):
            ax.contourf(t,y,data['blackbody_caution'].astype(float),levels=[.5,1.5],colors='none',hatches=['///'])
        ax.set(xscale='log',yscale='log',xlabel='Lifetime [s]',ylabel='Initial YX = nX/s',title=name)
    fig.colorbar(mesh,ax=list(axes.flat),label='log10(abundance / standard BBN)',extend='both')
    fig.suptitle(f'X -> gamma gamma, mX={mass:g} GeV; entropy and expansion included\nFinal eta fixed to CMB; hatching: blackbody thermalization not assured')
    for ext in ['png','pdf']:fig.savefig(out/f'abundance_contours.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['standard','bank','background','point','scan'])
    p.add_argument('--output',type=Path,default=ROOT/'results/reheating_200GeV')
    p.add_argument('--bank',type=Path,default=DEFAULT_BANK)
    p.add_argument('--mass',type=float,default=200.)
    p.add_argument('--lifetime',type=float,default=1e5)
    p.add_argument('--yield-x',type=float,default=1e-8)
    p.add_argument('--eta-cmb',type=float,default=ETA)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--points-per-decade',type=float,default=120.)
    p.add_argument('--temperature-intervals',type=int,default=10)
    p.add_argument('--scan-points',type=int,default=25)
    p.add_argument('--yield-min',type=float,default=1e-18)
    p.add_argument('--yield-max',type=float,default=1e-6)
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    model=ReheatingModel(a.mass,a.lifetime,a.yield_x,a.eta_cmb)
    if a.stage=='standard':standard(a.output,a.eta_cmb)
    elif a.stage=='bank':
        standard(a.output/'reference',a.eta_cmb)
        prepare_bank(a.bank,a.output/'reference',mass=a.mass,ppd=a.points_per_decade,intervals=a.temperature_intervals,workers=a.workers)
    elif a.stage=='background':
        h=evolve(model);h.save(a.output);plot_history(h,a.output)
    elif a.stage=='point':
        report=run_point(model,a.output,ResponseBank(a.bank))
        print(json.dumps(dict(eta_initial=report['background']['eta_initial'],eta_final=report['background']['eta_final'],ratios=report['ratios']),indent=2))
    else:scan(a.output,a.bank,mass=a.mass,eta_cmb=a.eta_cmb,points=a.scan_points,ymin=a.yield_min,ymax=a.yield_max,workers=a.workers)

if __name__=='__main__':main()
