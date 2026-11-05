"""Fixed-background AlterAlterBBN adapter and reproducible decay scans.

C state is process-local. Do not call Backend from multiple threads.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
from functools import lru_cache
import numpy as np
from .decay import HBAR, CHARGES

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'vendor/alteralterbbn'
ETA = 6.137e-10


def fingerprint():
    h = hashlib.sha256()
    for base, glob in [(VENDOR/'src', '*.[ch]'), (ROOT/'src/bbn_cascade', '*.py')]:
        for p in sorted(base.glob(glob)):
            if p.name == 'bbn_decay.py': continue
            h.update(p.name.encode()); h.update(p.read_bytes())
    h.update((VENDOR/'io/sm/cosmo_file.dat').read_bytes())
    return h.hexdigest()


def build():
    dest = VENDOR/'bin/libdecay.so'
    dest.parent.mkdir(exist_ok=True)
    sources = sorted((VENDOR/'src').glob('*.c'))
    if not dest.exists() or max(p.stat().st_mtime for p in [*sources, VENDOR/'src/include.h']) > dest.stat().st_mtime:
        subprocess.run(['cc', '-shared', '-fPIC', '-O3', '-std=c99', '-Wl,-Bsymbolic',
                        *map(str, sources), '-lm', '-o', str(dest)], check=True)
    return dest


class Backend:
    def __init__(self, library=None):
        self.lib = ctypes.CDLL(str(library or build()))
        self.run = self.lib.decay_run
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_int, ptr, ptr]
        self.run.restype = ctypes.c_int

    def solve(self, table=None, *, tau=1., yield_X=0., end=3e9, history=None):
        if not np.isfinite(tau) or tau <= 0 or not np.isfinite(yield_X) or yield_X < 0:
            raise ValueError('Invalid decay parameters')
        data = np.empty((0,18)) if table is None else np.ascontiguousarray(table, dtype=float)
        if table is not None and (len(data)<2 or data.shape[1]!=18 or np.any(~np.isfinite(data)) or np.any(data<0) or np.any(np.diff(data[:,0])<=0)):
            raise ValueError('Invalid time-response table')
        if not np.isfinite(end) or end < 0:
            raise ValueError('Invalid final time')
        if table is not None:
            c,_=cosmo()
            t0=loginterp(100/11604.52165624311,c[::-1,1]*1e-3,c[::-1,0])
            tend=end if end else loginterp(.01/11604.52165624311,c[::-1,1]*1e-3,c[::-1,0])
            if data[0,0]>t0 or data[-1,0]<1.05*tend:
                raise ValueError('Response table does not cover integration including endpoint margin')
        # Keep the same adaptive-step path for the zero-injection reference.
        effective_yield=yield_X
        if table is not None and yield_X == 0:
            data=data.copy(); data[:,1:]=0.; effective_yield=1.
        out = np.zeros(26)
        status = self.run(str(VENDOR/'io/sm/cosmo_file.dat').encode(),
                          str(history).encode() if history else b'', ETA, end,
                          tau, effective_yield, len(data), data, out)
        if status or np.any(~np.isfinite(out)) or np.any(out<0):
            raise RuntimeError(f'BBN failure {status}')
        weights=np.array([1,1,2,3,3,4,6,7,7,8,8,9,10,11,11,12,12,12,13,13,14,14,14,15,15,16])
        if abs(out@weights-1)>1e-4:
            raise RuntimeError('Baryon conservation failed')
        return out


    def solve_background(self, cosmology, rates=None, *, eta_cmb=ETA, tau=1e5, end=None, history=None):
        """Evolve actual supplied nb(t), T(t), Tnu(t); rates are directly in s^-1."""
        path=Path(cosmology).resolve();c=np.loadtxt(path)
        if c.ndim!=2 or c.shape[1]!=6 or np.any(~np.isfinite(c)) or np.any(np.diff(c[:,0])<=0) or np.any(c[:,[0,1,3,4,5]]<=0):
            raise ValueError('Invalid external cosmology')
        stop=float(end or c[-1,0]/1.1)
        if not c[0,0]<stop<c[-1,0] or not np.isfinite(eta_cmb) or eta_cmb<=0 or not np.isfinite(tau) or tau<=0:
            raise ValueError('Invalid eta, lifetime or final time')
        data=np.empty((0,18)) if rates is None else np.ascontiguousarray(rates,dtype=float)
        if rates is not None and (data.ndim!=2 or data.shape[1]!=18 or len(data)<2 or np.any(~np.isfinite(data)) or np.any(data<0) or np.any(np.diff(data[:,0])<=0) or data[0,0]>c[0,0]*(1+1e-12) or data[-1,0]<stop*1.02):
            raise ValueError('Physical rate table does not cover the background')
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        run=self.lib.decay_run_background
        run.argtypes=[ctypes.c_char_p,ctypes.c_char_p,*([ctypes.c_double]*4),ctypes.c_int,ptr,ptr]
        run.restype=ctypes.c_int
        out=np.zeros(26)
        status=run(str(path).encode(),str(history).encode() if history else b'',eta_cmb,stop,
                   float(c[0,0]*(1+1e-10)),tau,len(data),data,out)
        weights=np.array([1,1,2,3,3,4,6,7,7,8,8,9,10,11,11,12,12,12,13,13,14,14,14,15,15,16])
        if status or np.any(~np.isfinite(out)) or np.any(out<0) or abs(out@weights-1)>1e-4:
            raise RuntimeError(f'External-background BBN failed: {status}')
        return out


def observables(y):
    return np.array([y[2]/y[1], (y[3]+y[4])/y[1], 4*y[5], (y[7]+y[8])/y[1]])


def loginterp(x, xp, fp):
    x = np.asarray(x)
    xp=np.asarray(xp); fp=np.asarray(fp)
    if xp.ndim!=1 or fp.shape!=xp.shape or len(xp)<2 or np.any(np.diff(xp)<=0) or any(np.any(~np.isfinite(v)) or np.any(v<=0) for v in (x,xp,fp)):
        raise ValueError('Log interpolation requires positive finite data and an increasing grid')
    if np.any(x < xp[0]) or np.any(x > xp[-1]):
        raise ValueError('Interpolation outside supplied range')
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))


def background(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    backend=Backend()
    y=backend.solve(history=output/'standard_history.dat')
    np.save(output/'standard_final.npy',y)
    return y


@lru_cache(maxsize=1)
def cosmo():
    c=np.loadtxt(VENDOR/'io/sm/cosmo_file.dat')
    # Entropy per conserved baryon-normalization density, calibrated after e+- annihilation.
    T,Tnu=c[-1,1:4:2]*1e-3
    sf=2*np.pi**2/45*(2*T**3+5.25*Tnu**3)
    entropy=c[:,5]*1e-9*(sf/(c[-1,5]*1e-9))
    return c,entropy


def response_task(task):
    i,T,eta,he,mass,ppd,out,identity=task
    from .config import CascadeConfig
    from .hybrid_solver import CascadeSolver
    from .photodisintegration import calculate_rates
    path=Path(out)/f'response_{i:03d}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['identity'])!=identity: raise ValueError('Incompatible response checkpoint')
        return i
    config=CascadeConfig(T,mass/2,eta_baryon=eta,helium_mass_fraction=he,
                         points_per_decade=ppd,quadrature_order=8)
    spectrum=CascadeSolver(config).solve()
    rates=calculate_rates(spectrum)
    # Residence-time diagnostic for photons capable of nuclear breakup.
    widths=np.diff(spectrum.boundaries)
    keep=spectrum.energy>=1.586627e-3
    residence=np.sum(spectrum.photon[keep]*widths[keep])*mass/2*HBAR
    np.savez(path, K=[r.response_GeV_minus4 for r in rates], temperature=T,
             eta=eta, helium=he, residence_s=residence, identity=identity)
    return i


def prepare(output, mass=200., ppd=120., intervals=10, workers=8):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=json.dumps(dict(source=fingerprint(),mass=mass,ppd=ppd,intervals=intervals),sort_keys=True)
    marker=output/'identity.json'
    if marker.exists() and marker.read_text()!=identity: raise ValueError('Use a new output directory for changed settings/source')
    marker.write_text(identity)
    background(output)
    hist=np.loadtxt(output/'standard_history.dat')
    c,s=cosmo()
    # Cover the initial 100 GK step and final-time overshoot without extrapolation.
    temperatures=np.geomspace(.009,1e-8,1+int(np.ceil(np.log10(.009/1e-8)*intervals)))
    times=loginterp(temperatures,c[::-1,1]*1e-3,c[::-1,0])
    eta=[];he=[]
    for t in times:
        y=np.array([np.interp(np.log(t),np.log(hist[:,0]),hist[:,j]) for j in range(1,10)])
        # Match both sum Z*n and sum Z^2*n in existing H/He target parametrization.
        z=float(CHARGES@y);z2=float(CHARGES**2@y)
        nb=ETA*loginterp(t,c[:,0],c[:,5]*1e-9)
        eta.append(z2*nb/(.243587656435*loginterp(t,c[:,0],c[:,1]*1e-3)**3))
        he.append(float(np.clip(2*(1-z/z2),0,1)))
    tasks=[(i,float(T),eta[i],he[i],mass,ppd,str(output),identity) for i,T in enumerate(temperatures)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(response_task,t) for t in tasks]):
            print(f'response {f.result()+1}/{len(tasks)} complete',flush=True)
    K=np.array([np.load(output/f'response_{i:03d}.npz')['K'] for i in range(len(tasks))])
    # rate = K * m*s * Y_X * exp(-t/tau)/tau; hbar cancels Q conversion.
    table=np.column_stack([times,K*mass*loginterp(times,c[:,0],s)[:,None]])
    np.save(output/'time_response.npy',table)
    np.savez(output/'response_table.npz',temperature_GeV=temperatures,time_s=times,K=K)
    return table



def scan_point(task):
    output,i,j,tau,y=task
    output=Path(output)
    table=np.load(output/'time_response.npy')
    key=hashlib.sha256(table.tobytes()+f'{tau:.17g},{y:.17g}'.encode()).hexdigest()[:24]
    path=output/'points'/f'{key}.npy'
    if path.exists():
        result=np.load(path)
        if result.shape!=(26,) or np.any(~np.isfinite(result)) or np.any(result<0):
            raise ValueError('Invalid abundance checkpoint')
    else:
        result=Backend().solve(table,tau=tau,yield_X=y)
        tmp=path.with_suffix('.tmp.npy');np.save(tmp,result);tmp.replace(path)
    return i,j,result

def scan(output, points=25, tau_min=1e2, tau_max=1e8, workers=8):
    if points<3 or not 1e2<=tau_min<tau_max<=1e8:
        raise ValueError('This prepared background supports 1e2 <= tau_min < tau_max <= 1e8 s and at least 3 scan points')
    output=Path(output); backend=Backend()
    identity=json.loads((output/'identity.json').read_text())
    if identity['source']!=fingerprint(): raise ValueError('Source changed since response generation')
    mass=identity['mass']; table=np.load(output/'time_response.npy')
    baseline=None
    c,s=cosmo(); t=c[:,0]
    # Friedmann radiation density from supplied standard H; reduced Planck mass.
    rho=3*(2.435e18)**2*(c[:,4]*1e-3)**2
    taus=np.geomspace(tau_min,tau_max,points)
    baseline=np.array([observables(backend.solve(table,tau=tau,yield_X=0)) for tau in taus])
    limits=np.array([.01/np.max(mass*s*np.exp(-t/tau)/rho) for tau in taus])
    upper=max(limits); lower=min(limits)*1e-6
    while True:
        low=np.array([observables(backend.solve(table,tau=tau,yield_X=lower)) for tau in taus])/baseline
        if np.max(abs(low-1))<.001: break
        lower/=10
    yields=np.geomspace(lower,upper,points)
    ratios=np.full((points,points,4),np.nan)
    cache=output/'points';cache.mkdir(exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    jobs=[]
    for j,y in enumerate(yields):
        for i,tau in enumerate(taus):
            if y<=limits[i]*(1+1e-12): jobs.append((str(output),i,j,float(tau),float(y)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for count,f in enumerate(as_completed([pool.submit(scan_point,job) for job in jobs]),1):
            i,j,result=f.result()
            ratios[j,i]=observables(result)/baseline[i]
            if count%100==0 or count==len(jobs): print(f'scan {count}/{len(jobs)}',flush=True)
    np.savez(output/'scan.npz',lifetime_s=taus,yield_X=yields,ratios=ratios,valid_yield_max=limits,baseline=baseline)
    rows=[[tau,y,*ratios[j,i]] for j,y in enumerate(yields) for i,tau in enumerate(taus)]
    np.savetxt(output/'scan.csv',rows,delimiter=',',header='lifetime_s,yield_X,D_ratio,He3_ratio,He4_ratio,Li7_ratio')
    metadata=dict(mass_GeV=mass, channel='X -> gamma gamma', eta_final=ETA,
        yield_definition='undecayed n_X / total conserved standard entropy density',
        source_sha256=identity['source'], upstream_commit=(VENDOR/'UPSTREAM_COMMIT').read_text().strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['standard-background response fixed; no composition feedback',
                'thermal e+- pairs omitted in cascade, retained in standard cosmology',
                'reaction 11 omitted: inclusive A=3 daughter multiplicities unresolved',
                'photonuclear convolution cutoff 1 GeV',
                'secondary reactions of nonthermal nuclei omitted',
                'thermal nuclear reactions frozen below original BBN endpoint T9=0.01; weak decays retained'],
        dilute_fraction_max=.01, end_time_s=3e9,
        temperature_intervals_per_decade=identity['intervals'], energy_points_per_decade=identity['ppd'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plot(output)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    output=Path(output); a=np.load(output/'scan.npz')
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    for k,(ax,name) in enumerate(zip(axes.flat,['D/H','(He3+T)/H','He4 mass fraction','(Li7+Be7)/H'])):
        r=a['ratios'][:,:,k]; z=np.ma.masked_invalid(np.log10(r))
        mesh=ax.pcolormesh(a['lifetime_s'],a['yield_X'],z,shading='auto',rasterized=True,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1))
        levels=[x for x in [.1,.5,.9,.99,1.01,1.1,2,10] if np.nanmin(r)<x<np.nanmax(r)]
        if levels:
            cs=ax.contour(a['lifetime_s'],a['yield_X'],r,levels=levels,colors='k',linewidths=.7)
            label_positions=[]
            for level,segments in zip(levels,cs.allsegs):
                candidates=[]
                for segment in segments:
                    if len(segment)<2: continue
                    boundary=np.exp(np.interp(np.log(segment[:,0]),np.log(a['lifetime_s']),np.log(a['valid_yield_max'])))
                    usable=segment[(segment[:,1]<.3*boundary)&(segment[:,0]<5e7)]
                    if len(usable)>1: candidates.append(usable)
                if candidates:
                    segment=max(candidates,key=len)
                    point=segment[len(segment)//2]
                    label_positions.append(tuple(point))
            if label_positions:
                ax.clabel(cs,manual=label_positions,fmt='%g',fontsize=7,inline_spacing=6)

        ax.plot(a['lifetime_s'],a['valid_yield_max'],'--',color='gray')
        ax.set(xscale='log',yscale='log',title=name,xlabel='Lifetime [s]',ylabel=r'$Y_X=n_X/s$ (before decay)')
    fig.colorbar(mesh,ax=list(axes.flat),label='log10(abundance / standard BBN)',shrink=.8,extend='both')
    mass=json.loads((output/'identity.json').read_text())['mass']
    size=len(a['lifetime_s'])
    fig.suptitle(f'X → γγ, mX={mass:g} GeV; {size} × {size} grid; fixed standard BBN background\nNo thermal-pair cascade background; reaction 11 omitted; photonuclear cutoff 1 GeV',fontsize=11)
    reports=[output/name for name in ['integration_validation.json','response_validation.json']]
    if any(path.exists() and not json.loads(path.read_text())['passed'] for path in reports):
        fig.text(.5,.01,'Numerical convergence is not uniform; see validation reports.',ha='center',fontsize=8)
    for ext in ['png','pdf']: fig.savefig(output/f'abundance_contours.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)



def contour_samples(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result={}
    for k in range(4):
        r=data['ratios'][:,:,k]
        for level in [.1,.5,.9,.99,1.01,1.1,2,10]:
            if not np.nanmin(r)<level<np.nanmax(r): continue
            fig,ax=plt.subplots()
            cs=ax.contour(data['lifetime_s'],data['yield_X'],r,levels=[level])
            pts=[]
            for seg in cs.allsegs[0]:
                seg=np.log10(seg)
                for a,b in zip(seg[:-1],seg[1:]):
                    pts.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.01)+1)))
            plt.close(fig)
            if pts: result[(k,level)]=np.array(pts)
    return result


def refine(output, max_points=385, workers=8):
    from scipy.spatial import cKDTree
    output=Path(output)
    report_path=output/'contour_convergence.json'
    checks=json.loads(report_path.read_text()) if report_path.exists() else []
    while True:
        old=np.load(output/'scan.npz'); n=len(old['lifetime_s'])
        before=contour_samples(old)
        if 2*n-1>max_points:
            checks.append(dict(max_points=max_points,passed=False,note='Refinement limit reached; contour precision is not certified'))
            (output/'contour_convergence.json').write_text(json.dumps(checks,indent=2)+'\n')
            return
        scan(output,2*n-1,float(old['lifetime_s'][0]),float(old['lifetime_s'][-1]),workers=workers)
        after=contour_samples(np.load(output/'scan.npz'))
        distance=0.
        if before.keys()!=after.keys(): distance=float('inf')
        else:
            for key in before:
                a,b=before[key],after[key]
                distance=max(distance,float(np.max(cKDTree(a).query(b)[0])),float(np.max(cKDTree(b).query(a)[0])))
        checks.append(dict(old_points=n,new_points=2*n-1,max_contour_distance_dex=distance if np.isfinite(distance) else None,
                           passed=bool(distance<=.05)))
        (output/'contour_convergence.json').write_text(json.dumps(checks,indent=2)+'\n')
        if distance<=.05: return


def validate(output):
    """Check time stepping and late endpoint at representative physical scan points.

    Builds a separate temporary library with half-sized time-step parameters.
    The production library and response backgrounds are never modified.
    """
    import tempfile
    import shutil
    output=Path(output); data=np.load(output/'scan.npz'); table=np.load(output/'time_response.npy')
    records=[]
    with tempfile.TemporaryDirectory(prefix='bbn-validation-') as tmp:
        tmp=Path(tmp); shutil.copytree(VENDOR/'src',tmp/'src')
        p=tmp/'src/bbn.c'; source=p.read_text().replace('double cy  = 0.1;', 'double cy  = 0.05;').replace('double ct  = 0.005;', 'double ct  = 0.0025;').replace('0.02 * decay_tau','0.01 * decay_tau')
        p.write_text(source)
        lib=tmp/'libcheck.so'
        subprocess.run(['cc','-shared','-fPIC','-O3','-std=c99','-Wl,-Bsymbolic',*map(str,sorted((tmp/'src').glob('*.c'))),'-lm','-o',str(lib)],check=True)
        fine=Backend(lib); base=Backend()
        p.write_text((VENDOR/'src/bbn.c').read_text().replace('if (T9 >= 0.01) rate_all', 'if (T9 >= 0.005) rate_all'))
        cutofflib=tmp/'libcutoff.so'
        subprocess.run(['cc','-shared','-fPIC','-O3','-std=c99','-Wl,-Bsymbolic',*map(str,sorted((tmp/'src').glob('*.c'))),'-lm','-o',str(cutofflib)],check=True)
        cutoff=Backend(cutofflib)
        cases=[]
        for i in [0,len(data['lifetime_s'])//2,len(data['lifetime_s'])-1]:
            tau=float(data['lifetime_s'][i])
            cases.append(('strong_injection',tau,float(data['valid_yield_max'][i])*.5))
            departure=np.max(abs(data['ratios'][:,i]-1),axis=1)
            j=int(np.nanargmin(abs(departure-.1)))
            cases.append(('near_standard',tau,float(data['yield_X'][j])))
        for label,tau,y in cases:
            nominal=observables(base.solve(table,tau=tau,yield_X=y))
            tighter=observables(fine.solve(table,tau=tau,yield_X=y))
            later=observables(base.solve(table,tau=tau,yield_X=y,end=6e9))
            shifted=observables(cutoff.solve(table,tau=tau,yield_X=y))
            records.append(dict(case=label,lifetime_s=tau,yield_X=y,
                standard_ratios=(nominal/observables(base.solve(table,tau=tau,yield_X=0))).tolist(),
                thermal_cutoff_max_relative_difference=float(np.max(abs(shifted/nominal-1))),
                timestep_max_relative_difference=float(np.max(abs(tighter/nominal-1))),
                endpoint_max_relative_difference=float(np.max(abs(later/nominal-1)))))
    report=dict(points=records,passed=all(max(r['timestep_max_relative_difference'],r['endpoint_max_relative_difference'],r['thermal_cutoff_max_relative_difference'])<=.01 for r in records),
                energy_temperature_convergence='Not certified by these checks; independent response refinements required')
    (output/'integration_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def validate_responses(output, workers=3):
    """Independent two-grid checks at three temperatures relevant to the scan.

    A failed check is recorded, never silently promoted to a precision claim.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    output=Path(output); identity=json.loads((output/'identity.json').read_text())
    table=np.load(output/'time_response.npy')
    checks_dir=output/'response_checks';checks_dir.mkdir(exist_ok=True)
    target_times=[1e4,1e6,1e8]
    indices=sorted(set(int(np.argmin(abs(np.log(table[:,0]/t)))) for t in target_times))
    tasks=[]
    for i in indices:
        saved=np.load(output/f'response_{i:03d}.npz')
        tasks.append((i,float(saved['temperature']),float(saved['eta']),float(saved['helium']),
                      identity['mass'],identity['ppd']*2,str(checks_dir),json.dumps(identity,sort_keys=True)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(response_task,tasks))
    energy=[]
    for i in indices:
        coarse=np.load(output/f'response_{i:03d}.npz')['K']
        fine=np.load(checks_dir/f'response_{i:03d}.npz')['K']
        diff=np.divide(abs(coarse-fine),fine,out=np.zeros_like(fine),where=fine>0)
        energy.append(dict(index=i,time_s=float(table[i,0]),relative_differences=diff.tolist(),
                           max_relative_difference=float(max(diff)),passed=bool(max(diff)<=.01)))
    # Compare every-other temperature node with the production table at fixed models.
    keep=np.unique(np.r_[np.arange(0,len(table),2),len(table)-1])
    coarser=table[keep]
    data=np.load(output/'scan.npz');backend=Backend();temps=[]
    cases=[]
    for i in [0,len(data['lifetime_s'])//2,len(data['lifetime_s'])-1]:
        tau=float(data['lifetime_s'][i])
        cases.append(('strong_injection',tau,float(data['valid_yield_max'][i])*.5))
        departure=np.max(abs(data['ratios'][:,i]-1),axis=1)
        j=int(np.nanargmin(abs(departure-.1)))
        cases.append(('near_standard',tau,float(data['yield_X'][j])))
    for label,tau,y in cases:
        fine=observables(backend.solve(table,tau=tau,yield_X=y))
        coarse=observables(backend.solve(coarser,tau=tau,yield_X=y))
        diff=abs(coarse/fine-1)
        temps.append(dict(case=label,lifetime_s=tau,yield_X=y,relative_differences=diff.tolist(),
                          max_relative_difference=float(max(diff)),passed=bool(max(diff)<=.01)))
    report=dict(energy_grid=energy,temperature_grid=temps,
        passed=all(r['passed'] for r in energy+temps),
        note='Representative two-grid comparisons, not global absolute-error guarantees.')
    (output/'response_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def stationarity_task(task):
    """First temporal moment of the linear cascade's impulse response.

    If A N0 = S, then A N1 = N0. A nuclear-rate-weighted N1/N0 is
    the mean delay of that nuclear response, in natural time units.
    """
    output,index=task;output=Path(output)
    from dataclasses import replace
    from .config import CascadeConfig
    from .photodisintegration import calculate_rates
    from . import hybrid_solver
    saved=np.load(output/f'response_{index:03d}.npz')
    identity=json.loads((output/'identity.json').read_text())
    original=hybrid_solver.spsolve; moments=[]
    def capture(matrix,source):
        populations=original(matrix,source)
        if not moments: moments.append(original(matrix,populations))
        return populations
    hybrid_solver.spsolve=capture
    try:
        spectrum=hybrid_solver.CascadeSolver(CascadeConfig(
            float(saved['temperature']),identity['mass']/2,
            eta_baryon=float(saved['eta']),helium_mass_fraction=float(saved['helium']),
            points_per_decade=60,quadrature_order=8)).solve()
    finally:
        hybrid_solver.spsolve=original
    n=len(spectrum.energy);photon_moment=moments[0][:n]/np.diff(spectrum.boundaries)
    if np.min(photon_moment)<-1e-10*np.max(abs(photon_moment)):
        raise RuntimeError('Negative cascade time moment')
    rates=calculate_rates(spectrum)
    weighted=calculate_rates(replace(spectrum,photon=np.maximum(photon_moment,0)))
    delays=np.array([HBAR*b.rate_GeV/a.rate_GeV if a.rate_GeV>0 else 0 for a,b in zip(rates,weighted)])
    c,_=cosmo(); t=float(loginterp(float(saved['temperature']),c[::-1,1]*1e-3,c[::-1,0]))
    hubble=float(loginterp(t,c[:,0],c[:,4]*1e-3/HBAR))
    # Characteristic source variation at tau=t: dilution plus exponential decay.
    ratio=float(max(delays)*(3*hubble+1/t))
    return dict(index=index,time_s=t,temperature_GeV=float(saved['temperature']),
                mean_response_delay_s=delays.tolist(),
                max_delay_over_source_timescale_at_tau_equals_t=ratio,
                passed=ratio<.1)


def validate_stationarity(output,workers=3):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    output=Path(output);table=np.load(output/'time_response.npy')
    indices=sorted(set(int(np.argmin(abs(np.log(table[:,0]/t)))) for t in [1e4,1e6,1e8]))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(stationarity_task,[(str(output),i) for i in indices]))
    report=dict(points=records,passed=all(r['passed'] for r in records),
                note='Nuclear-response-weighted mean delay at 60 energy points/decade; representative checks, not a global spectral-gap bound.')
    (output/'stationarity_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['background','prepare','scan','plot','validate','validate-responses','validate-stationarity','refine','all'])
    p.add_argument('--output',type=Path,default=ROOT/'results/decay_200GeV')
    p.add_argument('--mass',type=float,default=200.)
    p.add_argument('--points-per-decade',type=float,default=120.)
    p.add_argument('--temperature-intervals',type=int,default=10)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--scan-points',type=int,default=25)
    args=p.parse_args()
    if args.workers<1 or args.scan_points<3 or args.temperature_intervals<1:
        p.error('workers >= 1, scan-points >= 3, and temperature-intervals >= 1 are required')
    if not np.isfinite(args.mass) or args.mass<=.002 or not np.isfinite(args.points_per_decade) or args.points_per_decade<=0:
        p.error('mass > 0.002 GeV and positive finite points-per-decade are required')
    if args.stage=='background': background(args.output)
    if args.stage in ['prepare','all']: prepare(args.output,args.mass,args.points_per_decade,args.temperature_intervals,args.workers)
    if args.stage in ['scan','all']: scan(args.output,args.scan_points,workers=args.workers)
    if args.stage=='plot': plot(args.output)
    if args.stage=='validate': validate(args.output)
    if args.stage=='refine': refine(args.output,workers=args.workers)
    if args.stage=='validate-stationarity': validate_stationarity(args.output,min(args.workers,3))
    if args.stage=='validate-responses': validate_responses(args.output,min(args.workers,3))
    if args.stage=='all':
        refine(args.output,workers=args.workers)
        validate(args.output)
        validate_responses(args.output,min(args.workers,3))
        validate_stationarity(args.output,min(args.workers,3))
        plot(args.output)

if __name__=='__main__': main()
