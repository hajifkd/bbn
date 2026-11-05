"""Compare corrected-FP spectra and rates at 120 and 240 points/decade.

Default: the formerly worst case, E0=10 TeV and T=100 eV. The pass criterion
is <=1% change in each of the 17 photonuclear rates, relative to the 240 grid.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bbn_cascade import CascadeConfig, CascadeSolver, calculate_rates
from bbn_cascade.photodisintegration import write_rates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--energy-GeV',type=float,default=1e4)
    parser.add_argument('--temperature-eV',type=float,default=100.)
    parser.add_argument('--output',default='results/fp_flux_convergence')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    spectra=[];rate_sets=[]
    for ppd in [120,240]:
        start=time.monotonic()
        n=int(np.ceil(ppd*np.log10(args.energy_GeV/.001)))+1
        cfg=CascadeConfig(args.temperature_eV*1e-9,args.energy_GeV,grid_size=n,quadrature_order=8)
        print(f'Starting {ppd} points/decade: {n} grid points',flush=True)
        s=CascadeSolver(cfg).solve()
        folder=out/f'ppd{ppd}';folder.mkdir(exist_ok=True)
        s.to_csv(folder/'spectrum.csv');np.save(folder/'boundaries_GeV.npy',s.boundaries)
        rows=calculate_rates(s)
        write_rates(folder,rows,dict(config=asdict(cfg),diagnostics=s.diagnostics,
                                    points_per_decade=ppd,
                                    source_sha256=hashlib.sha256((folder/'spectrum.csv').read_bytes()).hexdigest(),
                                    elapsed_seconds=time.monotonic()-start))
        spectra.append(s);rate_sets.append(rows)
        print(f'Completed {ppd}: {time.monotonic()-start:.1f}s; D rate {rows[0].rate_GeV:.8e} GeV',flush=True)
    comparisons=[]
    for a,b in zip(*rate_sets):
        relative=(a.rate_GeV-b.rate_GeV)/b.rate_GeV if b.rate_GeV else (0. if a.rate_GeV==0 else None)
        comparisons.append(dict(reaction_id=a.reaction_id,reaction=a.reaction,
                                rate120_GeV=a.rate_GeV,rate240_GeV=b.rate_GeV,
                                relative_120_minus_240=relative,
                                within_one_percent=relative is not None and abs(relative)<=.01))
    with (out/'rate_comparison.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    a,b=spectra
    # Nested grids: compare interior differential densities, excluding the
    # injection line whose finite-volume height depends on its bin width.
    assert np.allclose(a.energy,b.energy[::2],rtol=1e-12,atol=0)
    ratio=np.divide(a.photon,b.photon[::2],out=np.ones_like(a.photon),where=b.photon[::2]>0)-1
    mask=(a.energy>=.001586627)&(a.energy<=1.)
    summary=dict(energy_GeV=args.energy_GeV,temperature_eV=args.temperature_eV,
                 criterion='abs(rate120/rate240-1) <= 0.01 for all 17 reactions',
                 all_rates_within_one_percent=all(r['within_one_percent'] for r in comparisons),
                 maximum_rate_relative_difference=max(abs(r['relative_120_minus_240']) for r in comparisons if r['relative_120_minus_240'] is not None),
                 maximum_photon_relative_difference_1p586627MeV_to_1GeV=float(np.max(abs(ratio[mask]))),
                 fp_flux='J=-(b+D_prime/2)f-D*f_prime/2',
                 limitation='One injection/temperature condition; a two-grid comparison is not an absolute error bound.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,(ax,ratio_ax)=plt.subplots(2,1,figsize=(8,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    for ppd,s in zip([120,240],spectra):
        ax.loglog(s.energy,s.photon,label=f'{ppd} points/decade')
    ax.set(ylabel=r'$f_\gamma$ [GeV$^2$]',title=f'E0={args.energy_GeV:g} GeV, T={args.temperature_eV:g} eV; corrected FP flux')
    ax.legend();ax.grid(alpha=.25)
    ratio_ax.semilogx(a.energy[:-1],100*ratio[:-1])
    ratio_ax.axhline(0,color='black',lw=.7)
    ratio_ax.set(xlabel='Photon energy [GeV]',ylabel='120/240 - 1 [%]')
    ratio_ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'spectra_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
