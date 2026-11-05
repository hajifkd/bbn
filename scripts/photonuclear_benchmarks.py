"""Generate six rate tables from saved hybrid spectra; optional fresh grid checks.

Run: uv run python scripts/photonuclear_benchmarks.py [--check-ppd 30]
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

from bbn_cascade import (CascadeConfig, CascadeSolver, CascadeSpectrum,
                         REACTIONS, calculate_rates, cross_section)
from bbn_cascade.constants import EV, MB_TO_GEV_MINUS2
from bbn_cascade.photodisintegration import write_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ppd', type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root/'results/9412055_hybrid'
    out = root/'results/photodisintegration'
    out.mkdir(parents=True, exist_ok=True)
    combined, convergence, fine_combined = [], [], []
    for e0 in (100., 1000., 10000.):
        for temp in (1., 100.):
            started = time.monotonic()
            stem = f'E0_{e0:g}GeV_T_{temp:g}eV'
            file = source/(stem+'.csv')
            data = np.genfromtxt(file, delimiter=',', names=True)
            energy = data['energy_GeV']
            edges = np.r_[energy[0], np.sqrt(energy[:-1]*energy[1:]), energy[-1]]
            source_metadata = json.loads(file.with_suffix('.json').read_text()) if file.with_suffix('.json').exists() else {}
            cfg = CascadeConfig(**source_metadata['config']) if source_metadata else CascadeConfig(temp*EV, e0, grid_size=len(energy), quadrature_order=8)
            sp = CascadeSpectrum(energy, data['photon_GeV2'], data['electron_GeV2'], {},
                                 {'injected_energy_rate': 1.}, edges)
            rows = calculate_rates(sp)
            tighter = calculate_rates(sp, rtol=1e-10)
            metadata = dict(config=asdict(cfg), injection_rate_GeV5=1., source_metadata=source_metadata,
                            source_spectrum=str(file.relative_to(root)),
                            source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            provenance='Hybrid spectra with sidecar configuration when available; unit injection. Cell boundaries reconstructed by exact hybrid rule.',
                            quadrature_comparison_max_relative=max(abs(a.rate_GeV-b.rate_GeV)/max(a.rate_GeV, 1e-300) for a,b in zip(rows,tighter)))
            write_rates(out/stem, rows, metadata)
            combined.extend(dict(E0_GeV=e0, T_eV=temp, **asdict(row)) for row in rows)
            print(f'{stem}: 17 rates, integration {time.monotonic()-started:.1f}s', flush=True)
            if args.check_ppd:
                n = int(np.ceil(np.log10(e0/.001)*args.check_ppd))+1
                fresh = CascadeSolver(CascadeConfig(temp*EV, e0, grid_size=n, quadrature_order=8)).solve()
                fresh_rows = calculate_rates(fresh)
                fine_combined.extend(dict(E0_GeV=e0, T_eV=temp, **asdict(row)) for row in fresh_rows)
                directory = out/'convergence'/f'ppd_{args.check_ppd:g}'/stem
                write_rates(directory, fresh_rows, dict(config=asdict(CascadeConfig(temp*EV, e0, grid_size=n, quadrature_order=8)), diagnostics=fresh.diagnostics))
                fresh.to_csv(directory/'spectrum.csv')
                np.save(directory/'boundaries_GeV.npy', fresh.boundaries)
                for a,b in zip(rows,fresh_rows):
                    convergence.append(dict(E0_GeV=e0,T_eV=temp,reaction_id=a.reaction_id,
                                            baseline_grid=len(energy),check_grid=n,
                                            relative_difference=abs(a.rate_GeV-b.rate_GeV)/max(a.rate_GeV,1e-300)))
                print(f'  grid check complete after {time.monotonic()-started:.1f}s',flush=True)
    def save(name, rows):
        with (out/name).open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save('rates_all.csv',combined)
    if fine_combined:
        save(f'rates_all_ppd{args.check_ppd:g}.csv', fine_combined)
    if convergence:
        save(f'convergence_ppd_{args.check_ppd:g}.csv',convergence)
    fig,axes=plt.subplots(3,1,figsize=(9,12))
    for ax, group in zip(axes,(REACTIONS[:5],REACTIONS[5:9],REACTIONS[9:])):
        for r in group:
            e=np.geomspace(r.threshold_MeV*1e-3*(1+1e-5),1,1200)
            ax.loglog(e*1e3,cross_section(r.id,e)/MB_TO_GEV_MINUS2,label=f'{r.id}: {r.name}')
        ax.set(xlabel='Photon energy [MeV]',ylabel='Cross section [mb]',ylim=(1e-6,100))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'cross_sections.png',dpi=160);plt.close(fig)
    # Compare the adopted curves in the energy range of 1403.5995 Fig. 1.
    fig,ax=plt.subplots()
    for rid in (12,15):
        e=np.linspace(REACTIONS[rid-1].threshold_MeV+.001,10.,800)
        ax.plot(e,cross_section(rid,e*.001)/MB_TO_GEV_MINUS2,label=REACTIONS[rid-1].name)
    ax.set(xlabel='Photon energy [MeV]',ylabel='Cross section [mb]');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'mass7_updated.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
