"""Summarize isolated Moroi-code interventions and common-cross-section checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bbn_cascade.hybrid_solver import CascadeSpectrum
from bbn_cascade.photodisintegration import calculate_rates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/high_temperature_diagnosis'
IDS=[1,2,3,4,5,6,7,9]
TEMPS=[1e-7,1e-6,1e-5,1e-4,10**-3.5,1e-3]
MODES=['original','dp_endpoint_fixed','dp_current','dp_current_ic_loss','dp_fixed_ic_loss']
LABELS=['Original old code','Pair quadrature fixed','+ current pair kinematics','+ IC loss upper limit aligned','Pair quadrature + IC loss fixed']


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    z=np.load(ROOT/'results/temperature_rates_E0_100GeV_eta6p1e-10/rates.npz')
    supplied=np.loadtxt(ROOT/'rate_moroi.dat');records=[];common=[];reproductions=[]
    ratios=np.zeros((len(MODES),len(TEMPS),8))
    for j,t in enumerate(TEMPS):
        ix=int(np.argmin(abs(z['temperature_GeV']-t)))
        assert np.isclose(z['temperature_GeV'][ix],t,rtol=1e-12,atol=0)
        new=z['rate_GeV'][ix,np.array(IDS)-1]
        for k,mode in enumerate(MODES):
            p=OUT/'old_code'/f'{mode}_T{t:.0e}'
            old=np.loadtxt(p/'rates.dat')
            assert np.isclose(old[2],t,rtol=1e-12,atol=0)
            old_rates=old[3:11]/100;ratios[k,j]=old_rates/new
            if mode=='original':
                reference=supplied[np.argmin(abs(supplied[:,2]-t)),3:11]/100
                reproductions.append(float(np.max(abs(old_rates/reference-1))))
            for i,rid in enumerate(IDS):
                records.append(dict(temperature_eV=t*1e9,variant=mode,reaction_id=rid,old_rate_Q1_GeV=old_rates[i],new_rate_Q1_GeV=new[i],old_over_new=ratios[k,j,i]))
            if mode=='dp_current_ic_loss':
                a=np.loadtxt(p/'spectrum.dat');e=a[:,0];edges=np.r_[e[0],np.sqrt(e[:-1]*e[1:]),e[-1]]
                spec=CascadeSpectrum(e,a[:,1]/100,a[:,2]/100,{},dict(injected_energy_rate=1.),edges)
                rates=calculate_rates(spec)
                for i,rid in enumerate(IDS):
                    r=rates[rid-1].rate_GeV
                    common.append(dict(temperature_eV=t*1e9,reaction_id=rid,old_spectrum_new_cross_section_rate_GeV=r,new_spectrum_new_cross_section_rate_GeV=new[i],spectrum_ratio=r/new[i],old_rate_over_common_cross_section=old_rates[i]/r))
    write_csv('rate_interventions.csv',records);write_csv('common_cross_sections.csv',common)
    names=[str(z['reaction_name'][i-1]) for i in IDS]
    fig,axes=plt.subplots(4,2,figsize=(12,13),sharex=True)
    for i,ax in enumerate(axes.flat):
        for k in range(4): ax.loglog(np.array(TEMPS)*1e9,ratios[k,:,i],'-o',ms=3,label=LABELS[k])
        ax.axhline(1,color='0.4',lw=.8);ax.set_title(names[i]);ax.set_ylabel('Old variant / current');ax.grid(True,which='both',alpha=.2)
        if i==0:ax.legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Bath temperature [eV]')
    fig.suptitle('Controlled interventions in the old code; same energy injection\nCross sections remain old in each Fortran variant',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'interventions.png',dpi=170);fig.savefig(OUT/'interventions.pdf');plt.close(fig)
    # Separate spectral differences from differences in nuclear cross sections.
    fig,ax=plt.subplots(figsize=(9,6))
    for i,rid in enumerate(IDS):
        vals=[r['spectrum_ratio'] for r in common if r['reaction_id']==rid]
        ax.semilogx(np.array(TEMPS)*1e9,vals,'-o',label=names[i])
    ax.axhline(1,color='0.4',lw=.8);ax.set_xlabel('Bath temperature [eV]');ax.set_ylabel('Aligned old spectrum / new spectrum\n(both convolved with current cross sections)');ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'common_cross_sections.png',dpi=170);plt.close(fig)
    probes=[]
    a=np.loadtxt(OUT/'old_code/ic_loss_original.dat');b=np.loadtxt(OUT/'old_code/ic_loss_dp_current_ic_loss.dat')
    for x,y in zip(a,b):
        probes.append(dict(temperature_eV=x[0]*1e9,electron_GeV=x[1],original_loss_GeV=x[2],extended_loss_GeV=y[2],original_over_extended=x[2]/y[2]))
    write_csv('ic_loss_upper_limit.csv',probes)
    checks=[]
    for name in ('new_q12','new_ppd240'):
        p=OUT/name/'rates.csv'
        if p.exists():
            with p.open() as f:rows=list(csv.DictReader(f))
            vals=np.array([float(row['rate_GeV']) for row in rows]);reference=z['rate_GeV'][0]
            checks.append(dict(variant=name,D_over_baseline=float(vals[0]/reference[0]),max_relative_rate_change_17=float(np.max(abs(vals/reference-1)))))
    summary=dict(temperatures_eV=[t*1e9 for t in TEMPS],variants=MODES,
        maximum_old_reproduction_relative_difference=max(reproductions),
        D_ratios_at_1MeV=dict(zip(MODES,ratios[:,-1,0].tolist())),
        all_8_aligned_old_over_new_at_1MeV=ratios[3,-1].tolist(),
        common_cross_section_ratios_at_1MeV=[r['spectrum_ratio'] for r in common if r['temperature_eV']==1e6],
        numerical_checks=checks,
        notes=['Original Moroi file is never edited. Production code and temperature tables unchanged.',
               'Aligned variants are controlled diagnostics, not a physical high-temperature model upgrade.',
               'Both original models omit thermal pairs; this shared limitation alone does not explain their difference.',
               'Common-cross-section convolution treats old point values as cell-constant densities on geometric boundaries; it also changes the old integration cutoff/quadrature.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
