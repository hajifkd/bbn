"""Compare moroi_old.f cross sections on the SAME saved Q=1 spectrum.

Diagnostic only; does not change production physics. Fortran DATA tables are
parsed directly. Old number-injection normalization is deliberately removed.
"""
from pathlib import Path
import re
import csv
import json
import hashlib
import numpy as np
from scipy.integrate import quad
from bbn_cascade.photonuclear import cross_section, REACTIONS
from bbn_cascade.constants import MB_TO_GEV_MINUS2

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'moroi_old.f'
text = source.read_text(encoding='latin1')
base = ROOT / 'results/photodisintegration_eta6p1e-10/E0_100GeV_T_100eV'
out = ROOT / 'results/moroi_old_comparison'
out.mkdir(exist_ok=True)
spec = np.genfromtxt(base/'spectrum.csv', delimiter=',', names=True)
energy, density = spec['energy_GeV'], spec['photon_GeV2']
edges = np.load(base/'boundaries_GeV.npy')

def old_table(letter):
    block = text.split(f'      subroutine data_{letter}(')[1].split('      return')[0]
    def read(name):
        s = re.search(r'data '+name+r'/([^/]+)/', block).group(1)
        return np.array([float(v) for v in s.replace('&', '').replace('\n','').split(',')])
    e, s = read('egy')*.001, read('cs')*2.5681
    # Actual Fortran replaces its final DATA entry before use.
    e[-1] = e[-2]+.001
    s[-1] = s[-2]*e[-2]/e[-1]
    const = s[-1]*(e[-1]/np.sqrt(e[0]*(e[-1]-e[0])))**3
    def sigma(x):
        if x <= e[0]: return 0.
        if x < e[-2]: return float(np.interp(x, e, s))
        return const*(np.sqrt(e[0]*(x-e[0]))/x)**3
    return e, sigma

def old_d(x, electric_energy=None):
    q, m, alpha = .00223, .938, 1/137
    if x <= q: return 0.
    rho = np.sqrt(1/(2*m*q)); a1 = -5.50*rho
    y = x if electric_energy is None else electric_energy
    se = (8*np.pi/3)*(alpha/m)*np.sqrt(q)*max(y-q,0)**1.5/y**3/(1-.394)
    k = np.sqrt(2*m*(x-q))
    sm = (2/3)*np.pi*alpha*(2.79284739+1.9130427)**2/(2*m)**2*rho*k*(1-a1/rho)**2/((1+(rho*k)**2)*(1+(a1*k)**2))
    return se+sm

def integrate(fn, low, high, knots=(), continuous=False):
    cuts = sorted(set([low,high]+[float(x) for x in (energy if continuous else edges) if low<x<high]+[float(x) for x in knots if low<x<high]))
    total=0.
    for a,b in zip(cuts[:-1], cuts[1:]):
        if continuous:
            f=lambda x: np.exp(np.interp(x,energy,np.log(np.maximum(density,1e-300))))
        else:
            d=density[np.searchsorted(edges,(a+b)/2)-1]
            f=lambda x: d
        # Scale spectrum out to keep quadrature absolute tolerance meaningful.
        scale=f((a+b)/2)
        if scale: total+=scale*quad(lambda x: fn(x)*f(x)/scale,a,b,epsabs=1e-25,epsrel=1e-9)[0]
    return total

rows=[]
csrows=[]
for rid,letter in [(1,None),(4,'d'),(5,'e')]:
    if letter:
        knots, old = old_table(letter); q=knots[0]; upper=10*q
    else:
        knots=[]; old=old_d; q=.00223; upper=100*q
    new=lambda x: cross_section(rid,x)
    current=integrate(new,REACTIONS[rid-1].threshold_MeV*.001,1.)
    old_full=integrate(old,q,1.,knots)
    old_cut=integrate(old,q,upper,knots)
    old_cont=integrate(old,q,upper,knots,True)
    x=np.geomspace(q,upper,100); mid=np.sqrt(x[:-1]*x[1:])
    f=lambda v: np.exp(np.interp(v,energy,np.log(np.maximum(density,1e-300))))
    even=np.array([old(v) for v in x])
    odd=np.array([old(v) if letter else old_d(v,left) for v,left in zip(mid,x[:-1])])
    simpson=np.sum(np.log(x[1:]/x[:-1])/6*(even[:-1]*f(x[:-1])*x[:-1]+4*odd*f(mid)*mid+even[1:]*f(x[1:])*x[1:]))
    row=dict(reaction=REACTIONS[rid-1].name,current_rate=current,old_sigma_1GeV=old_full,old_sigma_original_cutoff=old_cut,old_cutoff_MeV=upper*1000,old_over_new=old_cut/current,old_logf_interpolation=old_cont,old_simpson100=simpson,interpolation_fraction=old_cont/old_cut-1,simpson_fraction=simpson/old_cont-1,cutoff_fraction=old_cut/old_full-1)
    rows.append(row)
    for mev in [6.5,7.5,8.5,9.5,10.5,11.5,15,20,30,50]:
        csrows.append(dict(reaction=row['reaction'],energy_MeV=mev,old_mb=old(mev*.001)/MB_TO_GEV_MINUS2,new_mb=new(mev*.001)/MB_TO_GEV_MINUS2))
for name,data in [('rates',rows),('cross_sections',csrows)]:
    with (out/f'{name}.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=data[0]);w.writeheader();w.writerows(data)
(out/'metadata.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),spectrum=str(base.relative_to(ROOT)),normalization='Same current Q=1 GeV^5 spectrum for all columns; not an old cascade rerun',note='Python transcription of old formulas, tables and quadrature; Fortran single precision literal rounding not emulated.'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
