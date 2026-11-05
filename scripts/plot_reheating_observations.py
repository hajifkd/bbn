"""Overlay observational abundance intervals on an existing reheating scan.

No BBN rerun. Absolute predictions come from each point's result.json, never
from an assumed literature standard-BBN normalization. Li is not used.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import RegularGridInterpolator

SOURCES = {
    'D': dict(mean=2.508e-5, sigma=0.029e-5,
              source='PDG 2026, BBN review, Eq. 24.2 (revised August 2025)',
              url='https://pdg.lbl.gov/2026/reviews/rpp2026-rev-bbang-nucleosynthesis.pdf'),
    'He4': dict(mean=0.2458, sigma=0.0013,
                source='Aver et al. 2026, LBT Yp Project IV',
                url='https://arxiv.org/abs/2601.22238'),
    'He3': dict(ratio_upper=0.83+0.27,
                source='User-specified He3/D upper bound: 0.83 + 0.27',
                assumption='Present-day He3/D bounds the primordial ratio from above'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or args.scan / 'observations_PDG_D_He3_D'
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.scan / 'scan.npz')
    t, y, ratios = data['lifetime_s'], data['yield_X'], data['ratios']
    absolute = np.full_like(ratios, np.nan)
    standards = np.full_like(ratios, np.nan)
    seen = set()
    for path in sorted((args.scan / 'points').glob('*/result.json')):
        report = json.loads(path.read_text())
        model = report['model']
        i = np.argmin(abs(np.log(t / model['lifetime_s'])))
        j = np.argmin(abs(np.log(y / model['yield_X'])))
        if not (np.isclose(t[i], model['lifetime_s'], rtol=1e-10, atol=0)
                and np.isclose(y[j], model['yield_X'], rtol=1e-10, atol=0)):
            continue
        if (j, i) in seen:
            raise ValueError(f'Duplicate scan point: {path}')
        seen.add((j, i))
        absolute[j, i] = report['observables']
        standards[j, i] = report['standard_observables']
        mass = model['mass_GeV']
    if not np.all(np.isfinite(absolute)) or np.any(absolute <= 0):
        raise ValueError('Missing, nonfinite or nonpositive predictions')
    np.testing.assert_allclose(absolute / standards, ratios, rtol=1e-10)

    # Refine only the drawing mesh, not the physical calculation. Interpolate
    # log abundances on log parameter axes to preserve positivity.
    lt, ly = np.log10(t), np.log10(y)
    tx, yx = np.linspace(lt[0], lt[-1], 385), np.linspace(ly[0], ly[-1], 385)
    X, Y = np.meshgrid(tx, yx)
    positions = np.stack([Y, X], axis=-1)
    def interpolate(a, logarithmic=True):
        v = np.log(a) if logarithmic else a
        r = RegularGridInterpolator((ly, lt), v)(positions)
        return np.exp(r) if logarithmic else r
    pred = interpolate(absolute)
    reference = interpolate(standards)
    ratio = pred / reference
    caution = interpolate(data['blackbody_caution'].astype(float), False)
    d, he = SOURCES['D'], SOURCES['He4']
    upper3 = SOURCES['He3']['ratio_upper']
    he3_d = pred[..., 1] / pred[..., 0]
    native_he3_d = absolute[..., 1] / absolute[..., 0]
    pull = np.maximum(abs((pred[..., 0]-d['mean'])/d['sigma']),
                      abs((pred[..., 2]-he['mean'])/he['sigma']))
    native_pull = np.maximum(abs((absolute[..., 0]-d['mean'])/d['sigma']),
                             abs((absolute[..., 2]-he['mean'])/he['sigma']))
    # The scalar boundary score is not a chi-square or statistical pull.
    score = np.maximum(pull/2, he3_d/upper3)
    native_pass = (native_pull <= 2) & (native_he3_d < upper3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.88, top=.86, bottom=.22, wspace=.20, hspace=.24)
    green = '#006b45'
    def boundary(ax, z, level, color, style='-', width=2):
        if np.min(z) < level < np.max(z):
            c = ax.contour(X, Y, z, levels=[level], colors=[color], linewidths=width, linestyles=style)
            c.set_path_effects([pe.Stroke(linewidth=width+1.7, foreground='white'), pe.Normal()])
    def shade_band(ax, z, lo, hi):
        # Shade the continuous abundance band rather than interpolated booleans.
        ax.contourf(X, Y, z, levels=[lo, hi], colors=[green], alpha=.20)
    panels = [(axes[0,0], 0, r'D/H', d),
              (axes[0,1], 1, r'($^3$He + T)/H', None),
              (axes[1,0], 2, r'$^4$He mass fraction', he)]
    for ax, k, title, obs in panels:
        # All artists use log10 coordinates, including the native color cells.
        mesh = ax.pcolormesh(lt, ly, np.log10(ratios[..., k]), shading='auto',
                            cmap='coolwarm', norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1), rasterized=True)
        levels = [v for v in [.1,.5,.9,1.1,2,10] if ratio[..., k].min()<v<ratio[..., k].max()]
        if levels:
            c = ax.contour(X,Y,ratio[..., k],levels=levels,colors='k',linewidths=.5,alpha=.5)
            ax.clabel(c,fmt='%g',fontsize=7)
        if obs:
            lo, hi = obs['mean']-2*obs['sigma'], obs['mean']+2*obs['sigma']
            shade_band(ax,pred[...,k],lo,hi)
            boundary(ax,pred[...,k],lo,green,'--')
            boundary(ax,pred[...,k],hi,green)
            note = (rf'${lo*1e5:.3f} < 10^5\,{{\rm D/H}} < {hi*1e5:.3f}$' if k==0
                    else r'$0.2432 < Y_p < 0.2484$')
        else:
            shade_band(ax,he3_d,0,upper3)
            boundary(ax,he3_d,upper3,'#7b3294',':',2.5)
            note = r'Upper bound: ($^3$He + T)/D $< 0.83+0.27=1.10$'
        ax.set_title(title+'\n'+note,fontsize=11)

    ax=axes[1,1]
    ax.contourf(X,Y,score,levels=[0,1,max(2,float(score.max())+1)],
                colors=['#b9e4c9','#e9e9e9'])
    for k,obs,color in [(0,d,'#0072b2'),(2,he,'#d55e00')]:
        for sign,style in [(-1,'--'),(1,'-')]:
            boundary(ax,pred[...,k],obs['mean']+sign*2*obs['sigma'],color,style,1.7)
    boundary(ax,he3_d,upper3,'#7b3294',':',2)
    ax.set_title('D + $^4$He + $^3$He/D: intersection of constraints\nGreen: all satisfied; gray: at least one fails',fontsize=11)
    ax.legend(handles=[Line2D([],[],color='#0072b2',label='D/H boundary'),
                       Line2D([],[],color='#d55e00',label='$^4$He boundary'),
                       Line2D([],[],color='#7b3294',ls=':',label='$^3$He/D upper bound')],loc='lower left',fontsize=9)
    for ax in axes.flat:
        if caution.max()>.5:
            hatch=ax.contourf(X,Y,caution,levels=[.5,1.1],colors='none',hatches=['///'])
            hatch.set_edgecolor((.25,.25,.25,.45));hatch.set_linewidth(0)
        ax.set_xlim(lt[0],lt[-1]);ax.set_ylim(ly[0],ly[-1])
        ax.set_xticks(np.arange(2,9),[rf'$10^{{{v}}}$' for v in range(2,9)])
        ax.set_yticks(np.arange(-18,-5,2),[rf'$10^{{{v}}}$' for v in range(-18,-5,2)])
        ax.set_xlabel(r'Lifetime $\tau_X$ [s]');ax.set_ylabel(r'Initial $Y_X=n_X/s$')
    cax=fig.add_axes([.90,.34,.018,.43])
    fig.colorbar(mesh,cax=cax,label=r'$\log_{10}$(abundance / standard BBN)',extend='both')
    fig.suptitle(rf'$X\to\gamma\gamma$, $m_X={mass:g}$ GeV: entropy production included'+'\nObservational overlays; final baryon/photon ratio fixed to CMB',fontsize=16,y=.97)
    fig.legend(handles=[Line2D([],[],color=green,lw=2,ls='--',label='Lower observational $2\\sigma$ boundary'),
                        Line2D([],[],color=green,lw=2,label='Upper observational $2\\sigma$ boundary'),
                        Line2D([],[],color='#7b3294',lw=2,ls=':',label='$^3$He/D upper bound: 1.10')],
               loc='lower center',bbox_to_anchor=(.5,.115),ncol=3,fontsize=9)
    fig.text(.085,.035,
             f'D/H = ({d["mean"]*1e5:.3f} +/- {d["sigma"]*1e5:.3f})e-5 (PDG 2026), +/- 2 sigma; He3/D < 1.10. He4: LBT 2026, +/- 2 sigma.\n'
             f'Observational errors only; nuclear/theory errors and CMB-eta uncertainty not propagated. No Li constraint.\n'
             f'{len(y)} x {len(t)} computed points; boundaries interpolated in log abundance and log parameters, not refined BBN runs.\n'
             'Hatching: blackbody thermalization not assured. The intersection is not a joint 95% confidence region.',fontsize=9)
    for ext in ['png','pdf']:
        fig.savefig(out/f'abundance_observational_constraints.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    metadata=dict(sources=SOURCES,sigma_multiplier=2,he3_D_upper_bound=upper3,
                  combined_uses=['D','He4','He3/D'],theory_error_included=False,
                  eta_uncertainty_included=False,li_used=False,
                  interpretation='D and He4 +/-2 sigma intervals intersected with He3/D < 1.10; not joint 95% CL',
                  native_shape=list(absolute.shape[:2]),points_passing_D_He4=int(np.sum(native_pull<=2)),
                  points_passing_all=int(np.sum(native_pass)),
                  standard_observables_min=standards.min(axis=(0,1)).tolist(),
                  standard_observables_max=standards.max(axis=(0,1)).tolist(),
                  scan_source=str(args.scan.resolve()))
    (out/'observational_inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savetxt(out/'observational_comparison.csv',
               [[tau,yield_x,*absolute[j,i,:3],native_he3_d[j,i],int(native_pull[j,i]<=2),int(native_he3_d[j,i]<upper3),int(native_pass[j,i])]
                for j,yield_x in enumerate(y) for i,tau in enumerate(t)],delimiter=',',
               header='tau_s,YX,D_H,He3_plus_T_H,He4_mass_fraction,He3_plus_T_over_D,pass_D_He4_2sigma,pass_He3_D_upper,pass_all')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    main()
