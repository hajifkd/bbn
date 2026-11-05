"""Plot supplied Moroi rates against the new temperature response table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IDS = [1, 2, 3, 4, 5, 6, 7, 9]
LABELS = [r'D($\gamma,p$)n', r't($\gamma,n$)D', r't($\gamma,p$)2n',
          r'$^3$He($\gamma,p$)D', r'$^3$He($\gamma,n$)2p',
          r'$^4$He($\gamma,p$)t', r'$^4$He($\gamma,n$)$^3$He',
          r'$^4$He($\gamma,pn$)D']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=ROOT/'rate_moroi.dat')
    parser.add_argument('--new', type=Path, default=ROOT/'results/temperature_rates_E0_100GeV_eta6p1e-10/rates.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'results/moroi_temperature_comparison')
    args = parser.parse_args()
    old = np.loadtxt(args.old)
    assert old.ndim == 2 and old.shape[1] == 17
    assert np.allclose(old[:, 0], 6.1e-10, rtol=1e-5, atol=0)
    assert np.allclose(old[:, 1], 100., rtol=1e-8, atol=0)
    old = old[np.argsort(old[:, 2])]
    t_old = old[:, 2]*1e9
    assert np.all(np.diff(t_old)>0) and np.all(old[:, 3:11]>0)
    # moroi_old.f sets ndotin=1 (number injection); Q=E0*ndotin.
    k_old = old[:, 3:11]/old[:, 1, None]
    with np.load(args.new) as new:
        order = np.argsort(new['temperature_eV'])
        t_new = new['temperature_eV'][order]
        columns = [list(new['reaction_id']).index(i) for i in IDS]
        k_new = new['response_GeV_minus4'][order][:, columns]
    assert k_new.shape == (61, 8) and np.all(k_new>0)
    assert t_new[0] >= t_old[0] and t_new[-1] <= t_old[-1]
    # The old output rounds temperatures to four significant digits.
    # Interpolate in log T and log K; never extrapolate.
    matched = np.column_stack([np.exp(np.interp(np.log(t_new), np.log(t_old), np.log(k_old[:, i]))) for i in range(8)])
    ratios = matched/k_new
    anchor_index = int(np.argmin(abs(t_new-100.)))
    assert np.isclose(t_new[anchor_index], 100.)
    common_anchor = k_new[anchor_index, 0]/matched[anchor_index, 0]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for suffix, factor, title in [('', 1., 'Same energy-injection normalization: old / 100'),
                                  ('_D100eV_matched', common_anchor, 'One additional common factor: D matched at 100 eV')]:
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.loglog(t_new, k_new[:, i], color='C0', lw=1.8, label='New calculation')
            ax.loglog(t_old, factor*k_old[:, i], '--', color='C1', lw=1.6,
                      label='Moroi / 100'+(' (D anchor)' if suffix else ''))
            ax.set_title(LABELS[i])
            ax.set_ylabel(r'$K_r$ [GeV$^{-4}$]'+(' (rescaled old)' if suffix else ''))
            ax.set_xlim(.1, 1e6)
            ax.grid(True, which='both', alpha=.2)
            ax.legend(fontsize=8)
        for ax in axes[-1]: ax.set_xlabel('Bath temperature [eV]')
        fig.suptitle('100 GeV photon injection; '+title+'\nNew model at high T omits thermal electron/positron pairs', fontsize=11)
        fig.tight_layout(rect=(0,0,1,.95))
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'responses_vs_temperature{suffix}.{extension}', dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.loglog(t_new, ratios[:, i], color='C1', label='(Moroi / 100) / new')
        ax.loglog(t_new, ratios[:, i]*common_anchor, '--', color='C2', label='With common D(100 eV) match')
        ax.axhline(1., color='0.3', lw=.8)
        ax.set_title(LABELS[i]); ax.set_ylabel('Ratio'); ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Bath temperature [eV]')
    fig.suptitle('Common temperature range only; no reaction-dependent normalization', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    for extension in ('png','pdf'): fig.savefig(out/f'ratios_vs_temperature.{extension}', dpi=170)
    plt.close(fig)
    with (out/'comparison.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['temperature_eV','reaction_id','new_K_GeV_minus4','moroi_K_GeV_minus4_log_interpolated','moroi_over_new','moroi_K_D100eV_matched','matched_moroi_over_new'])
        for j,t in enumerate(t_new):
            for i,rid in enumerate(IDS):
                writer.writerow([t,rid,k_new[j,i],matched[j,i],ratios[j,i],matched[j,i]*common_anchor,ratios[j,i]*common_anchor])
    with (out/'moroi_eight_reactions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['temperature_eV']+[f'r{i:02d}_K_GeV_minus4' for i in IDS])
        writer.writerows([t,*row] for t,row in zip(t_old,k_old))
    metadata=dict(old_file=str(args.old),new_file=str(args.new),
                  old_sha256=hashlib.sha256(args.old.read_bytes()).hexdigest(),
                  new_sha256=hashlib.sha256(args.new.read_bytes()).hexdigest(),
                  old_rows=len(old),new_rows=len(t_new),reaction_ids=IDS,
                  old_columns_1based=list(range(4,12)),temperature_unit_in_old_file='GeV',
                  old_number_injection_rate_GeV4=1.,energy_GeV=100.,old_to_K_factor=.01,
                  normalization_source='moroi_old.f: ndotin=1.0d0; epho0=1.0d2; spectrum source proportional to ndotin',
                  optional_D100eV_common_factor=float(common_anchor),
                  interpolation='log K versus log T, common range only; handles four-significant-digit old temperatures',
                  old_over_new_at_100eV=ratios[anchor_index].tolist(),
                  limitation='New high-temperature model omits thermal pairs. No per-reaction or temperature-dependent normalization fitted.',
                  extra_old_columns_1based={'12':'rateli7to432 = rateli7tot - rateli7to6','13':'rateli7to6','14':'constant placeholder 1e-30','15':'rateli6tot','16':'rli6t (nonthermal Li6 from T)','17':'rli6h (nonthermal Li6 from He3)'})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),old_to_K_factor=.01,optional_common_factor=common_anchor,
                          old_over_new_at_100eV=ratios[anchor_index].tolist()),indent=2))


if __name__=='__main__': main()
