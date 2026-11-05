"""Checkpointed temperature sweep for the existing stationary cascade model."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from bbn_cascade.config import CascadeConfig
from bbn_cascade.hybrid_solver import CascadeSolver
from bbn_cascade.sources import MonochromaticPhotonSource
from bbn_cascade.photodisintegration import calculate_rates, write_rates
from bbn_cascade.photonuclear import REACTIONS

MODEL_NOTE = ('Existing stationary downward cascade; background electrons are baryonic only, '
              'Compton targets are at rest, and thermal electron/positron pairs are omitted. '
              'High-temperature entries, especially near 1 MeV, are formal model outputs, '
              'not validated physical BBN rates. Helium mass fraction is fixed to zero. '
              'This table supplies per-target electromagnetic photodisintegration responses, '
              'not a complete BBN abundance network.')


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve().parents[1]/'src/bbn_cascade').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def configuration(temperature):
    return CascadeConfig(temperature=float(temperature), maximum_energy=100.,
                         points_per_decade=120., quadrature_order=8,
                         eta_baryon=6.100e-10, helium_mass_fraction=0.)


def run_temperature(task):
    index, temperature, root, fingerprint = task
    out = Path(root)/f'T_{index:02d}'
    cfg = configuration(temperature)
    identity = dict(config=asdict(cfg), source_sha256=fingerprint,
                    injection_rate_GeV5=1., maximum_photon_energy_GeV=1.)
    marker = out/'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'{out}: incompatible checkpoint; choose a new output directory')
        for name in ('spectrum.csv', 'boundaries_GeV.npy', 'metadata.json', 'rates.csv'):
            if not (out/name).is_file():
                raise ValueError(f'{out}: incomplete checkpoint: {name}')
        return index, saved['seconds'], True
    start = time.monotonic()
    spectrum = CascadeSolver(cfg).solve(MonochromaticPhotonSource(100., 1.))
    if not np.all(np.isfinite(spectrum.electron)) or np.any(spectrum.electron < 0):
        raise ValueError(f'{out}: invalid electron spectrum')
    rows = calculate_rates(spectrum, maximum_energy=1.)
    for row in rows:
        values = (row.rate_GeV, row.rate_per_s, row.response_GeV_minus4)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f'{out}: invalid rate')
    write_rates(out, rows, dict(identity, temperature_eV=temperature*1e9,
                               diagnostics=spectrum.diagnostics, model_limitations=MODEL_NOTE))
    spectrum.to_csv(str(out/'spectrum.csv'))
    np.save(out/'boundaries_GeV.npy', spectrum.boundaries)
    elapsed = time.monotonic()-start
    temporary = out/'complete.json.tmp'
    temporary.write_text(json.dumps(dict(identity=identity, seconds=elapsed), indent=2, allow_nan=False)+'\n')
    temporary.replace(marker)
    return index, elapsed, False


def aggregate(root, temperatures, fingerprint):
    long_rows = []
    arrays = {key: [] for key in ('rate_GeV', 'rate_per_s', 'response_GeV_minus4')}
    ids = [r.id for r in REACTIONS]
    for i, temperature in enumerate(temperatures):
        out = root/f'T_{i:02d}'
        saved = json.loads((out/'complete.json').read_text())
        assert saved['identity']['source_sha256'] == fingerprint
        with (out/'rates.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row['reaction_id']) for row in rows] == ids
        for key in arrays:
            arrays[key].append([float(row[key]) for row in rows])
        for row in rows:
            long_rows.append(dict(temperature_GeV=float(temperature), temperature_eV=float(temperature*1e9), **row))
    with (root/'rates_long.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    for key in arrays:
        arrays[key] = np.asarray(arrays[key])
        assert arrays[key].shape == (61, 17)
        assert np.all(np.isfinite(arrays[key])) and np.all(arrays[key] >= 0)
        with (root/f'{key}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['temperature_GeV', 'temperature_eV']+[f'r{id_:02d}' for id_ in ids])
            writer.writerows([float(t), float(t*1e9), *values] for t, values in zip(temperatures, arrays[key]))
    np.savez(root/'rates.npz', temperature_GeV=temperatures, temperature_eV=temperatures*1e9,
             reaction_id=ids, reaction_name=[r.name for r in REACTIONS], **arrays)
    metadata = dict(source_sha256=fingerprint, temperature_count=61, intervals_per_decade=10,
                    temperature_order='descending', reaction_count=17,
                    config_at_1MeV=asdict(configuration(temperatures[0])),
                    injection_rate_GeV5=1., maximum_photon_energy_GeV=1.,
                    normalization='Per target: rate=int sigma(E) dn_gamma/dE dE; response=rate/Q. No target abundance factor.',
                    model_limitations=MODEL_NOTE, reactions=[asdict(r) for r in REACTIONS],
                    interpolation='Not prescribed. Resolve normalization and applicability before BBN coupling.')
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    for i, reaction in enumerate(REACTIONS):
        axes[i//6].loglog(temperatures*1e9, arrays['response_GeV_minus4'][:, i], label=f'{reaction.id}: {reaction.name}')
    for ax in axes:
        ax.set_ylabel(r'$K_r$ [GeV$^{-4}$]')
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Bath temperature [eV]')
    fig.suptitle('100 GeV photon injection; existing stationary cascade\nHigh-temperature outputs require thermal-pair model extension', fontsize=11)
    fig.tight_layout()
    fig.savefig(root/'responses_vs_temperature.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/temperature_rates_E0_100GeV_eta6p1e-10'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    temperatures = np.logspace(-3, -9, 61)
    assert np.allclose(np.diff(np.log10(temperatures)), -.1)
    fingerprint = source_fingerprint()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(i, float(t), str(args.output), fingerprint) for i, t in enumerate(temperatures)]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_temperature, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                index, elapsed, reused = future.result()
                print(f'{completed}/61 T={temperatures[index]*1e9:.8g} eV: OK ({elapsed:.1f}s, cached={reused})', flush=True)
            except Exception as exc:
                failures.append(dict(index=task[0], temperature_GeV=task[1], error=repr(exc)))
                print(f'{completed}/61 T={task[1]*1e9:.8g} eV: FAILED {exc!r}', flush=True)
    (args.output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    if failures:
        raise RuntimeError(f'{len(failures)} temperatures failed; see failures.json. Completed temperatures are checkpointed.')
    aggregate(args.output, temperatures, fingerprint)
    print(f'Validated 61 x 17 rates; wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
