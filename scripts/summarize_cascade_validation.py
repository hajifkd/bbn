"""Convolve independently computed validation spectra and report differences."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from bbn_cascade import CascadeSpectrum, calculate_rates
from bbn_cascade.photodisintegration import write_rates


def main():
    root = Path('results/cascade_integration_checks')
    baseline = Path('results/photodisintegration/E0_10000GeV_T_100eV/rates.csv')
    with baseline.open() as handle:
        base = list(csv.DictReader(handle))
    comparisons = []
    for label in ['grid120', 'quadrature12']:
        path = root/f'{label}.csv'
        data = np.genfromtxt(path, delimiter=',', names=True)
        metadata = json.loads(path.with_suffix('.json').read_text())
        spectrum = CascadeSpectrum(data['energy_GeV'], data['photon_GeV2'], data['electron_GeV2'], {},
                                   metadata['diagnostics'], np.load(root/f'{label}_boundaries.npy'))
        rows = calculate_rates(spectrum)
        write_rates(root/label, rows, dict(metadata, source_spectrum=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for original, row in zip(base, rows):
            ref = float(original['rate_GeV'])
            comparisons.append(dict(check=label, reaction_id=row.reaction_id, reaction=row.reaction,
                                    baseline_rate_GeV=ref, check_rate_GeV=row.rate_GeV,
                                    relative_difference=row.rate_GeV/ref-1))
        print(label, 'max absolute relative rate change', max(abs(x['relative_difference']) for x in comparisons if x['check']==label))
    with (root/'rate_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


if __name__ == '__main__':
    main()
