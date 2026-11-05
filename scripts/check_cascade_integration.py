"""Check the previously worst case against finer grid and panel quadrature.

Run AFTER reproduce-9412055, then regenerate the nuclear rate tables.
This script writes validation spectra and comparisons only.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from bbn_cascade import CascadeConfig, CascadeSolver


def main():
    out = Path('results/cascade_integration_checks')
    out.mkdir(parents=True, exist_ok=True)
    for label, n, order in [('grid120', 841, 8), ('quadrature12', 421, 12)]:
        start = time.monotonic()
        config = CascadeConfig(1e-7, 1e4, grid_size=n, quadrature_order=order)
        spectrum = CascadeSolver(config).solve()
        spectrum.to_csv(out/f'{label}.csv')
        np.save(out/f'{label}_boundaries.npy', spectrum.boundaries)
        (out/f'{label}.json').write_text(json.dumps(dict(config=asdict(config), diagnostics=spectrum.diagnostics), indent=2)+'\n')
        print(f'{label}: {time.monotonic()-start:.1f}s',flush=True)


if __name__ == '__main__':
    main()
