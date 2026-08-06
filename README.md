# bbn-cascade

Early-Universe electromagnetic cascade spectra targeting Kawasaki & Moroi
(astro-ph/9412055). Energies are in GeV and natural units are used. The
default injected energy rate is `1 GeV^5`, hence `f_gamma` is in `GeV^2`.

## Setup

```console
uv sync --dev
uv run pytest
uv run bbn-cascade reproduce-9412055
uv run bbn-cascade solve --energy 1TeV --temperature 1eV --mode transport
```

The reproduction command writes Figures 1--3 and a Table 1 CSV below
`results/9412055`. A single run writes photon/electron spectra and photon
loss rates by process.

## Physics modes

- `published-fit` evaluates Eqs. (15)--(16) using Table 1. It is the exact
  regression reference for the published figures.
- `transport` solves a stationary, downward-in-energy cascade including
  gamma-gamma pair production/scattering, nuclear pair production, Compton
  scattering and inverse-Compton cooling. Redistribution currently uses
  energy-conserving characteristic-energy kernels; precision BBN constraints
  will require replacing these with the full differential Appendix kernels.

Legacy defaults are `eta=3e-10`, complete ionization and a pure-proton target.
Hubble expansion is omitted under the paper's stationary approximation.

## Python API

```python
from bbn_cascade import CascadeConfig, CascadeSolver
from bbn_cascade.constants import EV, TEV

result = CascadeSolver(CascadeConfig(EV, TEV, mode="transport")).solve()
result.to_csv("spectrum.csv")
```

`CascadeSpectrum` exposes the energy grid, photon/electron distributions,
process rates and numerical diagnostics for later coupling to a BBN network.
