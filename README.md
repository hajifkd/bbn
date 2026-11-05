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

## Photonuclear reaction rates

The current public solver is the hybrid Boltzmann/Fokker--Planck implementation
in `hybrid_solver.py` (the earlier characteristic-energy description above is
historical). Photonuclear rates can be evaluated with:

```console
uv run bbn-cascade rates --energy 1TeV --temperature 100eV
```

Fits are independently transcribed from astro-ph/0211258, with the He4 update
from 1808.09324 and the Li7/Be7 corrections from 1403.5995. The convolution
stops at 1 GeV while the cascade retains the full injection energy.
See [PHOTODISINTEGRATION.md](PHOTODISINTEGRATION.md) for units, provenance,
extrapolation assumptions, benchmark tables and convergence checks.

The default energy grid is **120 points per decade** in the Python API and all
CLI calculations. The grid size is computed from the energy bounds as
`max(40, ceil(120 * log10(Emax/Emin)) + 1)`, including both endpoints.
Use `points_per_decade` / `--points-per-decade` to adjust the density, or an
explicit `grid_size` / `--grid-size` to override the automatic grid size.

### Temperature-dependent photonuclear tables

For 100 GeV photon injection at 61 temperatures from 1 MeV to 1 eV
(10 intervals/decade), eta = 6.100e-10, and 120 energy points/decade:

```sh
uv run python scripts/temperature_rate_table.py --workers 12
```

See [TEMPERATURE_RATE_TABLE.md](TEMPERATURE_RATE_TABLE.md) for table formats,
normalization, checkpointing, and the existing cascade model's high-temperature
limitations (thermal electron/positron pairs are not included).

## Decaying particles and BBN

`uv run bbn-decay all --workers 12` computes fixed-standard-background responses
for X → γγ, evolves the AlterAlterBBN network, and writes lifetime–yield abundance
ratio contours. Defaults are mX=200 GeV and tau=10²–10⁸ s. See
[DECAY_BBN.md](DECAY_BBN.md) for normalization, individual stages, numerical
checks, and the explicit low-temperature and nuclear-network approximations.

`uv run bbn-reheating scan` includes decay heating, modified expansion, and
initial baryon-density adjustment to the final CMB value. See
[REHEATING_BBN.md](REHEATING_BBN.md) for the operator cache, completed scan,
trace-species time-step tolerance, and thermalization assumptions.
