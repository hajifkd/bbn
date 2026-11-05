# X → γγ and BBN

Run from this repository with a C99 compiler and `uv` available:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run bbn-decay all --workers 12
```

The default is mX=200 GeV, tau=1e2–1e8 seconds, and undecayed
YX=nX/s, where s is total standard-background entropy density (including
neutrinos). The radiation background is the cosmology supplied with the pinned
AlterAlterBBN revision in `vendor/alteralterbbn/UPSTREAM_COMMIT`. Its final eta
is 6.137e-10. The entropy normalization is calibrated from photon and neutrino
temperatures after electron annihilation and continued using conserved comoving
baryon density. No entropy injection or expansion change from X is included.

## Stages and outputs

`background`, `prepare`, `scan`, `refine`, `validate`, `validate-responses`, `validate-stationarity`, and `plot` can be run
individually. All accept `--output PATH`. `prepare` also accepts `--mass`,
`--points-per-decade`, `--temperature-intervals`, and `--workers`.
The scan initially uses 25 points on each axis (`--scan-points`). `all` performs
background/response preparation, scan, contour refinement, integration checks, representative response-grid comparisons, and stationary-response checks.
Do not use the shared-library backend concurrently in threads; use processes.

- `standard_history.dat`: time in seconds, then all 26 n_i/n_b abundances.
- `standard_final.npy`: 26 final abundances, with residual free neutrons decayed.
- `response_*.npz`: reusable, individually checkpointed fixed-background responses.
- `response_table.npz`: temperatures, times, and 17 K=Gamma/Q coefficients in GeV^-4.
- `time_response.npy`: time and 17 coefficients mX*s*K. C multiplies these by
  YX*exp(-t/tau)/tau to obtain per-target rates in s^-1. This includes both
  photons through Q=mX*nX/tau; no additional factor of two is applied.
- `scan.npz`, `scan.csv`: lifetimes, yields, and standard-BBN abundance ratios.
- `abundance_contours.png`, `abundance_contours.pdf`: four abundance-ratio panels.
- `metadata.json`: model assumptions, source identity, and upstream revision.
- `contour_convergence.json`, `integration_validation.json`, `response_validation.json`, `stationarity_validation.json`: numerical checks.

The displayed observables are D/H, (He3+T)/H, 4*nHe4/nb, and (Li7+Be7)/H.
Each lifetime uses a zero-source reference with the same adaptive-step settings, so ratios approach unity continuously as YX goes to zero.
Unstable T and Be7 are combined with their eventual daughters for reporting.
The upper allowed yield is set by max(rhoX/rho_background)=0.01, with background
energy density inferred from the supplied Hubble rate. Points outside this
criterion are masked, not extrapolated. No observational abundance cuts are
applied: these are abundance-ratio contours, not exclusion limits.

The abundance lower limit is reduced until all four ratios at the sampled
lifetimes lie within 0.1% of unity. Contours use levels 0.1, 0.5, 0.9, 0.99,
1.01, 1.1, 2, and 10 where present. Refinement doubles the axis intervals until
the sampled contour curves change by at most 0.05 dex; it records a failed convergence check if
385 points per axis are insufficient, while retaining the computed figures and data.

## Coupling and approximations

The standard BBN background is computed once. The cascade's effective H/He
parameters match the standard background's electron density sum(Z*n) and
nuclear-pair target density sum(Z^2*n), using the nine dominant light species.
Free neutrons do not contribute to either. This background is fixed across
all decay models. It is never updated from the perturbed abundances.

The photonuclear reaction matrix is added to the same implicit linear solve as
the thermal reactions. Actual evolving target abundances multiply the rates.
Responses with positive endpoints are interpolated logarithmically; intervals
containing a zero use linear values in log time. No response extrapolation is
allowed. Early thermal evolution uses the upstream integrator. Below its original
endpoint T9=0.01 (~0.862 keV), thermal nuclear reactions are frozen, weak decays
continue, and an implicit Euler step handles the stiff late network. This avoids
both unstable low-temperature extrapolation of thermal fits and late explicit
predictor instability. The zero-injection observables are checked against the
upstream endpoint and the extended solution.

Thermal electron/positron pairs are omitted from the *cascade*, as requested;
they remain in standard cosmology/BBN. High-temperature cascade responses are
therefore conditional on that approximation. Nonthermal nuclear secondaries
are omitted, and photonuclear convolution stops at 1 GeV. Reaction 11 of
astro-ph/0211258 is inclusive Li6(gamma,X)A=3: its appendix does not specify the
exclusive daughter multiplicities. It is omitted from both production and
destruction rather than assigning an arbitrary branch.

Integration validation includes both near-standard (~10% departure where available) and strong-injection examples, and tests the sensitivity to moving the thermal cutoff to T9=0.005.

The response checkpoints contain a photon-occupancy diagnostic `residence_s`,
normalized per injected photon. This is not a rigorous slowest relaxation time
for the coupled photon/electron cascade. The separate stationarity check solves A*N0=S and A*N1=N0 for the cascade operator, then computes the nuclear-rate-weighted mean delay from N1/N0. At three representative temperatures it compares this delay to the source variation scale (3H+1/tau)^-1 at tau=t. This is a mean-delay diagnostic, not a global bound on the slowest mode. Energy/temperature discretization is assessed separately; integration checks do not certify its accuracy. Any failed numerical check remains visible
in the corresponding JSON, and must not be described as a converged prediction.

Changing physics sources or response settings requires a new output directory.
The pinned third-party code is GPL-3.0; its license is retained in the vendor
folder. Local changes are the decay hook, late integration, history output, and
shared-library adapter. The library builds without `-ffast-math`.
