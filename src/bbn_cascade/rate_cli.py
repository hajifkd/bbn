"""CLI entry point for photon injection and photonuclear reaction rates."""
from dataclasses import asdict
import numpy as np

from .config import CascadeConfig, DEFAULT_POINTS_PER_DECADE
from .hybrid_solver import CascadeSolver
from .sources import MonochromaticPhotonSource
from .photodisintegration import calculate_rates, write_rates


def rates_command(args):
    from .cli import _energy
    cfg = CascadeConfig(_energy(args.temperature), _energy(args.energy),
                        grid_size=args.grid_size, points_per_decade=args.points_per_decade,
                        quadrature_order=args.quadrature_order,
                        eta_baryon=args.eta_baryon, helium_mass_fraction=args.helium_mass_fraction)
    if not np.isfinite(args.injection_rate) or args.injection_rate <= 0:
        raise ValueError("injection-rate must be positive in GeV^5")
    spectrum = CascadeSolver(cfg).solve(MonochromaticPhotonSource(cfg.maximum_energy, args.injection_rate))
    rows = calculate_rates(spectrum, maximum_energy=_energy(args.maximum_photon_energy))
    write_rates(args.output, rows, dict(config=asdict(cfg), injection_rate_GeV5=args.injection_rate,
                                      diagnostics=spectrum.diagnostics))
    spectrum.to_csv(str(args.output)+'/spectrum.csv')
    np.save(str(args.output)+'/boundaries_GeV.npy', spectrum.boundaries)
    print(f'wrote {args.output}/rates.csv (17 reactions)')


def add_rates_parser(sub):
    parser = sub.add_parser('rates', help='convolve photon injection spectrum with photonuclear cross sections')
    parser.add_argument('--energy', default='1TeV')
    parser.add_argument('--temperature', default='1eV')
    parser.add_argument('--maximum-photon-energy', default='1GeV')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='fixed grid size (overrides --points-per-decade)')
    parser.add_argument('--points-per-decade', type=float, default=DEFAULT_POINTS_PER_DECADE)
    parser.add_argument('--quadrature-order', type=int, default=8)
    parser.add_argument('--eta-baryon', type=float, default=3e-10)
    parser.add_argument('--helium-mass-fraction', type=float, default=0.)
    parser.add_argument('--injection-rate', type=float, default=1., help='energy injection rate in GeV^5')
    parser.add_argument('--output', default='results/photodisintegration/single')
    parser.set_defaults(func=rates_command)
