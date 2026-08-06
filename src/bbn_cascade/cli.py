import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import CascadeConfig
from .constants import EV, MEV
from .hybrid_solver import CascadeSolver
from .published import table_parameters
from .sources import MonochromaticPhotonSource


def _energy(text: str) -> float:
    # Longest suffix first: every GeV/MeV string also ends in "eV".
    units = {"tev": 1e3, "gev": 1.0, "mev": MEV, "kev": 1e-6, "ev": EV}
    value = text.strip().lower()
    for unit, factor in units.items():
        if value.endswith(unit):
            return float(value[:-len(unit)]) * factor
    return float(value)


def solve_command(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    e0, temp = _energy(args.energy), _energy(args.temperature)
    cfg = CascadeConfig(temp, e0, grid_size=args.grid_size, mode=args.mode)
    result = CascadeSolver(cfg).solve(MonochromaticPhotonSource(e0))
    result.to_csv(out / "spectrum.csv")
    fig, ax = plt.subplots()
    positive = result.photon > 0
    ax.loglog(result.energy[positive], result.photon[positive])
    ax.set(xlabel=r"$E_\gamma\,[\mathrm{GeV}]$", ylabel=r"$f_\gamma\,[\mathrm{GeV}^2]$")
    ax.grid(True, which="both", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "spectrum.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'spectrum.csv'} and {out / 'spectrum.png'}")


def reproduce_command(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for figure, e0 in enumerate((1e4, 1e3, 1e2), start=1):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if args.grid_size is None:
            decades = np.log10(e0 / MEV)
            grid_size = int(np.ceil(args.points_per_decade * decades)) + 1
        else:
            grid_size = args.grid_size
        for temp_ev, color in ((1.0, "tab:blue"), (100.0, "tab:orange")):
            cfg = CascadeConfig(
                temp_ev * EV, e0, grid_size=grid_size,
                quadrature_order=args.quadrature_order,
                soft_fraction=args.soft_fraction,
            )
            result = CascadeSolver(cfg).solve()
            stem = f"E0_{e0:g}GeV_T_{temp_ev:g}eV".replace("+", "")
            result.to_csv(out / f"{stem}.csv")
            mask = result.photon > 0
            ax.plot(result.energy[mask], np.log10(result.photon[mask]), color=color,
                    label=rf"$T={temp_ev:g}\,\mathrm{{eV}}$ (hybrid FP)")
            fit_cfg = CascadeConfig(temp_ev * EV, e0, grid_size=grid_size,
                                    mode="published-fit")
            fit = CascadeSolver(fit_cfg).solve()
            fit_mask = fit.photon > 0
            ax.plot(fit.energy[fit_mask], np.log10(fit.photon[fit_mask]), "--",
                    color=color, alpha=0.8,
                    label=rf"$T={temp_ev:g}\,\mathrm{{eV}}$ (Table 1 fit)")
            pl, nl, pp, npp = table_parameters(temp_ev * EV, e0)
            rows.append((figure, e0, temp_ev, grid_size, args.quadrature_order,
                         args.soft_fraction, pl, nl, pp, npp))
        ax.set_xscale("log")
        ax.set_ylim(10, 50)
        ax.set(xlabel=r"$E_\gamma\,[\mathrm{GeV}]$", ylabel=r"$\log_{10}(f_\gamma/\mathrm{GeV}^2)$", title=f"$E_0={e0:g}$ GeV")
        ax.legend()
        ax.grid(True, which="both", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / f"figure_{figure}.png", dpi=180)
        plt.close(fig)
    with (out / "table1_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "E0_GeV", "T_eV", "grid_size",
                         "quadrature_order", "soft_fraction", "P_low",
                         "N_low_GeV2", "P_pp", "N_pp_GeV2"])
        writer.writerows(rows)
    print(f"wrote Figures 1-3 and {out / 'table1_comparison.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bbn-cascade")
    sub = parser.add_subparsers(required=True)
    solve = sub.add_parser("solve", help="calculate one stationary spectrum")
    solve.add_argument("--energy", default="1TeV")
    solve.add_argument("--temperature", default="1eV")
    solve.add_argument("--mode", choices=("transport", "published-fit"), default="transport")
    solve.add_argument("--grid-size", type=int, default=400)
    solve.add_argument("--output", default="results/solve")
    solve.set_defaults(func=solve_command)
    reproduce = sub.add_parser("reproduce-9412055", help="reproduce paper Figs. 1-3")
    reproduce.add_argument("--grid-size", type=int, default=None,
                           help="fixed grid size (overrides --points-per-decade)")
    reproduce.add_argument("--points-per-decade", type=float, default=60.0)
    reproduce.add_argument("--quadrature-order", type=int, default=8)
    reproduce.add_argument("--soft-fraction", type=float, default=0.05)
    reproduce.add_argument("--output", default="results/9412055")
    reproduce.set_defaults(func=reproduce_command)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

