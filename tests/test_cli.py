import pytest

from bbn_cascade.cli import _energy


def test_energy_units_use_longest_suffix():
    assert _energy("100GeV") == 100.0
    assert _energy("1TeV") == 1000.0
    assert _energy("100eV") == pytest.approx(1e-7)
