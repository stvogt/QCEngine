"""
Tests for the MACE harness hessian driver: full and partial (active-atom) Hessians by central
differences of the forces and by autograd, on a water dimer with the MACE-OFF23 small model.
"""

import numpy as np
import pytest

import qcengine as qcng
from qcengine.testing import uusing

WATER_DIMER = """
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
"""


def _run(driver, **keywords):
    from qcelemental.models.v2 import AtomicInput, Molecule

    mol = Molecule.from_data(WATER_DIMER, fix_com=True, fix_orientation=True)
    inp = AtomicInput(
        molecule=mol, specification={"driver": driver, "model": {"method": "small"}, "keywords": keywords}
    )
    ret = qcng.compute(inp, "mace", raise_error=True, return_version=2)
    assert ret.success
    return ret


@uusing("mace")
def test_mace_partial_hessian_is_the_block_of_the_full_hessian():
    active = [3, 4, 5]  # the second water; the first one stays as its fixed environment
    full = np.asarray(_run("hessian").return_result)
    part = _run("hessian", active_atoms=active)
    block = np.asarray(part.return_result)
    assert full.shape == (18, 18) and block.shape == (9, 9)
    sel = [3 * i + k for i in active for k in range(3)]
    assert np.allclose(block, full[np.ix_(sel, sel)], atol=1.0e-10)
    assert np.allclose(full, full.T)
    info = part.extras["mace_hessian"]
    assert info["active_atoms"] == active and info["partial"] and info["method"] == "fd"
    assert info["fd_step_angstrom"] == pytest.approx(0.005) and info["n_atoms"] == 6


@uusing("mace")
def test_mace_hessian_matches_differences_of_the_gradient():
    # row of atom 3, x: central difference of the harness gradient itself
    from qcelemental.models.v2 import AtomicInput, Molecule

    step = 0.005 / 0.529177210903
    mol = Molecule.from_data(WATER_DIMER, fix_com=True, fix_orientation=True)
    grads = []
    for sign in (+1, -1):
        geom = np.array(mol.geometry, dtype=float)
        geom[3, 0] += sign * step
        disp = Molecule(symbols=mol.symbols, geometry=geom.ravel(), fix_com=True, fix_orientation=True)
        inp = AtomicInput(molecule=disp, specification={"driver": "gradient", "model": {"method": "small"}})
        grads.append(np.asarray(qcng.compute(inp, "mace", raise_error=True, return_version=2).return_result).ravel())
    row = (grads[0] - grads[1]) / (2 * step)
    hess = np.asarray(_run("hessian").return_result)
    # the harness symmetrizes, so compare within the asymmetry of the finite differences
    assert np.allclose(hess[9], row, atol=5.0e-5)


@uusing("mace")
def test_mace_analytic_hessian_agrees_with_finite_differences():
    active = [0, 1, 2, 3]
    fd = np.asarray(_run("hessian", active_atoms=active).return_result)
    an = _run("hessian", active_atoms=active, hessian_method="analytic")
    assert an.extras["mace_hessian"]["method"] == "analytic" and an.extras["mace_hessian"]["fd_step_angstrom"] is None
    assert np.allclose(np.asarray(an.return_result), fd, atol=5.0e-4)


@uusing("mace")
@pytest.mark.parametrize(
    "keywords, match",
    [
        ({"active_atoms": [0, 0]}, "repeated"),
        ({"active_atoms": [6]}, "must lie in"),
        ({"active_atoms": []}, "empty"),
        ({"hessian_method": "numerical"}, "'fd' or 'analytic'"),
        ({"fd_step": 0.0}, "positive"),
    ],
)
def test_mace_hessian_keywords_are_validated(keywords, match):
    with pytest.raises(qcng.exceptions.InputError, match=match):
        _run("hessian", **keywords)
