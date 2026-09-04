"""Tests for the ASE optimization procedure."""
import os

import numpy as np
import pytest

import qcengine as qcng
from qcengine.exceptions import InputError
from qcengine.procedures.ase import (
    DEFAULT_FMAX_EV_PER_ANGSTROM,
    DEFAULT_OPTIMIZER,
    _build_optimizer,
    _frozen_indices,
)

ase = pytest.importorskip("ase")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_ase_procedure_is_registered_and_available():
    assert "ase" in qcng.list_all_procedures()
    assert "ase" in qcng.list_available_procedures()
    assert qcng.get_procedure("ase").get_version()


# ---------------------------------------------------------------------------
# constraints: geomeTRIC-compatible dict, and never silently ignored
# ---------------------------------------------------------------------------

def test_frozen_indices_parses_geometric_style_dict():
    """Same structure geomeTRIC takes, so a workflow can switch optimizer unchanged."""
    assert _frozen_indices({"freeze": [{"type": "xyz", "indices": [2, 0, 1]}]}) == [0, 1, 2]
    assert _frozen_indices(None) == []
    assert _frozen_indices({}) == []


def test_frozen_indices_deduplicates_across_blocks():
    cons = {"freeze": [{"type": "xyz", "indices": [0, 1]}, {"type": "xyz", "indices": [1, 5]}]}
    assert _frozen_indices(cons) == [0, 1, 5]


@pytest.mark.parametrize(
    "constraints, match",
    [
        ("$freeze\nxyz 1-3\n", "must be a dict"),          # the classic text form
        ({"set": [{"type": "distance", "indices": [0, 1], "value": 1.0}]}, "only 'freeze'"),
        ({"freeze": [{"type": "distance", "indices": [0, 1]}]}, "only 'xyz'"),
    ],
)
def test_frozen_indices_refuses_rather_than_ignores(constraints, match):
    """Dropping an unsupported constraint would relax atoms the caller believes are
    fixed and quietly produce wrong physics; refuse loudly instead."""
    with pytest.raises(InputError, match=match):
        _frozen_indices(constraints)


# ---------------------------------------------------------------------------
# optimizer selection + calibrated defaults
# ---------------------------------------------------------------------------

def test_build_optimizer_known_names():
    from ase import Atoms
    from ase.optimize import FIRE, LBFGS

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert isinstance(_build_optimizer("lbfgs", atoms, {}), LBFGS)
    assert isinstance(_build_optimizer("fire", atoms, {}), FIRE)


def test_build_optimizer_rejects_unknown():
    from ase import Atoms

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    with pytest.raises(InputError, match="Unknown ASE optimizer"):
        _build_optimizer("nonsense", atoms, {})


def test_calibrated_defaults_are_not_silently_changed():
    """These are measured values, not taste (CO/W52 vs geomeTRIC-tric minima):

    - fmax 0.005 eV/A reproduces the tric minimum to 0.02 kcal/mol; the nominal
      geomeTRIC gmax (0.023 eV/A) leaves structures ~0.7 kcal/mol under-relaxed.
    - plain LBFGS plateaus ~0.43 kcal/mol above the minimum at ANY fmax, so the
      preconditioned optimizer is the default.
    """
    assert DEFAULT_FMAX_EV_PER_ANGSTROM == 0.005
    assert DEFAULT_OPTIMIZER == "precon-lbfgs"


# ---------------------------------------------------------------------------
# end-to-end (needs a local MACE model)
# ---------------------------------------------------------------------------

def _model():
    m = os.environ.get("MACE_TEST_MODEL")
    if not m or not os.path.isfile(m):
        pytest.skip("set MACE_TEST_MODEL to a local .model file")
    return m


def _opt_input(model, **keywords):
    from qcelemental.models.v2 import Molecule, OptimizationInput

    mol = Molecule.from_data("O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\nC 1.5 1.5 4.0\nO 1.5 1.5 5.13")
    kw = {"program": "mace", "precon_r_cut": 5.0, "precon_r_NN": 1.0}
    kw.update(keywords)
    return mol, OptimizationInput(
        initial_molecule=mol,
        specification={
            "program": "ase",
            "protocols": {"trajectory_results": "all"},
            "keywords": kw,
            "specification": {
                "driver": "gradient",
                "program": "mace",
                "model": {"method": model, "basis": None},
                "keywords": {},
            },
        },
    )


def test_frozen_atoms_do_not_move():
    mol, inp = _opt_input(_model(), constraints={"freeze": [{"type": "xyz", "indices": [0, 1, 2]}]})
    res = qcng.compute(inp, "ase", raise_error=True)
    g0 = np.asarray(mol.geometry).reshape(-1, 3)
    g1 = np.asarray(res.final_molecule.geometry).reshape(-1, 3)
    disp = np.linalg.norm(g1 - g0, axis=1)
    assert disp[:3].max() == 0.0, "FixAtoms must hold frozen atoms exactly"
    assert disp[3:].max() > 1e-4, "mobile atoms should have relaxed"


def test_one_gradient_per_geometry_not_per_query():
    """ASE asks for energy and forces separately and repeatedly. Without the
    per-geometry cache each step fired ~5.7 gradient evaluations."""
    _, inp = _opt_input(_model())
    res = qcng.compute(inp, "ase", raise_error=True)
    steps = max(res.properties.optimization_iterations, 1)
    assert len(res.trajectory_results) / steps < 3.0


def test_trajectory_requires_the_protocol():
    """v2 defaults trajectory_results to 'none'; callers opt in."""
    from qcelemental.models.v2 import Molecule, OptimizationInput

    model = _model()
    mol, inp = _opt_input(model)
    assert len(qcng.compute(inp, "ase", raise_error=True).trajectory_results) > 0
    stripped = OptimizationInput(
        initial_molecule=mol,
        specification={
            "program": "ase",
            "keywords": {"program": "mace", "precon_r_cut": 5.0, "precon_r_NN": 1.0},
            "specification": {
                "driver": "gradient",
                "program": "mace",
                "model": {"method": model, "basis": None},
                "keywords": {},
            },
        },
    )
    assert len(qcng.compute(stripped, "ase", raise_error=True).trajectory_results) == 0


def test_non_convergence_is_an_error_not_a_silent_result():
    _, inp = _opt_input(_model(), maxiter=1, fmax=1e-8)
    res = qcng.compute(inp, "ase", raise_error=False)
    assert not res.success
    assert "did not converge" in str(res.error.error_message)


# ---------------------------------------------------------------------------
# Preconditioner stabilisation. ASE adds the diagonal stabilisation only when
# there are NO fixed atoms, assuming constraints remove the singular modes. That
# fails when a mobile fragment drifts beyond r_cut of everything else: it gets an
# empty neighbour list, a zero diagonal entry, and an EXACTLY SINGULAR matrix.
# Observed on ~19% of periodic slab sites (adsorbate desorbing into the vacuum
# gap); each stuck worker then spun in C at 100% CPU, never calling the
# calculator, until the queue deadlocked.
# ---------------------------------------------------------------------------

def test_precon_forces_stabilisation_by_default():
    import inspect

    from qcengine.procedures.ase import _build_optimizer

    src = inspect.getsource(_build_optimizer)
    assert 'keywords.get("precon_force_stab", True)' in src


def test_precon_matrix_is_nonsingular_for_a_detached_fragment():
    """The real failure: fixed atoms present AND a fragment outside r_cut of everything.

    Without force_stab the preconditioner is exactly singular and spsolve produces
    garbage; with it the solve is well posed.
    """
    import warnings

    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize.precon import Exp

    # small slab-like block, plus a molecule parked far away in "vacuum"
    pos = [[i * 2.0, j * 2.0, 0.0] for i in range(4) for j in range(4)]
    pos += [[3.0, 3.0, 40.0], [3.0, 3.0, 41.1]]     # detached, >> r_cut from the block
    atoms = Atoms("H16CO", positions=pos, cell=[20.0, 20.0, 80.0], pbc=[True, True, False])
    atoms.set_constraint(FixAtoms(indices=list(range(8))))   # fixed atoms present
    from ase.calculators.lj import LennardJones

    atoms.calc = LennardJones(rc=6.0)   # make_precon estimates mu from forces

    for force_stab, expect_singular in [(False, True), (True, False)]:
        precon = Exp(A=3.0, r_cut=5.0, r_NN=1.0, force_stab=force_stab)
        precon.make_precon(atoms)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y = precon.solve(np.ones(precon.P.shape[0]))
        singular = any("singular" in str(w.message).lower() for w in caught) or not np.all(
            np.isfinite(y)
        )
        assert singular is expect_singular, (
            f"force_stab={force_stab}: expected singular={expect_singular}, got {singular}"
        )
