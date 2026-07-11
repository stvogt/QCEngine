import numpy as np
import pytest
import qcelemental
from qcelemental.testing import compare_values

import qcengine as qcng
from qcengine.exceptions import ConvergenceError, InputError, RandomError, UnknownError
from qcengine.programs.orca import check_orca_errors, harvest_stdout, parse_engrad, parse_hessian
from qcengine.testing import checkver_and_convert, from_v2, schema_versions, using, uusing


@pytest.fixture
def h2o_data():
    return """
            O 0.000000000000     0.000000000000    -0.068516245955
            H 0.000000000000    -0.790689888800     0.543701278274
            H 0.000000000000     0.790689888800     0.543701278274
    """


# ------------------------------------------------------------------
# Offline tests: parsers and error mapping, no ORCA installation needed
# ------------------------------------------------------------------

_CO_ENGRAD = """#
# Number of atoms
#
 2
#
# The current total energy in Eh
#
   -113.161159604822
#
# The current gradient in Eh/bohr
#
      -0.000000000003
      -0.000000000009
      -0.000017969626
       0.000000000003
       0.000000000009
       0.000017969626
#
# The atomic numbers and current coordinates in Bohr
#
   6     0.0000000    0.0000000   -0.0074401
   8     0.0000000    0.0000000    2.1390512
"""

# 6x6 symmetric matrix in ORCA's column-blocked layout (5 columns, then 1)
_DIATOMIC_HESS = """
$orca_hessian_file

$act_energy
     -113.161160

$hessian
6
                    0                  1                  2                  3                  4
    0      1.0000000000E-01   1.0000000000E-02   2.0000000000E-02  -1.0000000000E-01   3.0000000000E-02
    1      1.0000000000E-02   2.0000000000E-01   4.0000000000E-02   5.0000000000E-02  -2.0000000000E-01
    2      2.0000000000E-02   4.0000000000E-02   3.0000000000E-01   6.0000000000E-02   7.0000000000E-02
    3     -1.0000000000E-01   5.0000000000E-02   6.0000000000E-02   4.0000000000E-01   8.0000000000E-02
    4      3.0000000000E-02  -2.0000000000E-01   7.0000000000E-02   8.0000000000E-02   5.0000000000E-01
    5      4.0000000000E-02   9.0000000000E-02  -3.0000000000E-01   1.0000000000E-02   2.0000000000E-02
                    5
    0      4.0000000000E-02
    1      9.0000000000E-02
    2     -3.0000000000E-01
    3      1.0000000000E-02
    4      2.0000000000E-02
    5      6.0000000000E-01

$vibrational_frequencies
"""


def test_parse_engrad():
    natoms, energy, gradient = parse_engrad(_CO_ENGRAD)

    assert natoms == 2
    assert compare_values(-113.161159604822, energy)
    assert gradient.shape == (2, 3)
    assert compare_values(-0.000017969626, gradient[0, 2])
    assert compare_values(0.000017969626, gradient[1, 2])


def test_parse_engrad_truncated():
    truncated = "\n".join(_CO_ENGRAD.splitlines()[:12])
    with pytest.raises(UnknownError):
        parse_engrad(truncated)


def test_parse_hessian():
    hessian = parse_hessian(_DIATOMIC_HESS)

    assert hessian.shape == (6, 6)
    assert not np.isnan(hessian).any()
    # spot-check both column blocks and symmetry
    assert compare_values(0.1, hessian[0, 0])
    assert compare_values(0.6, hessian[5, 5])
    assert compare_values(-0.3, hessian[5, 2])
    assert compare_values(hessian, hessian.T)


def test_parse_hessian_no_block():
    with pytest.raises(UnknownError):
        parse_hessian("$orca_hessian_file\n\n$act_energy\n -1.0\n")


def test_harvest_stdout_last_energy_wins():
    stdout = """
-------------------------   --------------------
FINAL SINGLE POINT ENERGY      -113.100000000000
-------------------------   --------------------

Total Energy       :         -113.15000000 Eh
Nuclear Repulsion  :           22.51234567 Eh
SCF CONVERGED AFTER  12 CYCLES

-------------------------   --------------------
FINAL SINGLE POINT ENERGY      -113.161159604822
-------------------------   --------------------
"""
    props = harvest_stdout(stdout)

    assert compare_values(-113.161159604822, props["return_energy"])
    assert compare_values(-113.15, props["scf_total_energy"])
    assert compare_values(22.51234567, props["nuclear_repulsion_energy"])
    assert props["scf_iterations"] == 12


@pytest.mark.parametrize(
    "stdout, stderr, exc",
    [
        ("... INPUT ERROR ...", "", InputError),
        ("UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE", "", InputError),
        ("Error : multiplicity (3) is even and number of electrons (14) is even", "", InputError),
        ("SCF NOT CONVERGED AFTER 8 CYCLES", "", ConvergenceError),
        ("This wavefunction IS NOT FULLY CONVERGED!", "", ConvergenceError),
        ("", "Signal: Segmentation fault (11)", RandomError),
        ("ORCA finished by error termination in SCF", "", UnknownError),
    ],
)
def test_check_orca_errors(stdout, stderr, exc):
    with pytest.raises(exc):
        check_orca_errors("! hf def2-svp", stdout, stderr)


def test_check_orca_errors_clean():
    check_orca_errors("! hf def2-svp", "ORCA TERMINATED NORMALLY", "")


def _build_input(molecule, driver="gradient", keywords=None, ncores=4, memory=8.0):
    harness = qcng.get_program("orca", check=False)
    config = qcng.config.get_config(task_config={"ncores": ncores, "memory": memory, "nnodes": 1})
    atin = qcelemental.models.v2.AtomicInput(
        molecule=molecule,
        specification={
            "driver": driver,
            "model": {"method": "b3lyp", "basis": "def2-SVP"},
            "keywords": keywords or {},
        },
    )
    job_inputs = harness.build_input(atin, config)
    return job_inputs["infiles"]["input.inp"]


def test_build_input_assembly(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, keywords={"simple_input": "D3BJ TightSCF", "blocks": "%scf maxiter 150 end"})

    lines = inp.splitlines()
    assert lines[0] == "! b3lyp def2-SVP EnGrad D3BJ TightSCF"
    assert "%pal nprocs 4 end" in inp
    assert "%maxcore 1536" in inp  # 8 GiB * 1024 * 0.75 / 4
    assert "%scf maxiter 150 end" in inp
    assert "*xyz 0 1" in inp
    assert inp.rstrip().endswith("*")


def test_build_input_user_pal_maxcore_win(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, keywords={"blocks": ["%pal nprocs 2 end", "%maxcore 1000"]})

    assert "%pal nprocs 2 end" in inp
    assert "%pal nprocs 4 end" not in inp
    assert "%maxcore 1000" in inp
    assert "%maxcore 1536" not in inp


def test_build_input_unknown_keyword(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    with pytest.raises(InputError):
        _build_input(h2o, keywords={"scf_conv": 8})


def test_build_input_serial_no_pal(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, ncores=1)

    assert "%pal" not in inp
    assert "%maxcore" in inp


# ------------------------------------------------------------------
# Integration tests: require an ORCA installation
# ------------------------------------------------------------------


def _make_resi(node_name, molecule, driver, model, keywords):
    """Build the input dict in the flat (v1) or nested-specification (v2) shape."""
    if from_v2(node_name):
        return {"molecule": molecule, "specification": {"driver": driver, "model": model, "keywords": keywords}}
    return {"molecule": molecule, "driver": driver, "model": model, "keywords": keywords}

@pytest.mark.parametrize(
    "method, keywords, ref_energy",
    [
        # HF is functional-definition-independent; reference from HF/def2-SVP on this geometry
        # NOTE: explicit ids -- a bare `None` param would leak "None" into the test id,
        # which checkver_and_convert substring-matches for the unversioned variant
        pytest.param("hf", {}, -75.95536954370, id="hf", marks=using("orca")),
        pytest.param("b3lyp", {"simple_input": "D3BJ"}, None, id="b3lyp-d3bj", marks=using("orca")),
    ],
)
def test_orca_energy(method, keywords, ref_energy, h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(
        request.node.name, h2o, "energy", {"method": method, "basis": "def2-SVP"}, keywords
    )

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "orca", raise_error=True, return_dict=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    if "v2" in request.node.name:
        assert res["input_data"]["specification"]["driver"] == "energy"
    else:
        assert res["driver"] == "energy"
    assert res["success"] is True
    if ref_energy is not None:
        assert compare_values(ref_energy, res["return_result"], atol=1.0e-6)


@uusing("orca")
def test_orca_gradient(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(request.node.name, h2o, "gradient", {"method": "hf", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "orca", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True
    assert res.properties.return_energy
    assert res.properties.calcinfo_natom == 3
    # HF/def2-SVP gradient norm on this geometry (program-independent)
    assert compare_values(0.099340, np.linalg.norm(res.return_result), atol=1.0e-4)


@uusing("orca")
def test_orca_hessian(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(request.node.name, h2o, "hessian", {"method": "b3lyp", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "orca", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True

    hessian = np.asarray(res.return_result)
    assert hessian.shape == (9, 9)
    assert compare_values(hessian, hessian.T, atol=1.0e-6)
    # Translations are exactly zero; on a non-stationary geometry the three
    # rotations pick up real curvature, so the remaining six are all > 0.
    eigvals = np.linalg.eigvalsh(hessian)
    assert (np.abs(eigvals) < 1.0e-6).sum() == 3
    assert (eigvals > 1.0e-3).sum() == 6


@uusing("orca")
def test_orca_user_blocks(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(
        request.node.name,
        h2o,
        "energy",
        {"method": "b3lyp", "basis": "def2-SVP"},
        {"simple_input": "RIJCOSX def2/J TightSCF", "blocks": ["%scf maxiter 150 end", "%maxcore 1000"]},
    )

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "orca", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True


@uusing("orca")
def test_orca_scf_nonconvergence(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(
        request.node.name,
        h2o,
        "energy",
        {"method": "b3lyp", "basis": "def2-SVP"},
        {"blocks": "%scf maxiter 2 end"},
    )

    resi = checkver_and_convert(resi, request.node.name, "pre")
    with pytest.raises(ConvergenceError):
        qcng.compute(resi, "orca", raise_error=True, return_version=retver)


@uusing("orca")
def test_orca_ghost_atoms(schema_versions, request):
    models, retver, _ = schema_versions
    hehe = models.Molecule.from_data(
        """
        He 0.0 0.0 0.0
        @He 0.0 0.0 3.0
        """
    )

    resi = _make_resi(request.node.name, hehe, "energy", {"method": "hf", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "orca", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True
    # ghost only contributes basis functions; energy is that of a single He atom
    assert res.return_result < -2.5
