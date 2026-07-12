import numpy as np
import pytest
import qcelemental
from qcelemental.testing import compare_values

import qcengine as qcng
from qcengine.exceptions import ConvergenceError, InputError, RandomError, ResourceError, UnknownError
from qcengine.programs.gaussian import (
    check_gaussian_errors,
    format_molecule,
    harvest_fchk,
    lt_to_square,
    normalize_basis,
    parse_fchk_array,
    parse_fchk_scalar,
)
from qcengine.testing import checkver_and_convert, from_v2, schema_versions, using, uusing


@pytest.fixture
def h2o_data():
    return """
            O 0.000000000000     0.000000000000    -0.068516245955
            H 0.000000000000    -0.790689888800     0.543701278274
            H 0.000000000000     0.790689888800     0.543701278274
    """


# ------------------------------------------------------------------
# Offline tests: parsers and error mapping, no Gaussian installation needed
# ------------------------------------------------------------------

# Layout verbatim from a real formatted checkpoint (G16 doc/ex.fchk):
# 40-char name field, type char, value(s); arrays 5 per line in %16.8E
_WATER_FCHK = """water
SP        RB3LYP                                                      def2SVP
Number of atoms                            I                3
Charge                                     I                0
Multiplicity                               I                1
Number of basis functions                  I               24
SCF Energy                                 R     -7.631000000000000E+01
Total Energy                               R     -7.631981019923900E+01
Cartesian Gradient                         R   N=           9
  6.02673261E-16  1.55171015E-15  1.57655616E-02 -1.06148385E-17 -7.85548622E-03
 -7.88278079E-03  4.51680295E-18  7.85548622E-03 -7.88278079E-03
Cartesian Force Constants                  R   N=          21
  1.00000000E+00  2.00000000E+00  3.00000000E+00  4.00000000E+00  5.00000000E+00
  6.00000000E+00  7.00000000E+00  8.00000000E+00  9.00000000E+00  1.00000000E+01
  1.10000000E+01  1.20000000E+01  1.30000000E+01  1.40000000E+01  1.50000000E+01
  1.60000000E+01  1.70000000E+01  1.80000000E+01  1.90000000E+01  2.00000000E+01
  2.10000000E+01
Mulliken Charges                           R   N=           3
 -3.00000000E-01  1.50000000E-01  1.50000000E-01
"""


def test_parse_fchk_scalar():
    assert compare_values(-76.319810199239, parse_fchk_scalar(_WATER_FCHK, "Total Energy"))
    assert compare_values(-76.31, parse_fchk_scalar(_WATER_FCHK, "SCF Energy"))
    assert parse_fchk_scalar(_WATER_FCHK, "Number of basis functions") == 24
    assert parse_fchk_scalar(_WATER_FCHK, "Multiplicity") == 1


def test_parse_fchk_scalar_missing():
    with pytest.raises(UnknownError):
        parse_fchk_scalar(_WATER_FCHK, "Nuclear Repulsion")


def test_parse_fchk_scalar_anchored():
    # "Energy" alone must not match inside "SCF Energy" or "Total Energy"
    with pytest.raises(UnknownError):
        parse_fchk_scalar(_WATER_FCHK, "Energy")


def test_parse_fchk_array():
    gradient = parse_fchk_array(_WATER_FCHK, "Cartesian Gradient")

    assert gradient.size == 9
    assert compare_values(1.57655616e-02, gradient[2])
    assert compare_values(-7.88278079e-03, gradient[8])


def test_parse_fchk_array_truncated():
    truncated = _WATER_FCHK[: _WATER_FCHK.index("6.00000000E+00")]
    with pytest.raises(UnknownError):
        parse_fchk_array(truncated, "Cartesian Force Constants")


def test_lt_to_square():
    packed = parse_fchk_array(_WATER_FCHK, "Cartesian Force Constants")
    full = lt_to_square(packed, 6)

    assert full.shape == (6, 6)
    assert compare_values(full, full.T)
    # fchk packs the lower triangle row-major: H(0,0), H(1,0), H(1,1), ...
    assert full[0, 0] == 1.0
    assert full[1, 0] == 2.0
    assert full[0, 1] == 2.0
    assert full[1, 1] == 3.0
    assert full[5, 5] == 21.0


def test_lt_to_square_size_mismatch():
    with pytest.raises(UnknownError):
        lt_to_square(np.arange(20, dtype=float), 6)


def test_harvest_fchk():
    props = harvest_fchk(_WATER_FCHK)

    assert compare_values(-76.319810199239, props["return_energy"])
    assert compare_values(-76.31, props["scf_total_energy"])
    assert props["calcinfo_nbasis"] == 24


@pytest.mark.parametrize(
    "stdout, stderr, exc",
    [
        ("The combination of multiplicity 2 and 10 electrons is impossible.", "", InputError),
        ("Error termination via Lnk1e in /opt/g16/l1.exe at Fri Jul 11.", "", InputError),
        ("A syntax error was detected in the input line.", "", InputError),
        ("Error termination via Lnk1e in /opt/g16/l301.exe at Fri Jul 11.", "", InputError),
        ("EOF while reading basis line.", "", InputError),
        # G16 terminates NORMALLY on unconverged SP SCF -- warning-only (verified RevB.01)
        (">>>>>>>>>> Convergence criterion not met.\n SCF Done:  E(RB3LYP) = -76.30", "", ConvergenceError),
        ("Convergence failure -- run terminated.", "", ConvergenceError),
        ("galloc:  could not allocate memory.", "", ResourceError),
        ("Erroneous write. Write 0 instead of 4096.", "", RandomError),
        ("", "Segmentation fault (core dumped)", RandomError),
        ("Error termination via Lnk1e in /opt/g16/l123.exe at Thu Jun  4.", "", UnknownError),
    ],
)
def test_check_gaussian_errors(stdout, stderr, exc):
    with pytest.raises(exc):
        check_gaussian_errors("#P hf/def2SVP", stdout, stderr)


def test_check_gaussian_errors_clean():
    check_gaussian_errors("#P hf/def2SVP", "Normal termination of Gaussian 16 at Fri Jul 11.", "")


@pytest.mark.parametrize(
    "basis, expected",
    [
        ("def2-SVP", "def2SVP"),
        ("def2-svp", "def2svp"),
        ("def2-TZVPP", "def2TZVPP"),
        ("cc-pVTZ", "cc-pVTZ"),
        ("aug-cc-pVDZ", "aug-cc-pVDZ"),
        ("6-31G*", "6-31G*"),
    ],
)
def test_normalize_basis(basis, expected):
    assert normalize_basis(basis) == expected


def _build_input(molecule, driver="gradient", keywords=None, ncores=4, memory=8.0):
    harness = qcng.get_program("gaussian", check=False)
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
    return job_inputs["infiles"]["input.gjf"]


def test_build_input_assembly(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, keywords={"route_input": "EmpiricalDispersion=GD3BJ", "link0": ["%rwf=big.rwf"]})

    lines = inp.splitlines()
    assert lines[0] == "%chk=input.chk"
    assert "%NProcShared=4" in lines
    assert "%Mem=6144MB" in lines  # 8 GiB * 1024 * 0.75
    assert "%rwf=big.rwf" in lines
    route = [ln for ln in lines if ln.startswith("#P")][0]
    assert route == "#P b3lyp/def2SVP Force EmpiricalDispersion=GD3BJ"
    assert "0 1" in lines
    # Gaussian requires the trailing blank line
    assert inp.endswith("\n\n")


def test_build_input_user_link0_wins(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, keywords={"link0": ["%NProcShared=2", "%Mem=1000MB"]})

    assert "%NProcShared=2" in inp
    assert "%NProcShared=4" not in inp
    assert "%Mem=1000MB" in inp
    assert "%Mem=6144MB" not in inp


def test_build_input_user_chk_forbidden(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    with pytest.raises(InputError):
        _build_input(h2o, keywords={"link0": ["%chk=mine.chk"]})


def test_build_input_unknown_keyword(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    with pytest.raises(InputError):
        _build_input(h2o, keywords={"scf_convergence": 8})


def test_build_input_bad_link0_line(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    with pytest.raises(InputError):
        _build_input(h2o, keywords={"link0": ["nprocshared=2"]})


def test_build_input_serial_no_nproc(h2o_data):
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, ncores=1)

    assert "%NProcShared" not in inp
    assert "%Mem=" in inp


def test_format_molecule_ghost():
    hehe = qcelemental.models.v2.Molecule.from_data(
        """
        He 0.0 0.0 0.0
        @He 0.0 0.0 3.0
        """
    )
    block = format_molecule(hehe)
    lines = block.splitlines()

    assert lines[0] == "0 1"
    assert lines[1].startswith("He ")
    assert lines[2].startswith("He-Bq")


def test_build_input_ghost_symmetry_none(h2o_data):
    hehe = qcelemental.models.v2.Molecule.from_data(
        """
        He 0.0 0.0 0.0
        @He 0.0 0.0 3.0
        """
    )
    inp = _build_input(hehe, driver="energy")
    route = [ln for ln in inp.splitlines() if ln.startswith("#P")][0]
    assert "Symmetry=None" in route

    # user-supplied symmetry choice wins
    inp = _build_input(hehe, driver="energy", keywords={"route_input": "NoSymm"})
    route = [ln for ln in inp.splitlines() if ln.startswith("#P")][0]
    assert "Symmetry=None" not in route
    assert "NoSymm" in route

    # no ghosts -> no symmetry override
    h2o = qcelemental.models.v2.Molecule.from_data(h2o_data)
    inp = _build_input(h2o, driver="energy")
    assert "Symmetry=None" not in inp


# ------------------------------------------------------------------
# Integration tests: require a Gaussian 16 installation
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
        pytest.param("hf", {}, -75.95536954370, id="hf", marks=using("gaussian")),
        pytest.param(
            "b3lyp", {"route_input": "EmpiricalDispersion=GD3BJ"}, None, id="b3lyp-gd3bj", marks=using("gaussian")
        ),
    ],
)
def test_gaussian_energy(method, keywords, ref_energy, h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(request.node.name, h2o, "energy", {"method": method, "basis": "def2-SVP"}, keywords)

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "gaussian", raise_error=True, return_dict=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    if "v2" in request.node.name:
        assert res["input_data"]["specification"]["driver"] == "energy"
    else:
        assert res["driver"] == "energy"
    assert res["success"] is True
    if ref_energy is not None:
        assert compare_values(ref_energy, res["return_result"], atol=1.0e-6)


@uusing("gaussian")
def test_gaussian_gradient(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(request.node.name, h2o, "gradient", {"method": "hf", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "gaussian", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True
    assert res.properties.return_energy
    assert res.properties.calcinfo_natom == 3
    # HF/def2-SVP gradient norm on this geometry (program-independent)
    assert compare_values(0.099340, np.linalg.norm(res.return_result), atol=1.0e-4)


@uusing("gaussian")
def test_gaussian_hessian(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(request.node.name, h2o, "hessian", {"method": "b3lyp", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "gaussian", raise_error=True, return_version=retver)
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


@uusing("gaussian")
def test_gaussian_scf_nonconvergence(h2o_data, schema_versions, request):
    models, retver, _ = schema_versions
    h2o = models.Molecule.from_data(h2o_data)

    resi = _make_resi(
        request.node.name,
        h2o,
        "energy",
        {"method": "b3lyp", "basis": "def2-SVP"},
        {"route_input": "SCF=(MaxCycle=2)"},
    )

    resi = checkver_and_convert(resi, request.node.name, "pre")
    with pytest.raises(ConvergenceError):
        qcng.compute(resi, "gaussian", raise_error=True, return_version=retver)


@uusing("gaussian")
def test_gaussian_ghost_atoms(schema_versions, request):
    models, retver, _ = schema_versions
    hehe = models.Molecule.from_data(
        """
        He 0.0 0.0 0.0
        @He 0.0 0.0 3.0
        """
    )

    resi = _make_resi(request.node.name, hehe, "energy", {"method": "hf", "basis": "def2-SVP"}, {})

    resi = checkver_and_convert(resi, request.node.name, "pre")
    res = qcng.compute(resi, "gaussian", raise_error=True, return_version=retver)
    res = checkver_and_convert(res, request.node.name, "post")

    assert res.success is True
    # ghost only contributes basis functions; energy is that of a single He atom
    assert res.return_result < -2.5
