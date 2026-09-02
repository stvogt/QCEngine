"""Unit tests for the MACE harness helpers that don't require a
functioning MACE install (no mace, torch, or e3nn imports).
"""
import numpy as np
import pytest

from qcengine.exceptions import InputError
from qcengine.programs.mace import _parse_periodic_keywords


def test_parse_periodic_keywords_default_nonperiodic():
    """Empty / missing keywords → non-periodic (pbc all False, cell None)."""
    assert _parse_periodic_keywords(None) == ((False, False, False), None)
    assert _parse_periodic_keywords({}) == ((False, False, False), None)


def test_parse_periodic_keywords_pbc_all_false_is_nonperiodic():
    """pbc=[False,False,False] explicitly is treated as non-periodic."""
    pbc, cell = _parse_periodic_keywords({"pbc": [False, False, False], "cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    assert pbc == (False, False, False)
    assert cell is None


def test_parse_periodic_keywords_full_pbc():
    """Fully periodic input with a 3x3 cell is passed through."""
    cell_in = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 30.0]]
    pbc, cell = _parse_periodic_keywords({"pbc": [True, True, True], "cell": cell_in})
    assert pbc == (True, True, True)
    assert isinstance(cell, np.ndarray)
    assert cell.shape == (3, 3)
    np.testing.assert_array_equal(cell, np.asarray(cell_in, dtype=float))


def test_parse_periodic_keywords_slab_pbc_xy_only():
    """2D-periodic slab: pbc=[True,True,False] is a legitimate use case."""
    cell_in = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 40.0]]
    pbc, cell = _parse_periodic_keywords({"pbc": [True, True, False], "cell": cell_in})
    assert pbc == (True, True, False)
    np.testing.assert_array_equal(cell, np.asarray(cell_in, dtype=float))


def test_parse_periodic_keywords_pbc_without_cell_raises():
    """Periodicity requested but no cell → hard error, no silent fallback."""
    with pytest.raises(InputError, match="no cell supplied"):
        _parse_periodic_keywords({"pbc": [True, True, True]})


def test_parse_periodic_keywords_wrong_pbc_length_raises():
    with pytest.raises(InputError, match="length 3"):
        _parse_periodic_keywords({"pbc": [True, True], "cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})


def test_parse_periodic_keywords_wrong_cell_shape_raises():
    with pytest.raises(InputError, match="3x3"):
        _parse_periodic_keywords({"pbc": [True, True, True], "cell": [[1, 0, 0], [0, 1, 0]]})


# ---------------------------------------------------------------------------
# Portable model resolution: a QCFractal spec is shared by every worker that may
# run the record, so a machine-specific absolute path pins it to one cluster (and
# one username). Bare filename + $MACE_MODEL_PATH, or a ~-relative path, travel.
# ---------------------------------------------------------------------------
import os

import pytest

from qcengine.programs.mace import MACEHarness


@pytest.fixture
def fake_model(tmp_path):
    f = tmp_path / "lmft-co-d-v0.model"
    f.write_bytes(b"not-a-real-model")
    return f


def test_resolve_absolute_path_unchanged(fake_model):
    """Existing specs carry absolute paths and must keep resolving."""
    assert MACEHarness.resolve_model_path(str(fake_model)) == str(fake_model)


def test_resolve_bare_filename_via_env(fake_model, monkeypatch):
    monkeypatch.setenv("MACE_MODEL_PATH", str(fake_model.parent))
    assert MACEHarness.resolve_model_path(fake_model.name) == str(fake_model)


def test_resolve_bare_filename_searches_all_roots(fake_model, monkeypatch):
    monkeypatch.setenv(
        "MACE_MODEL_PATH", os.pathsep.join(["/nonexistent/a", str(fake_model.parent)])
    )
    assert MACEHarness.resolve_model_path(fake_model.name) == str(fake_model)


def test_resolve_bare_filename_without_env_is_left_alone(fake_model, monkeypatch):
    """Unresolved names pass through so load_model raises one clear error."""
    monkeypatch.delenv("MACE_MODEL_PATH", raising=False)
    assert MACEHarness.resolve_model_path(fake_model.name) == fake_model.name


def test_env_is_not_consulted_when_a_directory_is_given(fake_model, monkeypatch):
    """A path with a directory component is explicit; don't silently search elsewhere."""
    monkeypatch.setenv("MACE_MODEL_PATH", str(fake_model.parent))
    assert MACEHarness.resolve_model_path("sub/dir/other.model") == "sub/dir/other.model"


def test_resolve_home_relative_path(fake_model, monkeypatch):
    """~-relative paths work across clusters even when the username differs."""
    monkeypatch.setenv("HOME", str(fake_model.parent))
    assert MACEHarness.resolve_model_path("~/" + fake_model.name) == str(fake_model)


def test_missing_model_error_names_the_path_and_the_env_var(monkeypatch):
    from qcengine.exceptions import InputError

    monkeypatch.delenv("MACE_MODEL_PATH", raising=False)
    with pytest.raises(InputError) as exc:
        MACEHarness().load_model("definitely-missing.model")
    msg = str(exc.value)
    assert "definitely-missing.model" in msg
    assert "MACE_MODEL_PATH" in msg


# ---------------------------------------------------------------------------
# GPU device handling. The harness moves the model to CUDA when available; the
# input batch and the returned tensors must travel with it, or you get
# "Expected all tensors to be on the same device" (inputs) or a .numpy() failure
# on CUDA tensors (results). Neither shows up on a CPU-only box, which is how it
# reached production twice.
# ---------------------------------------------------------------------------

def _has_cuda():
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available()


def test_mace_harness_moves_inputs_and_results_with_the_model():
    """Source-level guard: runs everywhere, including CPU-only CI."""
    import inspect

    from qcengine.programs.mace import MACEHarness

    src = inspect.getsource(MACEHarness.compute)
    assert "next(iter(data_loader)).to(device)" in src, "input batch must move to the model's device"
    assert '_energy = mace_data["energy"].detach().cpu()' in src, "energy must come back to host"
    assert '_forces = mace_data["forces"].detach().cpu()' in src, "forces must come back to host"
    # the pre-fix spellings must not creep back
    assert "next(iter(data_loader)).to_dict()" not in src
    assert 'mace_data["energy"] * ureg' not in src


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA device available")
def test_mace_energy_matches_between_cpu_and_gpu(tmp_path, monkeypatch):
    """On a GPU box, the CUDA and CPU paths must agree."""
    import torch
    import qcengine as qcng
    import qcelemental as qcel

    model = os.environ.get("MACE_TEST_MODEL")
    if not model or not os.path.isfile(model):
        pytest.skip("set MACE_TEST_MODEL to a local .model file")
    mol = qcel.models.Molecule.from_data("O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0")
    mk = lambda: qcel.models.AtomicInput(
        molecule=mol, driver="energy", model={"method": model, "basis": None}, keywords={}
    )
    gpu = qcng.compute(mk(), "mace", raise_error=True).return_result
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu = qcng.compute(mk(), "mace", raise_error=True).return_result
    assert gpu == pytest.approx(cpu, rel=1e-8)


# ---------------------------------------------------------------------------
# Precision selection. float64 is the default (binding energies need it), but a
# 1500-atom periodic slab needs ~43 GB of a 44 GB L40S in float64 and OOMs;
# float32 halves that and is fine for geometry optimization.
# ---------------------------------------------------------------------------

def test_mace_dtype_defaults_to_float64_and_honours_the_env(monkeypatch):
    torch = pytest.importorskip("torch")
    import qcelemental as qcel
    import qcengine as qcng

    model = os.environ.get("MACE_TEST_MODEL")
    if not model or not os.path.isfile(model):
        pytest.skip("set MACE_TEST_MODEL to a local .model file")
    mol = qcel.models.Molecule.from_data("O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0")
    inp = lambda: qcel.models.AtomicInput(
        molecule=mol, driver="energy", model={"method": model, "basis": None}, keywords={}
    )

    monkeypatch.delenv("MACE_DTYPE", raising=False)
    qcng.compute(inp(), "mace", raise_error=True)
    assert torch.get_default_dtype() is torch.float64

    monkeypatch.setenv("MACE_DTYPE", "float32")
    qcng.compute(inp(), "mace", raise_error=True)
    assert torch.get_default_dtype() is torch.float32

    monkeypatch.setenv("MACE_DTYPE", "float64")
    qcng.compute(inp(), "mace", raise_error=True)
    assert torch.get_default_dtype() is torch.float64


def test_mace_dtype_is_read_from_env_not_hardcoded():
    """Source-level guard: runs on CPU-only CI, catches a revert to a fixed dtype."""
    import inspect

    from qcengine.programs.mace import MACEHarness

    src = inspect.getsource(MACEHarness.compute)
    assert "self.DTYPE_ENV" in src, "precision must come from $MACE_DTYPE"
    assert "torch.set_default_dtype(torch.float64)" not in src, "dtype must not be hardcoded"


# ---------------------------------------------------------------------------
# cuEquivariance: a hardware switch that must never change the numbers.
# Measured (float64, RTX3080): dE = 6e-9 kcal/mol, max|dG| = 5e-10, periodic and
# not; 4.6 GiB per 1000 atoms vs ~29 for e3nn, which is what makes float64
# affordable on a 1500-atom slab (~6.8 GiB instead of ~43).
# ---------------------------------------------------------------------------

def test_cueq_is_env_switched_not_spec_keyword():
    """Keeping it out of the spec means existing records stay valid and comparable;
    it is numerically identical, so it is a deployment choice, not part of the method."""
    from qcengine.programs.mace import MACEHarness

    assert MACEHarness.CUEQ_ENV == "MACE_CUEQ"


def test_cueq_model_cache_is_keyed_on_the_flag():
    """A cuEq-converted model is not interchangeable with an e3nn one, so serving a
    cached model built under the other setting would silently mix backends."""
    import inspect

    from qcengine.programs.mace import MACEHarness

    src = inspect.getsource(MACEHarness.load_model)
    assert "use_cueq" in src
    assert "(resolved, dtype, use_cueq)" in src, "cache key must include the cuEq flag"
    assert "run_e3nn_to_cueq(model" in src
    # a converted model must not be pushed through e3nn's jit
    assert src.index("run_e3nn_to_cueq") < src.index("jit.compile(model)")


def test_cueq_requested_without_the_package_raises_clearly(monkeypatch):
    from qcengine.exceptions import ResourceError
    from qcengine.programs.mace import MACEHarness

    monkeypatch.setenv("MACE_CUEQ", "1")
    monkeypatch.setitem(__import__("sys").modules, "mace.cli.convert_e3nn_cueq", None)
    with pytest.raises((ResourceError, Exception)) as exc:
        MACEHarness().load_model("definitely-missing.model")
    # either the clear cuEq message or the model-not-found one; both are actionable
    assert "MACE_CUEQ" in str(exc.value) or "not found" in str(exc.value)
