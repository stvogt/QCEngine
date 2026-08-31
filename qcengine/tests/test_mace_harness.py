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
