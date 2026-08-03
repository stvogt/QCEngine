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
