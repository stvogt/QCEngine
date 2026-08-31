"""Unit tests for the periodic-dispersion helpers in dftd_ng.

Pure-python helpers only — no dftd3/dftd4 install required.
"""
import numpy as np
import pytest

from qcengine.exceptions import InputError
from qcengine.programs.dftd_ng import (
    _parse_periodic_keywords,
    _resolve_d3_damping_class,
    _strip_dispersion_level,
)


# ---------------------------------------------------------------------------
# _parse_periodic_keywords
# ---------------------------------------------------------------------------

def test_parse_periodic_keywords_default_nonperiodic():
    assert _parse_periodic_keywords(None) == (None, None)
    assert _parse_periodic_keywords({}) == (None, None)


def test_parse_periodic_keywords_all_false_pbc_is_nonperiodic():
    lat, pbc = _parse_periodic_keywords({
        "pbc": [False, False, False],
        "cell": [[10.0, 0, 0], [0, 10, 0], [0, 0, 10]],
    })
    assert lat is None and pbc is None


def test_parse_periodic_keywords_full_periodic_converts_to_bohr():
    lat, pbc = _parse_periodic_keywords({
        "pbc": [True, True, True],
        "cell": [[10.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]],
    })
    # 1 Angstrom = 1.8897259886 Bohr
    assert lat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(lat), [10.0, 20.0, 30.0] * np.array(1.8897259886), rtol=1e-6)
    np.testing.assert_array_equal(pbc, [True, True, True])


def test_parse_periodic_keywords_slab_xy_only():
    """2D slab: pbc=[T,T,F] is a common case."""
    lat, pbc = _parse_periodic_keywords({
        "pbc": [True, True, False],
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 30.0]],
    })
    np.testing.assert_array_equal(pbc, [True, True, False])


def test_parse_periodic_keywords_pbc_without_cell_raises():
    with pytest.raises(InputError, match="no cell supplied"):
        _parse_periodic_keywords({"pbc": [True, True, True]})


def test_parse_periodic_keywords_wrong_pbc_length_raises():
    with pytest.raises(InputError, match="length 3"):
        _parse_periodic_keywords({
            "pbc": [True, True],
            "cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        })


def test_parse_periodic_keywords_wrong_cell_shape_raises():
    with pytest.raises(InputError, match="3x3"):
        _parse_periodic_keywords({
            "pbc": [True, True, True],
            "cell": [[1, 0, 0], [0, 1, 0]],
        })


# ---------------------------------------------------------------------------
# _resolve_d3_damping_class
# ---------------------------------------------------------------------------

def test_resolve_d3_damping_class_from_level_hint():
    dftd3 = pytest.importorskip("dftd3")
    from dftd3.interface import (
        RationalDampingParam,
        ZeroDampingParam,
        ModifiedRationalDampingParam,
        ModifiedZeroDampingParam,
        OptimizedPowerDampingParam,
    )
    assert _resolve_d3_damping_class("d3bj", None) is RationalDampingParam
    assert _resolve_d3_damping_class("d3zero", None) is ZeroDampingParam
    assert _resolve_d3_damping_class("d3mbj", None) is ModifiedRationalDampingParam
    assert _resolve_d3_damping_class("d3mzero", None) is ModifiedZeroDampingParam
    assert _resolve_d3_damping_class("d3op", None) is OptimizedPowerDampingParam


def test_resolve_d3_damping_class_from_method_suffix():
    dftd3 = pytest.importorskip("dftd3")
    from dftd3.interface import RationalDampingParam, ZeroDampingParam
    assert _resolve_d3_damping_class(None, "b3lyp-d3bj") is RationalDampingParam
    assert _resolve_d3_damping_class(None, "b3lyp-d3") is ZeroDampingParam


def test_resolve_d3_damping_class_defaults_to_rational():
    """Unrecognised → RationalDampingParam (D3BJ), matching s-dftd3's own default."""
    dftd3 = pytest.importorskip("dftd3")
    from dftd3.interface import RationalDampingParam
    assert _resolve_d3_damping_class(None, None) is RationalDampingParam
    assert _resolve_d3_damping_class("unknown", "unknown") is RationalDampingParam


# ---------------------------------------------------------------------------
# End-to-end smoke tests — only run when the underlying package is installed
# ---------------------------------------------------------------------------

def test_periodic_dftd4_returns_different_energy_from_nonperiodic():
    """Sanity check: periodic and non-periodic H2 differ (lattice-image contribution)."""
    dftd4 = pytest.importorskip("dftd4")
    from qcelemental.models.v2 import AtomicInput
    from qcengine.programs.dftd_ng import DFTD4Harness

    mol = {
        "symbols": ["H", "H"],
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
    }
    base = {
        "molecule": mol,
        "specification": {
            "model": {"method": "b3lyp"},
            "driver": "energy",
            "extras": {},
        },
        "id": None,
    }
    harness = DFTD4Harness()
    non_periodic = harness.compute(AtomicInput(**{**base, "specification": {**base["specification"], "keywords": {}}}), None)
    periodic = harness.compute(AtomicInput(**{**base, "specification": {**base["specification"], "keywords": {
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        "pbc": [True, True, True],
    }}}), None)
    assert non_periodic.success and periodic.success
    assert non_periodic.properties.return_energy != periodic.properties.return_energy
    # Periodic energy is more negative (more attractive pairs from images)
    assert periodic.properties.return_energy < non_periodic.properties.return_energy


def test_periodic_dftd3_returns_different_energy_from_nonperiodic():
    dftd3 = pytest.importorskip("dftd3")
    from qcelemental.models.v2 import AtomicInput
    from qcengine.programs.dftd_ng import SDFTD3Harness

    mol = {
        "symbols": ["H", "H"],
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
    }
    base = {
        "molecule": mol,
        "specification": {
            "model": {"method": "b3lyp"},
            "driver": "energy",
            "extras": {},
        },
        "id": None,
    }
    harness = SDFTD3Harness()
    non_periodic = harness.compute(AtomicInput(**{**base, "specification": {**base["specification"], "keywords": {"level_hint": "d3bj"}}}), None)
    periodic = harness.compute(AtomicInput(**{**base, "specification": {**base["specification"], "keywords": {
        "level_hint": "d3bj",
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        "pbc": [True, True, True],
    }}}), None)
    assert non_periodic.success and periodic.success
    assert non_periodic.properties.return_energy != periodic.properties.return_energy
    assert periodic.properties.return_energy < non_periodic.properties.return_energy


# ---------------------------------------------------------------------------
# _strip_dispersion_level — the parameter DBs are keyed by the BARE functional
# ---------------------------------------------------------------------------

_D3 = (lambda level: level.startswith("d3"))
_D4 = (lambda level: level in ("d4bjeeqatm", "d4bjeeqtwo"))


@pytest.mark.parametrize(
    "method, accept, expected",
    [
        ("mpwb1k-d3bj", _D3, "mpwb1k"),
        ("b3lyp-d3", _D3, "b3lyp"),
        ("mpwb1k-d4", _D4, "mpwb1k"),
        ("mpwb1k", _D3, "mpwb1k"),        # already bare -> unchanged
        ("mpwb1k-d4", _D3, "mpwb1k-d4"),  # other harness's level -> untouched
        (None, _D3, None),
        ("", _D3, ""),
    ],
)
def test_strip_dispersion_level(method, accept, expected):
    assert _strip_dispersion_level(method, accept) == expected


def test_strip_dispersion_level_is_idempotent():
    once = _strip_dispersion_level("mpwb1k-d3bj", _D3)
    assert _strip_dispersion_level(once, _D3) == once


def _periodic_energy(harness_cls, method, keywords):
    from qcelemental.models.v2 import AtomicInput
    inp = AtomicInput(**{
        "molecule": {"symbols": ["O", "H", "H"],
                     "geometry": [0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.45, 1.76, 0.0]},
        "specification": {"model": {"method": method}, "driver": "energy",
                          "keywords": keywords, "extras": {}},
        "id": None,
    })
    out = harness_cls().compute(inp, None)
    assert out.success
    return out.properties.return_energy


_CELL = {"cell": [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 20.0]], "pbc": [True, True, False]}


def test_periodic_dftd3_accepts_suffixed_method_name():
    """Regression: the periodic branch used to hand 'mpwb1k-d3bj' straight to the
    parameter DB and die with "No entry for 'mpwb1k-d3bj' present", while the
    non-periodic path stripped the level first. Both spellings must now resolve to
    the same MPWB1K D3BJ parameters."""
    pytest.importorskip("dftd3")
    from qcengine.programs.dftd_ng import SDFTD3Harness
    suffixed = _periodic_energy(SDFTD3Harness, "mpwb1k-d3bj", dict(_CELL))
    bare = _periodic_energy(SDFTD3Harness, "mpwb1k", {**_CELL, "level_hint": "d3bj"})
    assert suffixed == pytest.approx(bare, rel=0, abs=1e-12)


def test_periodic_dftd4_accepts_suffixed_method_name():
    pytest.importorskip("dftd4")
    from qcengine.programs.dftd_ng import DFTD4Harness
    suffixed = _periodic_energy(DFTD4Harness, "mpwb1k-d4", dict(_CELL))
    bare = _periodic_energy(DFTD4Harness, "mpwb1k", {**_CELL, "level_hint": "d4"})
    assert suffixed == pytest.approx(bare, rel=0, abs=1e-12)


def test_suffixed_periodic_still_differs_from_nonperiodic():
    """Guard against 'fixing' the name handling by silently dropping periodicity."""
    pytest.importorskip("dftd3")
    from qcengine.programs.dftd_ng import SDFTD3Harness
    periodic = _periodic_energy(SDFTD3Harness, "mpwb1k-d3bj", dict(_CELL))
    non_periodic = _periodic_energy(SDFTD3Harness, "mpwb1k-d3bj", {})
    assert periodic != non_periodic
