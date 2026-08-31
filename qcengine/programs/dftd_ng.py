"""
Harness for the DFT-D dispersion correction.
This implementation interfaces with the dftd3 and dftd4 Python-API, which provides
native support for QCSchema.

Therefore, this harness only has to provide a thin wrapper to integrate the
respective dispersion correction.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple

import qcelemental
from qcelemental.models.v2 import AtomicInput, AtomicResult, FailedOperation, Provenance
from qcelemental.util import parse_version, safe_version, which_import

from ..config import TaskConfig
from ..exceptions import InputError, ResourceError
from ..units import ureg
from .empirical_dispersion_resources import from_arrays, get_dispersion_aliases
from .model import ProgramHarness

_ANG2BOHR = ureg.conversion_factor("angstrom", "bohr")


def _parse_periodic_keywords(
    keywords: Optional[Dict[str, Any]],
):
    """Extract periodic-boundary settings from a QC-spec keywords dict.

    Returns ``(lattice_bohr, pbc)`` where:
    - ``lattice_bohr`` is a numpy (3, 3) array (Angstrom cell → Bohr), or ``None``
    - ``pbc`` is a numpy length-3 bool array, or ``None``

    Both are ``None`` when the keywords don't request periodicity (either
    ``pbc`` is missing or every axis is False) — that path defers to the
    non-periodic qcschema route, byte-identical to prior behaviour.

    Any-pbc-True with no cell → ``InputError`` (never silently non-periodic).
    Bad shapes → ``InputError``. Recognised keys:
        ``pbc``:  length-3 iterable of bools
        ``cell``: 3x3 lattice vectors in Angstrom
    """
    import numpy as np

    kw = keywords or {}
    kw_pbc = kw.get("pbc")
    kw_cell = kw.get("cell")

    if kw_pbc is None or not any(bool(x) for x in kw_pbc):
        return None, None

    pbc_arr = np.asarray([bool(x) for x in kw_pbc])
    if pbc_arr.shape != (3,):
        raise InputError(
            f"Dispersion harness: keywords['pbc'] must be length 3 (got {pbc_arr.shape})."
        )
    if kw_cell is None:
        raise InputError(
            "Dispersion harness: periodic boundary conditions requested "
            f"(keywords['pbc'] = {list(pbc_arr)}) but no cell supplied. "
            "Set keywords['cell'] to a 3x3 list of lattice vectors in Angstrom."
        )
    lattice_ang = np.asarray(kw_cell, dtype=float)
    if lattice_ang.shape != (3, 3):
        raise InputError(
            "Dispersion harness: keywords['cell'] must be a 3x3 matrix "
            f"(got shape {lattice_ang.shape})."
        )
    return lattice_ang * _ANG2BOHR, pbc_arr


def _resolve_d3_damping_class(level_hint: Optional[str], method: Optional[str]):
    """Pick the s-dftd3 DampingParam class for a given damping variant.

    Resolution order: ``level_hint`` (if set) then the trailing suffix of
    ``method`` (e.g. ``-d3bj``). Falls back to ``RationalDampingParam``
    (D3BJ) when neither is recognised, matching s-dftd3's own default.
    """
    from dftd3.interface import (
        RationalDampingParam,
        ZeroDampingParam,
        ModifiedRationalDampingParam,
        ModifiedZeroDampingParam,
        OptimizedPowerDampingParam,
    )

    tag = (level_hint or method or "").lower()
    if "d3mbj" in tag:
        return ModifiedRationalDampingParam
    if "d3mzero" in tag or "d3m0" in tag:
        return ModifiedZeroDampingParam
    if "d3op" in tag:
        return OptimizedPowerDampingParam
    if "d3zero" in tag or tag.endswith("-d3") or tag == "d3":
        return ZeroDampingParam
    return RationalDampingParam


def _strip_dispersion_level(method: Optional[str], accept) -> Optional[str]:
    """Drop a trailing dispersion-level alias from ``method`` ("mpwb1k-d3bj" -> "mpwb1k").

    The dftd3/dftd4 parameter databases are keyed by the BARE functional, so the
    suffixed spelling raises "No entry for 'mpwb1k-d3bj' present" /
    "Functional 'mpwb1k-d4' not known". The non-periodic qcschema path strips this
    before handing the name over; the periodic branches must do the same or they
    reject exactly the spelling callers use.

    ``accept`` filters which canonical levels this harness owns, mirroring the
    predicate each non-periodic path already applies (d3: ``level.startswith("d3")``;
    d4: ``level in {"d4bjeeqatm", "d4bjeeqtwo"}``).

    Note the ordering constraint: :func:`_resolve_d3_damping_class` reads the
    *suffixed* name to pick the damping form, so strip only after the class is
    resolved and only for the parameter lookup.
    """
    if not method:
        return method
    for alias, level in get_dispersion_aliases().items():
        if accept(level) and method.lower().endswith(alias):
            return method[: -(len(alias) + 1)]
    return method


def _build_atomic_result(
    input_model: AtomicInput,
    input_data: Dict[str, Any],
    energy: float,
    gradient,
    creator: str,
    version: str,
    dispersion_key: str,
    qcvkey: Optional[str],
) -> AtomicResult:
    """Wrap a native-API dispersion result in an AtomicResult, matching the
    non-periodic path's qcvars payload so downstream consumers don't drift."""
    import numpy as np

    driver = input_model.specification.driver
    return_result = energy if driver == "energy" else np.asarray(gradient).ravel().tolist()

    calcinfo: Dict[str, Any] = {
        "CURRENT ENERGY": energy,
        "DISPERSION CORRECTION ENERGY": energy,
    }
    if qcvkey:
        calcinfo[f"{qcvkey} DISPERSION CORRECTION ENERGY"] = energy
    if driver == "gradient":
        grad_list = np.asarray(gradient).ravel().tolist()
        calcinfo["CURRENT GRADIENT"] = grad_list
        calcinfo["DISPERSION CORRECTION GRADIENT"] = grad_list
        if qcvkey:
            calcinfo[f"{qcvkey} DISPERSION CORRECTION GRADIENT"] = grad_list

    ret_data: Dict[str, Any] = {
        "input_data": input_data,
        "molecule": input_model.molecule,
        "properties": {"return_energy": energy},
        "return_result": return_result,
        "provenance": Provenance(creator=creator, version=version, routine=f"{creator}.native-periodic"),
        "schema_name": "qcschema_atomic_result",
        "success": True,
        "extras": {"qcvars": calcinfo, dispersion_key: {"periodic": True}},
    }
    return AtomicResult(**ret_data)


class DFTD4Harness(ProgramHarness):
    """Calculation harness for the DFT-D4 dispersion correction."""

    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "dftd4",
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": True,
        "node_parallel": False,
        "managed_memory": False,
    }
    version_cache: Dict[str, str] = {}

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        """Check for the availability of the Python API of dftd4"""

        return which_import(
            "dftd4",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install a dftd4 version with enabled Python API"
            + " (e.g. conda install dftd4-python -c conda-forge)",
        )

    def get_version(self) -> str:
        """Return the currently used version of dftd4"""
        self.found(raise_error=True)

        which_prog = which_import("dftd4")
        if which_prog not in self.version_cache:
            import dftd4

            self.version_cache[which_prog] = safe_version(dftd4.__version__)

        return self.version_cache[which_prog]

    def compute(self, input_model: AtomicInput, config: TaskConfig) -> AtomicResult:
        """
        Actual interface to the dftd4 package. The compute function is just a thin
        wrapper around the native QCSchema interface of the dftd4 Python-API.

        Periodic mode: when ``keywords['cell']`` (3x3 Angstrom lattice) and
        ``keywords['pbc']`` (length-3 bools) are set and any pbc axis is True,
        the harness bypasses ``run_qcschema`` (which is non-periodic because
        QCSchema Molecules can't carry a cell) and calls ``dftd4.interface``
        directly with lattice + periodic. Non-periodic path is unchanged.
        """

        self.found(raise_error=True)

        import dftd4
        import numpy as np
        from dftd4.qcschema import run_qcschema

        # strip engine hint
        input_data = input_model.model_dump()
        method = input_model.specification.model.method
        if method.startswith("d4-"):
            method = method[3:]
            input_data["specification"]["model"]["method"] = method
        qcvkey = method.upper() if method is not None else None

        # Periodic branch — bypass qcschema (which is inherently non-periodic)
        lattice_bohr, pbc = _parse_periodic_keywords(input_model.specification.keywords)
        if lattice_bohr is not None:
            from dftd4.interface import DispersionModel, DampingParam
            numbers = np.asarray(input_model.molecule.atomic_numbers, dtype=int)
            positions_bohr = np.asarray(input_model.molecule.geometry, dtype=float).reshape(-1, 3)
            model = DispersionModel(numbers, positions_bohr, lattice=lattice_bohr, periodic=pbc)
            # dftd4's DampingParam takes method= plus keyword-value tweaks.
            # Parameter DB is keyed by the bare functional (see _strip_dispersion_level).
            method_bare = _strip_dispersion_level(
                method, lambda level: level in ("d4bjeeqatm", "d4bjeeqtwo")
            )
            damping_kw = {"method": method_bare} if method_bare else {}
            damping_kw.update(input_model.specification.keywords.get("params_tweaks", {}) or {})
            param = DampingParam(**damping_kw)
            res = model.get_dispersion(param, grad=(input_model.specification.driver == "gradient"))
            return _build_atomic_result(
                input_model, input_data,
                energy=float(res["energy"]),
                gradient=res.get("gradient"),
                creator="dftd4", version=dftd4.__version__,
                dispersion_key="dftd4", qcvkey=qcvkey,
            )

        # send `from_arrays` the dftd4 behavior of functional specification overrides explicit parameters specification
        # * differs from dftd4 harness behavior where parameters extend or override functional
        # * stash the resolved plan in extras or, if errored, leave it for the proper dftd4 api to reject
        param_tweaks = None if method else input_model.specification.keywords.get("params_tweaks", None)
        try:
            planinfo = from_arrays(
                verbose=1,
                name_hint=method,
                level_hint=input_model.specification.keywords.get("level_hint", None),
                param_tweaks=param_tweaks,
                dashcoeff_supplement=input_model.specification.keywords.get("dashcoeff_supplement", None),
            )
        except InputError:
            pass
        else:
            input_data["specification"]["extras"]["info"] = planinfo

        # strip dispersion level from method
        for alias, d4 in get_dispersion_aliases().items():
            if d4 in ["d4bjeeqatm", "d4bjeeqtwo"] and method.lower().endswith(alias):
                method = method[: -(len(alias) + 1)]
                input_data["specification"]["model"]["method"] = method

        # consolidate dispersion level aliases
        level_hint = input_model.specification.keywords.get("level_hint", None)
        if level_hint and get_dispersion_aliases()[level_hint.lower()] in ["d4bjeeqatm", "d4bjeeqtwo"]:
            level_hint = "d4"
            input_data["specification"]["keywords"]["level_hint"] = level_hint

        if parse_version(self.get_version()) < parse_version("4.1.0"):
            # dftd4 speaks qcsk.v1
            input_model_v1 = qcelemental.models.v2.AtomicInput(**input_data).convert_v(1)

            # Run the Harness
            output_v1 = run_qcschema(input_model_v1)

            # d4 qcschema interface stores error in Result model
            if not output_v1.success:
                return FailedOperation(input_data=input_data, error=output_v1.error.model_dump())

            # Unclear whether external_input_data should be input_model (user input in v2) or input_data
            #   (user input + processing above with tweaks and hints). Former seems cleaner (and works)
            #   but places "extras.info" differently in pre-/post-1.3.0 routes, so using latter.
            output = output_v1.convert_v(2, external_input_data=input_data)

        else:
            # dftd4 >4.1 speaks qcsk.v1 or qcsk.v2
            input_model_v2 = qcelemental.models.v2.AtomicInput(**input_data)

            # Run the Harness
            output_v2 = run_qcschema(input_model_v2)

            # d4 qcschema interface stores error in Result model
            # TODO
            if not output_v2.success:
                # return FailedOperation(input_data=input_data, error=output_v2.error.model_dump())
                return output_v2

            output = output_v2

        if "info" in output.input_data.specification.extras:
            # formerly output.extras["info"]
            qcvkey = output.input_data.specification.extras["info"]["fctldash"].upper()

        calcinfo = {}
        energy = output.properties.return_energy
        calcinfo["CURRENT ENERGY"] = energy
        calcinfo["DISPERSION CORRECTION ENERGY"] = energy
        if qcvkey:
            calcinfo[f"{qcvkey} DISPERSION CORRECTION ENERGY"] = energy

        if output.input_data.specification.driver == "gradient":
            gradient = output.return_result
            calcinfo["CURRENT GRADIENT"] = gradient
            calcinfo["DISPERSION CORRECTION GRADIENT"] = gradient
            if qcvkey:
                calcinfo[f"{qcvkey} DISPERSION CORRECTION GRADIENT"] = gradient

        if output.input_data.specification.keywords.get("pair_resolved", False):
            pw2 = output.extras["dftd4"]["additive pairwise energy"]
            pw3 = output.extras["dftd4"]["non-additive pairwise energy"]
            assert abs(pw2.sum() + pw3.sum() - energy) < 1.0e-8, f"{pw2.sum()} + {pw3.sum()} != {energy}"
            calcinfo["2-BODY DISPERSION CORRECTION ENERGY"] = pw2.sum()
            calcinfo["3-BODY DISPERSION CORRECTION ENERGY"] = pw3.sum()
            calcinfo["2-BODY PAIRWISE DISPERSION CORRECTION ANALYSIS"] = pw2
            calcinfo["3-BODY PAIRWISE DISPERSION CORRECTION ANALYSIS"] = pw3

        output.extras["qcvars"] = calcinfo

        return output


class SDFTD3Harness(ProgramHarness):
    """
    Calculation harness for the DFT-D3 dispersion correction.

    This implementation of DFT-D3 supports the several damping functions, which
    are selected via the *level_hint* keyword. Damping parameter can be specified
    via the *param_tweaks* dictionary. If no *param_tweaks* are provided the
    functional parameters are obtained from the internal database of the library.

    The following damping function are available via *level_hint*:

    - ``d3bj``:
      Rational damping function for DFT-D3. The original scheme was proposed by
      Becke and Johnson and implemented in a slightly adjusted form using only
      the C8/C6 ratio in the critical radius for DFT-D3.
      Requires at least three parameters: *s8*, *a1*, and *a2*.
      The parameters *s6*, *s9*, and *alpha6* can be adjusted as well.
    - ``d3zero``:
      Original DFT-D3 damping function, based on a variant proposed by Chai and Head-Gordon.
      Requires at least two parameters: *s8* and *sr6*.
      The parameters *s6*, *s9*, *sr8*, and *alpha6* can be adjusted as well.
    - ``d3mbj``:
      Modified version of the rational damping parameters. The functional form of the
      damping function is *unmodified* with respect to the original rational damping scheme.
      However, for a number of functionals new parameters were introduced.
      Requires at least three parameters: *s8*, *a1*, and *a2*.
      The parameters *s6*, *s9*, and *alpha6* can be adjusted as well.
    - ``d3mzero``:
      Modified zero damping function for DFT-D3. This scheme adds an additional offset
      parameter to the zero damping scheme of the original DFT-D3.
      Requires at least three parameters: *s8*, *sr6*, and *beta*.
      The parameters *s6*, *s9*, *sr8*, and *alpha6* can be adjusted as well.
    - ``d3op``:
      Optimized power version of the rational damping function for DFT-D3.
      The functional form of the damping function is modified by adding an additional
      zero-damping like power function.
      Requires at least four parameters: *s8*, *a1*, *a2*, and *beta*.
      The parameters *s6*, *s9*, and *alpha6* can be adjusted as well.

    All damping functions by default *include* the ATM three-body contributions,
    it must be explicitly disabled by setting the *s9* value to zero.
    """

    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "s-dftd3",
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": True,
        "node_parallel": False,
        "managed_memory": False,
    }
    version_cache: Dict[str, str] = {}

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        """Check for the availability of the Python API of dftd3"""

        return which_import(
            "dftd3",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install a dftd3 version with enabled Python API"
            + " (e.g. conda install dftd3-python -c conda-forge)",
        )

    def get_version(self) -> str:
        """Return the currently used version of dftd3"""
        self.found(raise_error=True)

        which_prog = which_import("dftd3")
        if which_prog not in self.version_cache:
            import dftd3

            self.version_cache[which_prog] = safe_version(dftd3.__version__)

        return self.version_cache[which_prog]

    def compute(self, input_model: AtomicInput, config: TaskConfig) -> AtomicResult:
        """
        Actual interface to the dftd3 package. The compute function is just a thin
        wrapper around the native QCSchema interface of the dftd3 Python-API.

        Periodic mode: when ``keywords['cell']`` (3x3 Angstrom lattice) and
        ``keywords['pbc']`` (length-3 bools) are set and any pbc axis is True,
        the harness bypasses ``run_qcschema`` and calls ``dftd3.interface``
        directly with lattice + periodic. Damping variant is picked from
        ``keywords['level_hint']`` (preferred) then a ``-d3*`` suffix on the
        method name (fallback: RationalDampingParam / D3BJ).
        """
        self.found(raise_error=True)
        if parse_version(self.get_version()) < parse_version("0.5.1"):
            raise ResourceError("QCEngine's dftd3 wrapper requires version 0.5.1 or greater.")

        import dftd3
        import numpy as np
        from dftd3.qcschema import run_qcschema

        # strip engine hint
        input_data = input_model.model_dump()
        method = input_model.specification.model.method
        if method.startswith("d3-"):
            method = method[3:]
            input_data["specification"]["model"]["method"] = method
        qcvkey = method.upper() if method is not None else None

        # Periodic branch — bypass qcschema (which is inherently non-periodic)
        lattice_bohr, pbc = _parse_periodic_keywords(input_model.specification.keywords)
        if lattice_bohr is not None:
            from dftd3.interface import DispersionModel
            numbers = np.asarray(input_model.molecule.atomic_numbers, dtype=int)
            positions_bohr = np.asarray(input_model.molecule.geometry, dtype=float).reshape(-1, 3)
            model = DispersionModel(numbers, positions_bohr, lattice=lattice_bohr, periodic=pbc)
            # Resolve the damping form from the SUFFIXED name first, then strip the
            # level for the parameter lookup (see _strip_dispersion_level).
            damping_cls = _resolve_d3_damping_class(
                input_model.specification.keywords.get("level_hint"), method,
            )
            method_bare = _strip_dispersion_level(
                method, lambda level: level.startswith("d3")
            )
            damping_kw = {"method": method_bare} if method_bare else {}
            damping_kw.update(input_model.specification.keywords.get("params_tweaks", {}) or {})
            param = damping_cls(**damping_kw)
            res = model.get_dispersion(param, grad=(input_model.specification.driver == "gradient"))
            return _build_atomic_result(
                input_model, input_data,
                energy=float(res["energy"]),
                gradient=res.get("gradient"),
                creator="dftd3", version=dftd3.__version__,
                dispersion_key="dftd3", qcvkey=qcvkey,
            )

        # send `from_arrays` the s-dftd3 behavior of functional specification overrides explicit parameters specification
        # * differs from classic-dftd3 harness behavior where parameters extend or override functional
        # * stash the resolved plan in extras or, if errored, leave it for the proper dftd3 api to reject
        param_tweaks = None if method else input_model.specification.keywords.get("params_tweaks", None)
        try:
            planinfo = from_arrays(
                verbose=1,
                name_hint=method,
                level_hint=input_model.specification.keywords.get("level_hint", None),
                param_tweaks=param_tweaks,
                dashcoeff_supplement=input_model.specification.keywords.get("dashcoeff_supplement", None),
            )
        except InputError:
            pass
        else:
            input_data["specification"]["extras"]["info"] = planinfo

        # strip dispersion level from method
        for alias, d3 in get_dispersion_aliases().items():
            if d3.startswith("d3") and method.lower().endswith(alias):
                method = method[: -(len(alias) + 1)]
                input_data["specification"]["model"]["method"] = method

        # consolidate dispersion level aliases
        if input_data["specification"]["keywords"].pop("apply_qcengine_aliases", False):
            level_hint = input_model.specification.keywords.get("level_hint", None)
            if level_hint:
                level_hint = get_dispersion_aliases()[level_hint.lower()]
                if level_hint.endswith("atm"):
                    level_hint = level_hint[:-3]
                    # re-route through params_tweaks needed for >=1.3.0 where atm=False became default
                    input_data["specification"]["keywords"]["params_tweaks"] = {**planinfo["dashparams"]}
                if level_hint.endswith("2b"):
                    level_hint = level_hint[:-2]
                    input_data["specification"]["keywords"]["params_tweaks"] = {**planinfo["dashparams"], "s9": 0.0}
                    input_data["specification"]["extras"]["info"]["dashparams"]["s9"] = 0.0
                input_data["specification"]["keywords"]["level_hint"] = level_hint

        if parse_version(self.get_version()) < parse_version("1.3.0"):
            # s-dftd3 speaks qcsk.v1
            input_model_v1 = qcelemental.models.v2.AtomicInput(**input_data).convert_v(1)

            # Run the Harness
            output_v1 = run_qcschema(input_model_v1)

            # d3 qcschema interface stores error in Result model
            if not output_v1.success:
                return FailedOperation(input_data=input_data, error=output_v1.error.model_dump())

            # Unclear whether external_input_data should be input_model (user input in v2) or input_data
            #   (user input + processing above with tweaks and hints). Former seems cleaner (and works)
            #   but places "extras.info" differently in pre-/post-1.3.0 routes, so using latter.
            output = output_v1.convert_v(2, external_input_data=input_data)

        else:
            # s-dftd3 >1.3.0 speaks qcsk.v1 or qcsk.v2
            input_model_v2 = qcelemental.models.v2.AtomicInput(**input_data)

            # Run the Harness
            output_v2 = run_qcschema(input_model_v2)

            # d3 qcschema interface stores error in Result model
            # TODO
            if not output_v2.success:
                # return FailedOperation(input_data=input_data, error=output_v2.error.model_dump())
                return output_v2

            output = output_v2

        if "info" in output.input_data.specification.extras:
            # formerly output.extras["info"]
            qcvkey = output.input_data.specification.extras["info"]["fctldash"].upper()

        calcinfo = {}
        energy = output.properties.return_energy
        calcinfo["CURRENT ENERGY"] = energy
        calcinfo["DISPERSION CORRECTION ENERGY"] = energy
        if qcvkey:
            calcinfo[f"{qcvkey} DISPERSION CORRECTION ENERGY"] = energy

        if output.input_data.specification.driver == "gradient":
            gradient = output.return_result
            calcinfo["CURRENT GRADIENT"] = gradient
            calcinfo["DISPERSION CORRECTION GRADIENT"] = gradient
            if qcvkey:
                calcinfo[f"{qcvkey} DISPERSION CORRECTION GRADIENT"] = gradient

        if output.input_data.specification.keywords.get("pair_resolved", False):
            pw2 = output.extras["dftd3"]["additive pairwise energy"]
            pw3 = output.extras["dftd3"]["non-additive pairwise energy"]
            assert abs(pw2.sum() + pw3.sum() - energy) < 1.0e-8, f"{pw2.sum()} + {pw3.sum()} != {energy}"
            calcinfo["2-BODY DISPERSION CORRECTION ENERGY"] = pw2.sum()
            calcinfo["3-BODY DISPERSION CORRECTION ENERGY"] = pw3.sum()
            calcinfo["2-BODY PAIRWISE DISPERSION CORRECTION ANALYSIS"] = pw2
            calcinfo["3-BODY PAIRWISE DISPERSION CORRECTION ANALYSIS"] = pw3

        output.extras["qcvars"] = calcinfo

        return output
