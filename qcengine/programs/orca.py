"""Calls the ORCA executable.

This harness intentionally covers a *subset* of ORCA's capabilities, aimed at
fast DFT/HF energy, gradient, and Hessian evaluations (e.g. driven through
QCFractal):

* ``driver=energy``   -> plain single point
* ``driver=gradient`` -> ``! EnGrad`` (analytic gradients, RIJCOSX-compatible)
* ``driver=hessian``  -> ``! Freq``   (analytic Hessians where ORCA supports
  them, e.g. HF/DFT; ORCA falls back to numerical otherwise)

Everything else in ORCA's very large keyword surface is reachable through two
escape hatches in ``AtomicInput.specification.keywords``:

* ``simple_input``: a string or list of strings appended verbatim to the
  ``!`` simple-input line (e.g. ``"RIJCOSX D3BJ TightSCF defgrid3"``).
* ``blocks``: a string or list of strings, each a raw ORCA ``%`` block emitted
  verbatim (e.g. ``"%scf maxiter 150 end"``). If a user block contains
  ``%pal`` or ``%maxcore``, the harness does not emit its own.

Notes:

* Dispersion is a separate ORCA keyword: use ``method="b3lyp"`` plus
  ``simple_input="D3BJ"``. Psi4-style compound method strings such as
  ``b3lyp-d3bj`` are not translated and will be rejected by ORCA.
* Multiplicity > 1 relies on ORCA's automatic UHF/UKS selection; force
  ``UKS``/``UHF``/``ROHF`` via ``simple_input`` if needed.
* Parallelism must go through the harness (``TaskConfig.ncores``) or a user
  ``%pal`` block. A ``!PALn`` keyword inside ``simple_input`` is *not*
  detected and would clash with the harness-generated ``%pal`` block.
* Custom basis/ECP definitions can be supplied via a ``%basis`` block.
"""

import re
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from qcelemental.models.v2 import AtomicResult, BasisSet, Provenance
from qcelemental.util import safe_version, which

from ..exceptions import ConvergenceError, InputError, RandomError, ResourceError, UnknownError
from ..util import execute, popen
from .model import ProgramHarness
from .util import error_stamp

if TYPE_CHECKING:
    from qcelemental.models.v2 import AtomicInput

    from ..config import TaskConfig


#: stem for all files written into the scratch directory
_INPUT_STEM = "input"

#: ORCA treats %maxcore as advisory and routinely overshoots it
#: (especially RIJCOSX and Freq), so leave headroom.
_MAXCORE_SAFETY = 0.75


def parse_engrad(engrad: str) -> Tuple[int, float, np.ndarray]:
    """Parse an ORCA ``.engrad`` file.

    The file consists of ``#`` comment lines and, in order: the number of
    atoms, the total energy in Eh, the 3N Cartesian gradient entries in
    Eh/bohr (one per line), and finally the atomic numbers and coordinates
    (ignored here).

    Returns
    -------
    (natoms, energy, gradient)
        ``gradient`` has shape ``(natoms, 3)``.
    """
    values = [line.strip() for line in engrad.splitlines() if line.strip() and not line.startswith("#")]
    try:
        natoms = int(values[0])
        energy = float(values[1])
        gradient = np.array(values[2 : 2 + 3 * natoms], dtype=float)
    except (IndexError, ValueError) as e:
        raise UnknownError(f"ORCA .engrad file could not be parsed: {e}")

    if gradient.size != 3 * natoms:
        raise UnknownError(
            f"ORCA .engrad file is truncated: expected {3 * natoms} gradient entries, found {gradient.size}"
        )

    return natoms, energy, gradient.reshape(-1, 3)


def parse_hessian(hess: str) -> np.ndarray:
    """Parse the ``$hessian`` block of an ORCA ``.hess`` file.

    The block starts with the dimension (3N) on its own line, followed by
    column-blocked sections: a header row of column indices (up to five
    columns per section) and then 3N rows of ``rowindex value ...``.
    Values are in Eh/bohr^2.

    Returns
    -------
    numpy.ndarray
        The full ``(3N, 3N)`` Cartesian Hessian.
    """
    lines = hess.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "$hessian")
    except StopIteration:
        raise UnknownError("ORCA .hess file has no $hessian block")

    try:
        dim = int(lines[start + 1].strip())
    except (IndexError, ValueError):
        raise UnknownError("ORCA .hess file has no dimension line after $hessian")

    hessian = np.full((dim, dim), np.nan)
    idx = start + 2
    try:
        while np.isnan(hessian).any():
            # Header row of column indices for this section
            cols = [int(c) for c in lines[idx].split()]
            for row_line in lines[idx + 1 : idx + 1 + dim]:
                tokens = row_line.split()
                row = int(tokens[0])
                hessian[row, cols] = [float(v) for v in tokens[1 : 1 + len(cols)]]
            idx += 1 + dim
    except (IndexError, ValueError) as e:
        raise UnknownError(f"ORCA .hess $hessian block could not be parsed: {e}")

    return hessian


def harvest_stdout(stdout: str) -> Dict[str, Any]:
    """Collect properties from ORCA stdout on a best-effort basis.

    Only ``return_energy`` is required by the caller; any other quantity that
    cannot be found is simply omitted.
    """
    props: Dict[str, Any] = {}

    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", stdout)
    if matches:
        # Take the last one; some modules print intermediate single point energies
        props["return_energy"] = float(matches[-1])

    optional = {
        "scf_total_energy": r"Total Energy\s+:\s+(-?\d+\.\d+)\s+Eh",
        "nuclear_repulsion_energy": r"Nuclear Repulsion\s+:\s+(-?\d+\.\d+)\s+Eh",
    }
    for key, pattern in optional.items():
        matches = re.findall(pattern, stdout)
        if matches:
            props[key] = float(matches[-1])

    mobj = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", stdout)
    if mobj:
        props["scf_iterations"] = int(mobj.group(1))

    mobj = re.search(r"# of contracted basis functions\s+\.+\s*(\d+)", stdout)
    if mobj:
        props["calcinfo_nbasis"] = int(mobj.group(1))

    return props


def check_orca_errors(stdin: str, stdout: str, stderr: str) -> None:
    """Map known ORCA failure signatures to typed exceptions.

    Ordered most-specific first; the generic ``error termination`` catch-all
    comes last. Must be called even when the process exit code is 0 -- ORCA
    does not always exit non-zero on failure.
    """
    stamp = error_stamp(stdin, stdout, stderr)

    if "UNRECOGNIZED OR DUPLICATED KEYWORD" in stdout or "INPUT ERROR" in stdout:
        raise InputError(stamp)
    if re.search(r"(?i)error.{0,20}multiplicity", stdout):
        raise InputError(stamp)
    # NOTE: no signature for a bad basis name: ORCA reports it through the
    # generic INPUT ERROR banner. A looser "basis ... not available" match
    # false-positives on the normal SCF summary (e.g. "Auxiliary Coulomb
    # fitting basis ... NOT available").

    if "SCF NOT CONVERGED AFTER" in stdout or "This wavefunction IS NOT FULLY CONVERGED!" in stdout:
        raise ConvergenceError(stamp)

    for text in (stdout, stderr):
        if (
            "mpirun detected that one or more processes exited" in text
            or "ORCA_PROCESS_ABORTED" in text
            or "Signal: Segmentation fault" in text
        ):
            raise RandomError(stamp)

    if "ORCA finished by error termination" in stdout:
        raise UnknownError(stamp)


class OrcaHarness(ProgramHarness):
    """Harness for the ORCA electronic structure program (energy/gradient/Hessian subset)."""

    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "ORCA",
        "scratch": True,
        "thread_safe": False,
        "thread_parallel": False,
        # ORCA parallelizes via its own MPI startup controlled by %pal;
        # the harness itself always launches a single process.
        "node_parallel": False,
        "managed_memory": True,
    }

    version_cache: ClassVar[Dict[str, str]] = {}
    # exe path -> is it really the ORCA quantum-chemistry program?
    _identity_cache: ClassVar[Dict[str, bool]] = {}

    _NOT_ORCA_MSG = (
        "Please install ORCA and make the `orca` binary available on PATH. "
        "See https://www.faccts.de/orca/ or https://orcaforum.kofo.mpg.de"
    )

    @classmethod
    def _parse_version(cls, exe: str) -> Optional[str]:
        """Return the ORCA version reported by ``exe``, or None if it isn't ORCA.

        Many Linux distributions ship the GNOME screen reader as ``orca`` in
        /usr/bin, so a bare ``which("orca")`` hit says nothing about whether the
        quantum-chemistry program is present.
        """
        try:
            with popen([exe, "--version"]) as exc:
                exc["proc"].wait(timeout=30)
            mobj = re.search(r"Program Version\s+([\d.]+)", exc["stdout"])
        except Exception:
            return None
        return mobj.group(1) if mobj else None

    @classmethod
    def _is_orca(cls, exe: str) -> bool:
        if exe not in cls._identity_cache:
            cls._identity_cache[exe] = cls._parse_version(exe) is not None
        return cls._identity_cache[exe]

    @classmethod
    def found(cls, raise_error: bool = False) -> bool:
        """True only when a *real* ORCA is on PATH.

        Validating here (rather than letting :meth:`get_version` raise) matters for
        callers that enumerate harnesses: QCFractal's compute manager builds its
        program list with ``get_version()`` over every *available* program, so a
        harness that reports found-but-unusable aborts manager startup entirely
        instead of simply being absent.
        """
        exe = which("orca", return_bool=False, raise_error=False)
        if exe is not None and cls._is_orca(exe):
            return True
        if raise_error:
            which("orca", raise_error=True, raise_msg=cls._NOT_ORCA_MSG)
            # `orca` resolves but is not the QC program
            raise ResourceError(
                f"`orca` on PATH ({exe}) is not the ORCA quantum-chemistry program "
                "(no 'Program Version' in `orca --version`); it is most likely the "
                "GNOME screen reader. " + cls._NOT_ORCA_MSG
            )
        return False

    def get_version(self) -> str:
        self.found(raise_error=True)

        which_prog = which("orca")
        if which_prog not in self.version_cache:
            version = self._parse_version(which_prog)
            if version is None:
                raise UnknownError(
                    "Could not parse an ORCA version from `orca --version`. "
                    "Note: on some Linux systems `orca` on PATH is the GNOME screen reader, "
                    "not the ORCA quantum chemistry program."
                )
            self.version_cache[which_prog] = safe_version(version)

        return self.version_cache[which_prog]

    def compute(self, input_data: "AtomicInput", config: "TaskConfig") -> AtomicResult:
        self.found(raise_error=True)

        job_inputs = self.build_input(input_data, config)
        success, dexe = self.execute(job_inputs)

        stdin = job_inputs["infiles"][f"{_INPUT_STEM}.inp"]
        # ORCA can exit 0 on failure, so always scan for error signatures
        check_orca_errors(stdin, dexe["stdout"], dexe["stderr"])

        if not success or "ORCA TERMINATED NORMALLY" not in dexe["stdout"]:
            raise UnknownError(error_stamp(stdin, dexe["stdout"], dexe["stderr"]))

        dexe["outfiles"]["stdout"] = dexe["stdout"]
        dexe["outfiles"]["stderr"] = dexe["stderr"]
        return self.parse_output(dexe["outfiles"], input_data)

    def build_input(
        self, input_model: "AtomicInput", config: "TaskConfig", template: Optional[str] = None
    ) -> Dict[str, Any]:
        spec = input_model.specification
        model = spec.model

        if isinstance(model.basis, BasisSet):
            raise InputError("QCSchema BasisSet for model.basis not implemented. Use string basis name.")

        if spec.driver not in ("energy", "gradient", "hessian"):
            raise InputError(f"Driver {spec.driver} not implemented for ORCA (energy, gradient, hessian only).")

        keywords = dict(spec.keywords)
        unknown = set(keywords) - {"simple_input", "blocks"}
        if unknown:
            raise InputError(
                f"Unrecognized ORCA keywords {sorted(unknown)}. "
                "This harness only accepts 'simple_input' (extra `!` keywords) "
                "and 'blocks' (raw `%` blocks)."
            )

        simple_input: Union[str, List[str]] = keywords.get("simple_input", "")
        if not isinstance(simple_input, str):
            simple_input = " ".join(simple_input)

        blocks: Union[str, List[str]] = keywords.get("blocks", [])
        if isinstance(blocks, str):
            blocks = [blocks]
        blocks_lc = "\n".join(blocks).lower()

        driver_keyword = {"energy": "", "gradient": "EnGrad", "hessian": "Freq"}[spec.driver]

        tokens = ["!", model.method]
        if model.basis:
            tokens.append(model.basis)
        if driver_keyword:
            tokens.append(driver_keyword)
        if simple_input:
            tokens.append(simple_input)

        lines = [" ".join(tokens)]

        if config.ncores > 1 and "%pal" not in blocks_lc:
            lines.append(f"%pal nprocs {config.ncores} end")
        if "%maxcore" not in blocks_lc:
            mb_per_core = max(256, int(config.memory * 1024 * _MAXCORE_SAFETY / config.ncores))
            lines.append(f"%maxcore {mb_per_core}")

        lines.extend(blocks)

        # to_string emits its own `! Bohrs` unit line ahead of the `*xyz` block.
        # ORCA merges all `!` lines in the file, so it is kept verbatim on purpose.
        lines.append(input_model.molecule.to_string(dtype="orca", units="Bohr"))

        outfiles = {
            "energy": [],
            "gradient": [f"{_INPUT_STEM}.engrad"],
            # Analytic Freq usually writes no .engrad; collect it opportunistically
            "hessian": [f"{_INPUT_STEM}.hess", f"{_INPUT_STEM}.engrad"],
        }[spec.driver]

        return {
            # ORCA locates its helper binaries (orca_scf, ...) and its MPI startup
            # relative to the invoked path, so the absolute path is mandatory
            "command": [which("orca"), f"{_INPUT_STEM}.inp"],
            "infiles": {f"{_INPUT_STEM}.inp": "\n".join(lines) + "\n"},
            "outfiles": outfiles,
            "scratch_directory": config.scratch_directory,
            "scratch_messy": config.scratch_messy,
        }

    def execute(
        self, inputs: Dict[str, Any], *, extra_outfiles=None, extra_commands=None, scratch_name=None, timeout=None
    ) -> Tuple[bool, Dict]:
        success, dexe = execute(
            inputs["command"],
            inputs["infiles"],
            inputs["outfiles"],
            scratch_directory=inputs["scratch_directory"],
            scratch_messy=inputs["scratch_messy"],
            timeout=timeout,
        )
        return success, dexe

    def parse_output(self, outfiles: Dict[str, str], input_model: "AtomicInput") -> AtomicResult:
        stdout = outfiles.pop("stdout")
        stderr = outfiles.pop("stderr", "")

        props = harvest_stdout(stdout)
        if "return_energy" not in props:
            raise UnknownError(error_stamp("", stdout, stderr))

        natom = len(input_model.molecule.symbols)
        # Required by the AtomicResultProperties validator whenever
        # return_gradient/return_hessian are present; harmless otherwise.
        props["calcinfo_natom"] = natom

        driver = input_model.specification.driver
        return_result: Any = props["return_energy"]

        if driver == "gradient":
            engrad = outfiles.get(f"{_INPUT_STEM}.engrad")
            if engrad is None:
                raise UnknownError(error_stamp("", stdout, stderr) + "\nNo .engrad file was produced.")
            grad_natom, energy, gradient = parse_engrad(engrad)
            if grad_natom != natom:
                raise UnknownError(f".engrad natoms ({grad_natom}) does not match the molecule ({natom})")
            # .engrad carries the energy at full precision
            props["return_energy"] = energy
            props["return_gradient"] = gradient
            return_result = gradient

        elif driver == "hessian":
            hess = outfiles.get(f"{_INPUT_STEM}.hess")
            if hess is None:
                raise UnknownError(error_stamp("", stdout, stderr) + "\nNo .hess file was produced.")
            hessian = parse_hessian(hess)
            if hessian.shape != (3 * natom, 3 * natom):
                raise UnknownError(f".hess dimension {hessian.shape} does not match the molecule (3*{natom})")
            props["return_hessian"] = hessian
            engrad = outfiles.get(f"{_INPUT_STEM}.engrad")
            if engrad is not None:
                grad_natom, _, gradient = parse_engrad(engrad)
                if grad_natom == natom:
                    props["return_gradient"] = gradient
            return_result = hessian

        keep = (f"{_INPUT_STEM}.engrad", f"{_INPUT_STEM}.hess")
        extras_outfiles = {k: v for k, v in outfiles.items() if k in keep and v is not None}

        return AtomicResult(
            input_data=input_model,
            molecule=input_model.molecule,
            properties=props,
            return_result=return_result,
            success=True,
            stdout=stdout,
            provenance=Provenance(creator="ORCA", version=self.get_version(), routine="orca"),
            extras={"outfiles": extras_outfiles},
        )
