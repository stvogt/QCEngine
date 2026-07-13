"""Calls the Gaussian 16 executable.

This harness intentionally covers a *subset* of Gaussian's capabilities,
aimed at DFT/HF energy, gradient, and Hessian evaluations (e.g. driven
through QCFractal):

* ``driver=energy``   -> plain single point
* ``driver=gradient`` -> ``Force`` (analytic gradients)
* ``driver=hessian``  -> ``Freq``  (analytic Hessians where Gaussian supports
  them; Gaussian falls back to numerical otherwise)

Everything else in Gaussian's large keyword surface is reachable through two
escape hatches in ``AtomicInput.specification.keywords``:

* ``route_input``: a string or list of strings appended verbatim to the
  ``#P`` route line (e.g. ``"EmpiricalDispersion=GD3BJ SCF=Tight"``).
* ``link0``: a list of raw Link 0 ``%`` lines emitted verbatim (e.g.
  ``"%rwf=/scratch/big.rwf"``). If a user line contains ``%nproc`` or
  ``%mem``, the harness does not emit its own. ``%chk`` is managed by the
  harness (result parsing reads the formatted checkpoint) and is rejected
  in user ``link0``.

Notes:

* Results are parsed from the formatted checkpoint (``formchk``), which is
  full precision and method-independent; the run log (captured as stdout)
  is used for success/error detection only.
* Dispersion is a route keyword: use ``method="b3lyp"`` plus
  ``route_input="EmpiricalDispersion=GD3BJ"``. Psi4-style compound method
  strings such as ``b3lyp-d3bj`` are not translated and will be rejected
  by Gaussian.
* The harness adds ``Force``/``Freq`` itself from the driver — do not
  repeat them in ``route_input``. Compound routes (e.g. ``opt``) are not
  prevented, but only the *final* job step's checkpoint is parsed.
* Multiplicity > 1 relies on Gaussian's automatic UHF/UKS selection; force
  a reference via the method name (e.g. ``ROB3LYP``) if needed.
* def2 basis names are normalized to Gaussian's hyphen-less spelling
  (``def2-SVP`` -> ``def2SVP``); all other basis names pass verbatim.
* Ghost atoms are emitted with Gaussian's ``El-Bq`` syntax and keep their
  basis functions (counterpoise-ready). ``Symmetry=None`` is added to the
  route automatically when ghosts are present — G16's symmetry machinery
  crashes on them ("Symmetry image not found in LdSEqv").
"""

import re
import subprocess
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Tuple, Union

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

#: %Mem is a working-memory target; the g16 process overshoots it with
#: integral buffers and link overhead, so leave headroom.
_MEM_SAFETY = 0.75
_MEM_FLOOR_MB = 512


def normalize_basis(basis: str) -> str:
    """Translate a basis name to Gaussian's spelling.

    The only rule is the def2 family, which Gaussian spells without the
    hyphen (``def2-SVP`` -> ``def2SVP``). Everything else passes verbatim;
    an unknown basis surfaces as a Gaussian input error.
    """
    if basis.lower().startswith("def2-"):
        return basis.replace("-", "", 1)
    return basis


def format_molecule(molecule) -> str:
    """Build the Gaussian charge/multiplicity + Cartesian block (Angstrom).

    Ghost atoms (``molecule.real[i]`` False) use Gaussian's ``El-Bq``
    syntax: no electrons or charge, but basis functions are kept.
    """
    from qcelemental import constants

    lines = [f"{int(molecule.molecular_charge)} {molecule.molecular_multiplicity}"]
    geom = np.asarray(molecule.geometry).reshape(-1, 3) * constants.bohr2angstroms
    for symbol, real, (x, y, z) in zip(molecule.symbols, molecule.real, geom):
        label = symbol if real else f"{symbol}-Bq"
        lines.append(f"{label:<8s} {x:>20.12f} {y:>20.12f} {z:>20.12f}")
    return "\n".join(lines)


def parse_fchk_scalar(fchk: str, key: str) -> Union[int, float]:
    """Parse a scalar field from a formatted checkpoint file.

    fchk scalar lines have a 40-column name field, a type character and the
    value, e.g. ``Total Energy   R     -3.875994220043694E+01``. The regex
    anchors at column 0 so a key can never match inside a longer field name.
    """
    mobj = re.search(
        rf"^{re.escape(key)}\s+([IR])\s+(-?\d+\.?\d*E?[+-]?\d*)\s*$",
        fchk,
        re.MULTILINE,
    )
    if mobj is None:
        raise UnknownError(f"Gaussian fchk field '{key}' not found")
    return int(mobj.group(2)) if mobj.group(1) == "I" else float(mobj.group(2))


def parse_fchk_array(fchk: str, key: str) -> np.ndarray:
    """Parse a real array block from a formatted checkpoint file.

    Block format: ``<key>   R   N=  <count>`` followed by the values, five
    per line in ``%16.8E`` fields.
    """
    mobj = re.search(rf"^{re.escape(key)}\s+R\s+N=\s+(\d+)\s*$", fchk, re.MULTILINE)
    if mobj is None:
        raise UnknownError(f"Gaussian fchk array '{key}' not found")

    n = int(mobj.group(1))
    tokens: List[str] = []
    for line in fchk[mobj.end():].splitlines():
        if not line.strip():
            continue
        tokens.extend(line.split())
        if len(tokens) >= n:
            break
    if len(tokens) < n:
        raise UnknownError(f"Gaussian fchk array '{key}' is truncated: expected {n} values, found {len(tokens)}")

    try:
        return np.array(tokens[:n], dtype=float)
    except ValueError as e:
        raise UnknownError(f"Gaussian fchk array '{key}' could not be parsed: {e}")


def lt_to_square(packed: np.ndarray, dim: int) -> np.ndarray:
    """Unpack a row-major lower-triangle array (fchk packing) to a full
    symmetric ``(dim, dim)`` matrix."""
    if packed.size != dim * (dim + 1) // 2:
        raise UnknownError(
            f"Lower-triangle size {packed.size} does not match dimension {dim} "
            f"(expected {dim * (dim + 1) // 2})"
        )
    full = np.zeros((dim, dim))
    i, j = np.tril_indices(dim)
    full[i, j] = packed
    full[j, i] = packed
    return full


def harvest_fchk(fchk: str) -> Dict[str, Any]:
    """Collect properties from the formatted checkpoint.

    Only ``return_energy`` (fchk ``Total Energy``) is required; any other
    quantity that cannot be found is simply omitted.
    """
    props: Dict[str, Any] = {"return_energy": parse_fchk_scalar(fchk, "Total Energy")}

    for prop_key, fchk_key in (
        ("scf_total_energy", "SCF Energy"),
        ("calcinfo_nbasis", "Number of basis functions"),
    ):
        try:
            props[prop_key] = parse_fchk_scalar(fchk, fchk_key)
        except UnknownError:
            pass

    return props


def check_gaussian_errors(stdin: str, stdout: str, stderr: str) -> None:
    """Map known Gaussian failure signatures to typed exceptions.

    Ordered most-specific first; the generic ``Error termination`` catch-all
    comes last. Must be called even when the process exit code is 0.
    Signatures marked UNVERIFIED are best-effort from Gaussian conventions
    and get tightened against real G16 failure logs.
    """
    stamp = error_stamp(stdin, stdout, stderr)

    # VERIFIED style (l301 charge/multiplicity check)
    if "The combination of multiplicity" in stdout:
        raise InputError(stamp)
    # UNVERIFIED: route-card parse errors (l1 / input scanner)
    if re.search(r"Error termination via Lnk1e in \S*l1\.exe", stdout) or (
        "A syntax error was detected in the input line" in stdout
    ):
        raise InputError(stamp)
    # UNVERIFIED: basis / molecule specification errors (l301)
    if (
        re.search(r"Error termination via Lnk1e in \S*l301\.exe", stdout)
        or "EOF while reading basis" in stdout
        or "Unrecognized atomic symbol" in stdout
    ):
        raise InputError(stamp)

    # VERIFIED (G16 RevB.01): for single points an unconverged SCF is only a
    # WARNING -- Gaussian prints ">>>>>>>>>> Convergence criterion not met.",
    # then "SCF Done" with the unconverged energy and terminates NORMALLY.
    # Without this check a "successful" record would carry a garbage energy.
    if "Convergence criterion not met" in stdout:
        raise ConvergenceError(stamp)
    # UNVERIFIED: hard SCF failure (l502 error termination path)
    if "Convergence failure -- run terminated" in stdout:
        raise ConvergenceError(stamp)

    # UNVERIFIED: memory allocation failures
    if re.search(r"galloc:\s+could not allocate memory", stdout) or "Not enough memory" in stdout:
        raise ResourceError(stamp)

    # UNVERIFIED: disk/IO trouble and crashes -- retryable
    for text in (stdout, stderr):
        if "Erroneous write" in text or "Write error in NtrExt1" in text:
            raise RandomError(stamp)
    if "Segmentation fault" in stderr:
        raise RandomError(stamp)

    # VERIFIED (real logs): "Error termination via Lnk1e in <link>.exe at <date>."
    if "Error termination" in stdout:
        raise UnknownError(stamp)


class GaussianHarness(ProgramHarness):
    """Harness for Gaussian 16 (energy/gradient/Hessian subset, fchk-based parsing)."""

    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "Gaussian",
        "scratch": True,
        "thread_safe": False,
        # Gaussian parallelizes with shared-memory threads via %NProcShared
        "thread_parallel": True,
        "node_parallel": False,
        "managed_memory": True,
    }

    version_cache: ClassVar[Dict[str, str]] = {}

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        return which(
            "g16",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install Gaussian 16 and make the `g16` binary available on PATH "
            "(e.g. `module load Gaussian/G16`). Note Gaussian is licensed software.",
        )

    def get_version(self) -> str:
        self.found(raise_error=True)

        which_prog = which("g16")
        if which_prog not in self.version_cache:
            # g16 has no version flag; with empty stdin it prints its banner
            # (including the revision) and error-terminates within a second.
            # The shell string must be a one-element list: popen() list()s
            # its argument, so a bare string would be split into characters.
            with popen([f'"{which_prog}" < /dev/null'], popen_kwargs={"shell": True}) as exc:
                try:
                    exc["proc"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass  # popen terminates the process on context exit
            # Both banner forms: "Gaussian 16, Revision B.01," and
            # "Gaussian 16:  ES64L-G16RevB.01"
            mobj = re.search(r"Gaussian\s+(\d+)[,:].*?Rev(?:ision)?\s*([A-Z])\.?(\d+)", exc["stdout"], re.DOTALL)
            if mobj is None:
                raise UnknownError(
                    "Could not parse a Gaussian version from the g16 banner. "
                    f"Banner head: {exc['stdout'][:500]!r}"
                )
            major, rev_letter, rev_minor = mobj.groups()
            # Letter -> ordinal keeps versions PEP440-comparable: RevB.01 ->
            # 16.2.1 (a naive "16.b.01" would parse as pre-release 16b1 < 16.0)
            letter_ord = ord(rev_letter.lower()) - ord("a") + 1
            self.version_cache[which_prog] = safe_version(f"{major}.{letter_ord}.{int(rev_minor)}")

        return self.version_cache[which_prog]

    def compute(self, input_data: "AtomicInput", config: "TaskConfig") -> AtomicResult:
        self.found(raise_error=True)

        job_inputs = self.build_input(input_data, config)
        success, dexe = self.execute(job_inputs)

        stdin = job_inputs["infiles"][f"{_INPUT_STEM}.gjf"]
        # Gaussian exit codes are unreliable across links; always scan
        check_gaussian_errors(stdin, dexe["stdout"], dexe["stderr"])

        if not success or "Normal termination of Gaussian" not in dexe["stdout"]:
            raise UnknownError(error_stamp(stdin, dexe["stdout"], dexe["stderr"]))

        dexe["outfiles"]["stdout"] = dexe["stdout"]
        dexe["outfiles"]["stderr"] = dexe["stderr"]
        return self.parse_output(dexe["outfiles"], input_data)

    def build_input(
        self, input_model: "AtomicInput", config: "TaskConfig", template: Any = None
    ) -> Dict[str, Any]:
        spec = input_model.specification
        model = spec.model

        if isinstance(model.basis, BasisSet):
            raise InputError("QCSchema BasisSet for model.basis not implemented. Use string basis name.")

        if spec.driver not in ("energy", "gradient", "hessian"):
            raise InputError(f"Driver {spec.driver} not implemented for Gaussian (energy, gradient, hessian only).")

        keywords = dict(spec.keywords)
        unknown = set(keywords) - {"route_input", "link0"}
        if unknown:
            raise InputError(
                f"Unrecognized Gaussian keywords {sorted(unknown)}. "
                "This harness only accepts 'route_input' (extra route keywords) "
                "and 'link0' (raw `%` lines)."
            )

        route_input: Union[str, List[str]] = keywords.get("route_input", "")
        if not isinstance(route_input, str):
            route_input = " ".join(route_input)

        link0: Union[str, List[str]] = keywords.get("link0", [])
        if isinstance(link0, str):
            link0 = [link0]
        link0 = [line.strip() for line in link0]
        for line in link0:
            if not line.startswith("%"):
                raise InputError(f"link0 lines must start with '%': {line!r}")
        link0_lc = "\n".join(link0).lower()

        if "%chk" in link0_lc:
            raise InputError(
                "The harness manages %chk itself (result parsing reads the formatted "
                "checkpoint); remove %chk from link0."
            )

        lines = [f"%chk={_INPUT_STEM}.chk"]
        if config.ncores > 1 and "%nproc" not in link0_lc:
            lines.append(f"%NProcShared={config.ncores}")
        if "%mem" not in link0_lc:
            # %Mem is the TOTAL working memory (not per-core like ORCA's %maxcore)
            mem_mb = max(_MEM_FLOOR_MB, int(config.memory * 1024 * _MEM_SAFETY))
            lines.append(f"%Mem={mem_mb}MB")
        lines.extend(link0)

        driver_keyword = {"energy": "", "gradient": "Force", "hessian": "Freq"}[spec.driver]

        method_token = model.method
        if model.basis:
            method_token += f"/{normalize_basis(model.basis)}"
        route_tokens = ["#P", method_token]
        if driver_keyword:
            route_tokens.append(driver_keyword)
        # G16's symmetry machinery crashes on ghost atoms (verified RevB.01:
        # "Symmetry image not found in LdSEqv" + segfault in l401)
        if not all(input_model.molecule.real) and "symm" not in route_input.lower():
            route_tokens.append("Symmetry=None")
        if route_input:
            route_tokens.append(route_input)
        lines.append(" ".join(route_tokens))

        # Gaussian's blank lines are load-bearing, including the trailing one
        lines.append("")
        lines.append("QCEngine Gaussian 16 single point")
        lines.append("")
        lines.append(format_molecule(input_model.molecule))
        lines.append("")

        g16 = which("g16")
        formchk = which("formchk")
        # Single shell string (one-element list). GAUSS_SCRDIR is pinned to the
        # qcengine scratch dir (the command's cwd); `&&` ensures a failed g16
        # skips formchk and keeps its own exit code -- the log is still fully
        # captured because it streams through stdout, not a file.
        command = [
            f'export GAUSS_SCRDIR="$PWD" && "{g16}" < {_INPUT_STEM}.gjf '
            f'&& "{formchk}" {_INPUT_STEM}.chk {_INPUT_STEM}.fchk'
        ]

        return {
            "command": command,
            "infiles": {f"{_INPUT_STEM}.gjf": "\n".join(lines) + "\n"},
            "outfiles": [f"{_INPUT_STEM}.fchk"],
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
            shell=True,
            scratch_directory=inputs["scratch_directory"],
            scratch_messy=inputs["scratch_messy"],
            timeout=timeout,
        )
        return success, dexe

    def parse_output(self, outfiles: Dict[str, str], input_model: "AtomicInput") -> AtomicResult:
        stdout = outfiles.pop("stdout")
        stderr = outfiles.pop("stderr", "")

        fchk = outfiles.get(f"{_INPUT_STEM}.fchk")
        if fchk is None:
            raise UnknownError(
                error_stamp("", stdout, stderr) + "\nNo .fchk file was produced despite normal termination."
            )

        props = harvest_fchk(fchk)

        natom = len(input_model.molecule.symbols)
        # Required by the AtomicResultProperties validator whenever
        # return_gradient/return_hessian are present; harmless otherwise.
        props["calcinfo_natom"] = natom

        driver = input_model.specification.driver
        return_result: Any = props["return_energy"]

        if driver == "gradient":
            gradient = parse_fchk_array(fchk, "Cartesian Gradient")
            if gradient.size != 3 * natom:
                raise UnknownError(
                    f"fchk Cartesian Gradient size ({gradient.size}) does not match the molecule (3*{natom})"
                )
            gradient = gradient.reshape(-1, 3)
            props["return_gradient"] = gradient
            return_result = gradient

        elif driver == "hessian":
            packed = parse_fchk_array(fchk, "Cartesian Force Constants")
            hessian = lt_to_square(packed, 3 * natom)
            props["return_hessian"] = hessian
            try:
                gradient = parse_fchk_array(fchk, "Cartesian Gradient")
                if gradient.size == 3 * natom:
                    props["return_gradient"] = gradient.reshape(-1, 3)
            except UnknownError:
                pass
            return_result = hessian

        return AtomicResult(
            input_data=input_model,
            molecule=input_model.molecule,
            properties=props,
            return_result=return_result,
            success=True,
            stdout=stdout,
            provenance=Provenance(creator="Gaussian", version=self.get_version(), routine="g16"),
            extras={"outfiles": {f"{_INPUT_STEM}.fchk": fchk}},
        )
