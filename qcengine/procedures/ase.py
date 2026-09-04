"""ASE-driven geometry optimization.

Cartesian optimizers (LBFGS / FIRE / PreconLBFGS) cost O(N) per step. geomeTRIC's
internal coordinates converge in fewer steps but build and invert Wilson-B / G
matrices every step, which is O(N^2-N^3): on a 1500-atom periodic slab that is ~200 s
of CPU per step against a ~1 s MLP gradient, i.e. the optimizer is three orders of
magnitude more expensive than the physics (measured: 0.5% GPU utilisation, ~14 h per
optimization). The same slab costs ~0.1 s/step of optimizer overhead here.

Constraints are a force mask (``FixAtoms``), so freezing slab layers is free. geomeTRIC
refuses constraints in Cartesian coordinates precisely because its constraint algebra
lives in the internal-coordinate space; that restriction does not exist here.

When to prefer which: this harness for large systems and cheap gradients (MLPs);
geomeTRIC for expensive gradients on normal-sized molecules, where converging in fewer
steps matters more than the cost of a step.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from qcelemental.models.v2 import AtomicInput, Molecule, OptimizationInput, OptimizationResult
from qcelemental.models.v2 import DriverEnum
from qcelemental.util import safe_version, which_import

from ..exceptions import ConvergenceError, InputError
from .model import ProcedureHarness

if TYPE_CHECKING:
    from ..config import TaskConfig

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
# Convergence threshold, in ASE units (eV/A). NOT geomeTRIC's nominal max-gradient value
# (4.5e-4 Hartree/Bohr = 0.023 eV/A): geomeTRIC must satisfy five criteria simultaneously
# and in practice stops 20-40x tighter than its nominal gmax, so matching gmax alone
# leaves structures visibly under-relaxed.
#
# Calibrated on CO/W52 against geomeTRIC-tric minima (3 sites, MACE lmft-co-d-v0):
#
#   fmax     PreconLBFGS                 plain LBFGS
#   0.023    +0.69 kcal/mol, 0.22 A       +0.97 kcal/mol, 0.27 A
#   0.010    +0.61 kcal/mol, 0.19 A       +0.74 kcal/mol, 0.23 A
#   0.005    +0.02 kcal/mol, 0.03 A       +0.63 kcal/mol, 0.21 A
#   0.001    +0.02 kcal/mol, 0.03 A       +0.43 kcal/mol, 0.17 A  (702 steps)
#
# 0.005 reproduces the tric minimum to 0.02 kcal/mol; tightening further buys ~0.001
# kcal/mol for ~10% more steps.
DEFAULT_FMAX_EV_PER_ANGSTROM = 0.005

# Preconditioned by default. Plain Cartesian LBFGS is ill-conditioned on floppy H-bonded
# systems and plateaus ~0.43 kcal/mol above the true minimum no matter how tight fmax is
# (see the table above) -- the same pathology that makes geomeTRIC's `cart` unusable on
# large ASW clusters. The Exp preconditioner removes it at comparable step count.
DEFAULT_OPTIMIZER = "precon-lbfgs"


def _frozen_indices(constraints: Optional[Dict[str, Any]]) -> List[int]:
    """Extract 0-based frozen-atom indices from a geomeTRIC-style constraints dict.

    Accepts the same structure geomeTRIC's JSON API takes, so a workflow can switch
    optimizer without rewriting its constraints::

        {"freeze": [{"type": "xyz", "indices": [0, 1, 2]}]}

    Anything else raises rather than being ignored: silently dropping a constraint
    would relax atoms the caller believes are fixed and quietly produce wrong physics.
    """
    if not constraints:
        return []
    if not isinstance(constraints, dict):
        raise InputError(
            f"ASE optimizer: 'constraints' must be a dict like "
            f"{{'freeze': [{{'type': 'xyz', 'indices': [...]}}]}}, got {type(constraints).__name__}."
        )
    unsupported = set(constraints) - {"freeze"}
    if unsupported:
        raise InputError(
            f"ASE optimizer supports only 'freeze' constraints, not {sorted(unsupported)}. "
            "Refusing rather than ignoring them."
        )
    indices: List[int] = []
    for entry in constraints.get("freeze", []):
        ctype = str(entry.get("type", "")).lower()
        if ctype != "xyz":
            raise InputError(f"ASE optimizer supports only 'xyz' freeze constraints, not {ctype!r}.")
        indices.extend(int(i) for i in entry.get("indices", []))
    return sorted(set(indices))


def _build_optimizer(name: str, atoms, keywords: Dict[str, Any]):
    """Return an ASE optimizer instance for ``name``."""
    from ase.optimize import FIRE, LBFGS

    name = (name or DEFAULT_OPTIMIZER).lower()
    if name == "lbfgs":
        return LBFGS(atoms, logfile=None)
    if name == "fire":
        return FIRE(atoms, logfile=None)
    if name in ("precon-lbfgs", "preconlbfgs"):
        from ase.optimize.precon import Exp, PreconLBFGS

        # The Exp preconditioner auto-detects a nearest-neighbour scale, which fails on
        # finite clusters in vacuum ("increased r_cut to twice system extent"). Let the
        # caller pin it; the defaults suit condensed/periodic systems.
        #
        # force_stab defaults to True here, unlike ASE. ASE only adds the diagonal
        # stabilisation when there are NO fixed atoms:
        #
        #     if force_stab or len(fixed_atoms) == 0:
        #         diag_coeff += self.mu * self.c_stab
        #
        # assuming constraints remove the singular modes. That assumption fails whenever a
        # mobile fragment drifts beyond r_cut of everything else -- an adsorbate desorbing
        # into a vacuum gap gets an empty neighbour list, hence a zero diagonal entry, hence
        # an EXACTLY SINGULAR matrix. spsolve then returns garbage, no downhill direction is
        # ever found, and the optimizer spins in C at 100% CPU without calling the
        # calculator again (unresponsive even to SIGINT). Observed on ~19% of periodic slab
        # sites, where each stuck worker held its slot until the queue deadlocked.
        # Since this workflow always freezes slab layers and adsorbates can always desorb,
        # stabilisation must be unconditional.
        precon = Exp(
            A=keywords.get("precon_A", 3.0),
            r_cut=keywords.get("precon_r_cut", None),
            r_NN=keywords.get("precon_r_NN", None),
            force_stab=keywords.get("precon_force_stab", True),
        )
        opt = PreconLBFGS(atoms, precon=precon, use_armijo=keywords.get("use_armijo", True), logfile=None)
        return opt
    raise InputError(f"Unknown ASE optimizer {name!r}; choose from 'lbfgs', 'fire', 'precon-lbfgs'.")


class _QCEngineCalculator:
    """ASE calculator that evaluates energy and forces through QCEngine.

    Deliberately routed through ``qcengine.compute`` rather than a program's own ASE
    calculator: the optimization then uses the identical code path as any single-point
    computed later for the same system (same keyword handling, same periodic treatment),
    so geometries and energies cannot drift between stages.
    """

    implemented_properties = ("energy", "forces", "free_energy")

    def __init__(self, spec, program: str, task_config):
        from ase.calculators.calculator import Calculator

        self._spec = spec
        self._program = program
        # qcengine.compute() takes a plain mapping, not a TaskConfig model
        self._task_config = task_config.model_dump() if hasattr(task_config, "model_dump") else task_config
        self.results: Dict[str, Any] = {}
        self.trajectory: List[Any] = []
        self.atoms = None
        self._Calculator = Calculator
        self._cached_positions: Optional[np.ndarray] = None

    # -- ASE calculator protocol -------------------------------------------------
    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self._evaluate(atoms)["energy"]

    def get_forces(self, atoms=None):
        return self._evaluate(atoms)["forces"]

    def get_property(self, name, atoms=None, allow_calculation=True):
        return self._evaluate(atoms)[name]

    def check_state(self, atoms, tol=1e-15):
        return []

    def get_stress(self, atoms=None):
        raise NotImplementedError("ASE procedure does not provide stress.")

    def _evaluate(self, atoms) -> Dict[str, Any]:
        import qcengine

        if atoms is None:
            atoms = self.atoms
        positions = np.asarray(atoms.get_positions(), dtype=float)
        # ASE asks for energy and forces separately, and several times per step; without
        # this cache each optimizer step would fire ~5 gradient evaluations instead of one.
        if self._cached_positions is not None and self.results and np.array_equal(positions, self._cached_positions):
            return self.results
        positions_bohr = positions / BOHR_TO_ANGSTROM
        molecule = Molecule(
            symbols=[str(s) for s in atoms.get_chemical_symbols()],
            geometry=positions_bohr.flatten(),
            fix_com=True,
            fix_orientation=True,
            validate=False,
        )
        atomic_input = AtomicInput(
            molecule=molecule,
            specification=self._spec.model_copy(update={"driver": DriverEnum.gradient}),
        )
        result = qcengine.compute(
            atomic_input, self._program, raise_error=True, task_config=self._task_config
        )
        self.trajectory.append(result)
        gradient = np.asarray(result.return_result, dtype=float).reshape(-1, 3)
        energy_ev = float(result.properties.return_energy) * HARTREE_TO_EV
        forces_ev_ang = -gradient * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        self.results = {"energy": energy_ev, "free_energy": energy_ev, "forces": forces_ev_ang}
        self._cached_positions = positions.copy()
        return self.results


class ASEProcedure(ProcedureHarness):

    _defaults: ClassVar[Dict[str, Any]] = {"name": "ase", "procedure": "optimization"}

    version_cache: Dict[str, str] = {}

    def found(self, raise_error: bool = False) -> bool:
        return which_import(
            "ase",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via `conda install ase -c conda-forge`.",
        )

    def get_version(self) -> str:
        self.found(raise_error=True)
        which_prog = which_import("ase")
        if which_prog not in self.version_cache:
            import ase

            self.version_cache[which_prog] = safe_version(ase.__version__)
        return self.version_cache[which_prog]

    def build_input_model(
        self, data: Union[Dict[str, Any], "OptimizationInput"], *, return_input_schema_version: bool = False
    ) -> "OptimizationInput":
        return self._build_model(data, "OptimizationInput", return_input_schema_version=return_input_schema_version)

    def compute(self, input_model: "OptimizationInput", config: "TaskConfig") -> "OptimizationResult":
        self.found(raise_error=True)

        from ase import Atoms

        opt_spec = input_model.specification
        keywords = dict(opt_spec.keywords or {})
        qc_spec = opt_spec.specification

        program = qc_spec.program or keywords.get("program")
        if not program:
            raise InputError("ASE optimizer requires a gradient program (keywords['program']).")

        fmax = float(keywords.get("fmax", DEFAULT_FMAX_EV_PER_ANGSTROM))
        maxiter = int(keywords.get("maxiter", 500))
        optimizer_name = keywords.get("optimizer", DEFAULT_OPTIMIZER)
        frozen = _frozen_indices(keywords.get("constraints"))

        # Cell / pbc live in the gradient program's keywords (that is where the program
        # itself reads them from); mirror them onto the Atoms so ASE's neighbour lists
        # and any preconditioner see the same periodicity.
        qc_keywords = dict(qc_spec.keywords or {})
        cell = qc_keywords.get("cell")
        pbc = qc_keywords.get("pbc", False)

        molecule = input_model.initial_molecule
        atoms = Atoms(
            symbols=[str(s) for s in molecule.symbols],
            positions=np.asarray(molecule.geometry).reshape(-1, 3) * BOHR_TO_ANGSTROM,
            cell=np.asarray(cell, dtype=float) if cell is not None else None,
            pbc=pbc if cell is not None else False,
        )
        if frozen:
            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(indices=frozen))

        calculator = _QCEngineCalculator(qc_spec, program, config)
        calculator.atoms = atoms
        atoms.calc = calculator

        optimizer = _build_optimizer(optimizer_name, atoms, keywords)
        initial_positions = atoms.get_positions().copy()
        try:
            optimizer.run(fmax=fmax, steps=maxiter)
        except RuntimeError as exc:
            # Exp() auto-detects a nearest-neighbour scale from the structure, which fails
            # on a finite cluster surrounded by vacuum. Say so, rather than letting a bare
            # RuntimeError surface: silently falling back to un-preconditioned LBFGS would
            # be worse, since that plateaus above the true minimum on these systems.
            if "r_cut" in str(exc) or "neighbours" in str(exc):
                raise InputError(
                    f"The '{optimizer_name}' preconditioner could not determine a neighbour "
                    f"cutoff for this structure ({exc}). Set 'precon_r_cut' (and optionally "
                    "'precon_r_NN') in the optimizer keywords -- typical values are "
                    "precon_r_cut=5.0, precon_r_NN=1.0 for molecular systems -- or select "
                    "optimizer='lbfgs', noting that plain LBFGS under-converges on floppy "
                    "H-bonded systems."
                ) from exc
            raise

        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(atoms) else 0.0
        nsteps = int(optimizer.get_number_of_steps())
        if max_force > fmax:
            # NB: ConvergenceError takes a single message argument; passing a program
            # name as a first positional silently discards the diagnostic.
            raise ConvergenceError(
                f"ASE {optimizer_name} did not converge: after {nsteps} steps the maximum force is "
                f"{max_force:.3e} eV/A, above fmax={fmax:.3e} eV/A. Raise 'maxiter' or loosen 'fmax'."
            )

        displacement = atoms.get_positions() - initial_positions
        final_molecule = molecule.model_copy(
            update={"geometry": (atoms.get_positions() / BOHR_TO_ANGSTROM).flatten()}
        )
        trajectory = calculator.trajectory
        return OptimizationResult(
            input_data=input_model,
            final_molecule=final_molecule,
            trajectory_results=trajectory,
            trajectory_properties=[point.properties for point in trajectory],
            success=True,
            provenance={
                "creator": "ase",
                "version": self.get_version(),
                "routine": f"ase.optimize.{optimizer_name}",
            },
            properties={
                "return_energy": trajectory[-1].properties.return_energy if trajectory else None,
                "optimization_iterations": nsteps,
                "final_max_force": max_force * BOHR_TO_ANGSTROM / HARTREE_TO_EV,
                "final_rms_force": float(np.sqrt((forces**2).sum(axis=1).mean()))
                * BOHR_TO_ANGSTROM
                / HARTREE_TO_EV,
                "final_max_displacement": float(np.linalg.norm(displacement, axis=1).max())
                / BOHR_TO_ANGSTROM,
                "final_rms_displacement": float(np.sqrt((displacement**2).sum(axis=1).mean()))
                / BOHR_TO_ANGSTROM,
            },
        )
