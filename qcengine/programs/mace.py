import os
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple, Union

from qcelemental.models.v2 import AtomicResult, FailedOperation, Provenance
from qcelemental.util import safe_version, which_import

from qcengine.exceptions import InputError, ResourceError
from qcengine.programs.model import ProgramHarness
from qcengine.units import ureg

if TYPE_CHECKING:
    from qcelemental.models.v2 import AtomicInput, FailedOperation

    from qcengine.config import TaskConfig


def _parse_periodic_keywords(
    keywords: Optional[Dict[str, Any]],
) -> Tuple[Tuple[bool, bool, bool], Optional[Any]]:
    """Extract MACE periodic-boundary settings from a QC-spec keywords dict.

    Recognised keys (both optional):
        ``pbc``:  length-3 iterable of bools (per-axis periodicity)
        ``cell``: 3x3 lattice vectors in Angstrom

    Returns ``(pbc, cell)`` suitable for a ``mace.data.utils.Configuration``.
    When ``pbc`` is absent or entirely False the harness runs non-periodic
    (``pbc=(False,False,False)``, ``cell=None``) — the historical default,
    which lets MACE auto-size a box around the atoms. When any pbc axis is
    True, a matching 3x3 ``cell`` must be supplied or ``InputError`` is
    raised so periodic runs never silently fall back to non-periodic.
    """
    import numpy as np

    kw = keywords or {}
    kw_pbc = kw.get("pbc")
    kw_cell = kw.get("cell")

    if kw_pbc is None or not any(bool(x) for x in kw_pbc):
        return (False, False, False), None

    pbc_tuple = tuple(bool(x) for x in kw_pbc)
    if len(pbc_tuple) != 3:
        raise InputError(
            f"MACE harness: keywords['pbc'] must be length 3 (got {len(pbc_tuple)})."
        )
    if kw_cell is None:
        raise InputError(
            "MACE harness: periodic boundary conditions requested "
            f"(keywords['pbc'] = {list(pbc_tuple)}) but no cell supplied. "
            "Set keywords['cell'] to a 3x3 list of lattice vectors in Angstrom."
        )
    cell = np.asarray(kw_cell, dtype=float)
    if cell.shape != (3, 3):
        raise InputError(
            "MACE harness: keywords['cell'] must be a 3x3 matrix "
            f"(got shape {cell.shape})."
        )
    return pbc_tuple, cell


def _parse_hessian_keywords(keywords: Optional[Dict[str, Any]], natom: int) -> Tuple[list, str, float]:
    """Hessian settings from a QC-spec keywords dict (all optional).

    ``active_atoms``    0-based atom indices of the partial Hessian (PHVA): only these atoms are displaced
                        / differentiated, all other atoms stay in the calculation as a fixed environment.
                        Default: all atoms (the full Hessian).
    ``hessian_method``  ``"fd"`` (default; central differences of the forces) or ``"analytic"`` (autograd
                        rows of the force graph).
    ``fd_step``         finite-difference displacement in Angstrom (default 0.005).

    Returns ``(active, method, step)``; ``active`` keeps the order given.
    """
    kw = keywords or {}
    active = kw.get("active_atoms")
    if active is None:
        active = list(range(natom))
    else:
        try:
            active = [int(i) for i in active]
        except (TypeError, ValueError):
            raise InputError("MACE harness: keywords['active_atoms'] must be a list of atom indices.")
        if not active:
            raise InputError("MACE harness: keywords['active_atoms'] is empty.")
        if len(set(active)) != len(active):
            raise InputError("MACE harness: keywords['active_atoms'] has repeated indices.")
        if min(active) < 0 or max(active) >= natom:
            raise InputError(
                f"MACE harness: keywords['active_atoms'] must lie in 0..{natom - 1} "
                f"(got {min(active)}..{max(active)})."
            )
    method = str(kw.get("hessian_method", "fd")).lower()
    if method not in ("fd", "analytic"):
        raise InputError(f"MACE harness: keywords['hessian_method'] must be 'fd' or 'analytic' (got {method!r}).")
    step = float(kw.get("fd_step", 0.005))
    if not step > 0:
        raise InputError("MACE harness: keywords['fd_step'] must be positive (Angstrom).")
    return active, method, step


class MACEHarness(ProgramHarness):
    """Can be used to execute a published MACE-OFF23 model or local mace model.
    For more info on the MACE-OFF23 models see <https://doi.org/10.48550/arXiv.2312.15211>.
    The models can be found at <https://github.com/ACEsuit/mace-off>
    """

    _CACHE = {}

    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "MACE",
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }
    version_cache: Dict[str, str] = {}

    def found(self, raise_error: bool = False) -> bool:
        return which_import(
            "mace",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via `mamba install pymace -c conda-forge`",
        )

    def get_version(self) -> str:
        self.found(raise_error=True)

        which_prog = which_import("mace")
        if which_prog not in self.version_cache:
            import mace

            self.version_cache[which_prog] = safe_version(mace.__version__)

        return self.version_cache[which_prog]

    MODEL_PATH_ENV: ClassVar[str] = "MACE_MODEL_PATH"
    DTYPE_ENV: ClassVar[str] = "MACE_DTYPE"
    CUEQ_ENV: ClassVar[str] = "MACE_CUEQ"

    @classmethod
    def resolve_model_path(cls, name: str) -> str:
        """Resolve a model reference to a readable file, portably across machines.

        A QCFractal spec is shared by every worker that may run the record, so a
        machine-specific absolute path pins the record to one cluster (and one
        username). Resolution order:

        1. ``~`` and ``$VAR`` expansion, then the name as given (absolute or
           relative to the worker's cwd);
        2. if the name has no directory component, each entry of
           ``$MACE_MODEL_PATH`` (os.pathsep-separated), so a spec can carry just
           ``lmft-co-d-v0.model`` and each cluster points at its own model store
           from the manager's ``worker_init``.

        Note ``$VAR`` rarely survives a round trip: qcportal lowercases QCSpec
        method strings server-side, so ``$MODELS`` comes back as ``$models``.
        Prefer a bare filename plus ``$MACE_MODEL_PATH``, or a ``~``-relative
        path (unaffected by lowercasing).

        Returns the resolved path, or the expanded name unchanged when nothing
        matched, so the caller raises a single consistent error.
        """
        expanded = os.path.expanduser(os.path.expandvars(name))
        if os.path.isfile(expanded):
            return expanded
        if os.path.dirname(expanded):
            return expanded
        search = os.environ.get(cls.MODEL_PATH_ENV, "")
        for root in (d for d in search.split(os.pathsep) if d):
            candidate = os.path.join(os.path.expanduser(root), expanded)
            if os.path.isfile(candidate):
                return candidate
        return expanded

    def load_model(self, name: str, device: str = "cpu"):
        """Compile and cache the model to make it faster when calling many times in serial.

        ``device`` is where the model will run; the cuEq conversion needs it because
        ``conv_fusion`` is only enabled for ``"cuda"``.
        """
        model_name = name.lower()
        import torch

        # Weights are cast to the active default dtype: MACE checkpoints are stored
        # in float64, so under MACE_DTYPE=float32 the inputs would be float32 while
        # the weights stayed float64 ("both inputs should have same dtype"). The
        # cache is keyed on the dtype too, since a compiled model is dtype-specific.
        dtype = torch.get_default_dtype()

        # cuEquivariance swaps in fused tensor-product kernels. Measured against the
        # e3nn path in float64: dE = 3e-6 kcal/mol, max|dF| = 2e-8 eV/A (i.e. identical),
        # with 5.4-5.8x less GPU memory -- 4.6 GiB per 1000 atoms instead of ~29, which
        # is what makes float64 affordable on large slabs (1500 atoms: ~6.8 vs ~43 GiB).
        # Kept an environment switch rather than a spec keyword: it is a hardware
        # deployment choice, and because it is numerically equivalent, existing records
        # stay valid and directly comparable.
        use_cueq = os.environ.get(self.CUEQ_ENV, "").lower() in ("1", "true", "yes")

        if model_name in ["small", "medium", "large"]:
            cache_key = (model_name, dtype, use_cueq)
            if cache_key in self._CACHE:
                return self._CACHE[cache_key]
            from e3nn.util import jit
            from mace.calculators.foundations_models import mace_off

            model = mace_off(model=model_name, return_raw_model=True)
        else:
            # Cache on the RESOLVED path so two spellings of the same file compile once.
            resolved = self.resolve_model_path(model_name)
            cache_key = (resolved, dtype, use_cueq)
            if cache_key in self._CACHE:
                return self._CACHE[cache_key]

            from e3nn.util import jit

            if not os.path.isfile(resolved):
                searched = os.environ.get(self.MODEL_PATH_ENV) or "(unset)"
                raise InputError(
                    f"MACE model not found: {name!r} (resolved to {resolved!r}). "
                    f"The mace harness runs a local model file or a MACE-OFF23 model "
                    f"(`small`, `medium`, `large`). For a bare filename set "
                    f"{self.MODEL_PATH_ENV} on the worker; it is currently {searched}."
                )
            model = torch.load(resolved, map_location=torch.device("cpu"))

        model = model.to(dtype=dtype)
        if use_cueq:
            try:
                from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
            except ImportError as exc:
                raise ResourceError(
                    f"{self.CUEQ_ENV} is set but cuequivariance is not installed in this "
                    "environment (pip install cuequivariance cuequivariance-torch "
                    "cuequivariance-ops-torch-cu12 nvidia-ml-py)."
                ) from exc
            # NB: a converted model must NOT go through e3nn's jit.compile -- cuEq
            # supplies its own fused ops, and MACE's own calculator likewise skips it.
            # Pass the target device: convert_e3nn_cueq sets conv_fusion=(device=="cuda"),
            # and without the fused convolution the first tensor product materialises
            # its per-edge output (8 GiB for a 1500-atom slab in float64), which is
            # exactly the allocation cuEq is meant to avoid. MACECalculator does the same.
            comp_mod = run_e3nn_to_cueq(model, device=device, return_model=True)
        else:
            comp_mod = jit.compile(model)
        self._CACHE[cache_key] = (comp_mod, float(model.r_max), model.atomic_numbers)
        return self._CACHE[cache_key]

    def compute(self, input_data: "AtomicInput", config: "TaskConfig") -> Union["AtomicResult", "FailedOperation"]:

        self.found(raise_error=True)

        import mace
        import numpy as np
        import torch
        from mace.data import AtomicData
        from mace.data.utils import AtomicNumberTable, Configuration
        from mace.tools.torch_geometric import DataLoader

        # Precision from $MACE_DTYPE (default float64). float32 roughly halves
        # activation memory, which is what lets large systems fit a GPU for
        # *geometry* work: a 1500-atom periodic slab needs ~43 GB of a 44 GB L40S
        # in float64 and OOMs, ~21 GB in float32. Binding energies still want
        # float64 (float32 noise is comparable to a physisorption well), and the
        # BE single-points are energy-only so they fit. Opt in per manager via
        # worker_init, so precision is a deployment choice, not a code change.
        torch.set_default_dtype(
            torch.float32 if os.environ.get(self.DTYPE_ENV, "").lower() == "float32" else torch.float64
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Failure flag
        ret_data = {"success": False}

        # Build model
        method = input_data.specification.model.method

        # load the torch model which can be a MACE-OFF23 or local model
        model, r_max, atomic_numbers = self.load_model(name=method, device=device.type)

        z_table = AtomicNumberTable([int(z) for z in atomic_numbers])
        atomic_numbers = input_data.molecule.atomic_numbers
        # Read pbc / cell from the QC specification keywords (both optional).
        # Default is non-periodic — cell=None lets mace auto-size a box around
        # the atoms, matching the harness's pre-existing behaviour.
        pbc, cell = _parse_periodic_keywords(input_data.specification.keywords)

        # mace >= 0.3.10 made `properties`/`property_weights` required
        # Configuration arguments (training labels; empty for inference).
        # Fall back to the old signature for older mace versions.
        positions = input_data.molecule.geometry * ureg.conversion_factor("bohr", "angstrom")

        def build_input(pos):
            """Model input (dict on the model's device) for positions in Angstrom; the neighbour
            list is rebuilt, so it is valid for displaced geometries too."""
            try:
                config = Configuration(
                    atomic_numbers=atomic_numbers,
                    positions=pos,
                    pbc=pbc,
                    cell=cell,
                    properties={},
                    property_weights={},
                )
            except TypeError:
                config = Configuration(
                    atomic_numbers=atomic_numbers,
                    positions=pos,
                    pbc=pbc,
                    cell=cell,
                )
            data_loader = DataLoader(
                dataset=[AtomicData.from_config(config, z_table=z_table, cutoff=r_max)],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            # GPU-safe: move the input batch onto the model's device (no-op on CPU).
            # Without this, model on cuda + data on cpu raises
            # "Expected all tensors to be on the same device".
            return next(iter(data_loader)).to(device).to_dict()

        model.to(device)
        input_dict = build_input(positions)
        mace_data = model(input_dict, compute_force=True)
        # Bring results back to host before pint/numpy/JSON serialization:
        # .numpy() raises on CUDA tensors.
        _energy = mace_data["energy"].detach().cpu()
        _forces = mace_data["forces"].detach().cpu()
        ret_data["properties"] = {"return_energy": _energy * ureg.conversion_factor("eV", "hartree")}

        if input_data.specification.driver == "energy":
            ret_data["return_result"] = ret_data["properties"]["return_energy"].numpy().item()
        elif input_data.specification.driver == "gradient":
            ret_data["return_result"] = (
                np.asarray(-1.0 * _forces * ureg.conversion_factor("eV / angstrom", "hartree / bohr"))
                .ravel()
                .tolist()
            )

        elif input_data.specification.driver == "hessian":
            # Partial Hessian (PHVA) of keywords['active_atoms'] (default: all atoms, the full Hessian).
            # Every model call is on the whole system: the fixed atoms are the environment of the active
            # ones. Only the active block is computed and returned, (3 n_active) x (3 n_active), atoms in
            # the order of 'active_atoms', coordinates x, y, z.
            active, hess_method, step = _parse_hessian_keywords(input_data.specification.keywords, len(atomic_numbers))
            pos0 = np.array(positions, dtype=float).reshape(-1, 3)
            cols = torch.as_tensor(active, device=device)
            ndof = 3 * len(active)
            if hess_method == "fd":
                # central differences of the forces: two force calls per active coordinate
                hess = np.zeros((ndof, ndof))
                for a, i in enumerate(active):
                    for k in range(3):
                        f = []
                        for sign in (+1.0, -1.0):
                            pos = pos0.copy()
                            pos[i, k] += sign * step
                            out = model(build_input(pos), compute_force=True)
                            f.append(out["forces"].detach()[cols].reshape(-1).cpu().numpy())
                        hess[3 * a + k] = -(f[0] - f[1]) / (2.0 * step)
            else:
                # analytic: one backward pass through the force graph per active coordinate
                # (training=True keeps the graph of the forces); columns restricted to the active atoms
                inp = build_input(pos0)
                out = model(inp, compute_force=True, training=True)
                forces, pos_t = out["forces"], inp["positions"]
                rows = []
                for i in active:
                    for k in range(3):
                        (g,) = torch.autograd.grad(forces[i, k], pos_t, retain_graph=True)
                        rows.append(-g[cols].reshape(-1).detach())
                hess = torch.stack(rows).cpu().numpy()
            hess = 0.5 * (hess + hess.T)
            ret_data["return_result"] = (
                (hess * ureg.conversion_factor("eV / angstrom**2", "hartree / bohr**2")).ravel().tolist()
            )
            ret_data["extras"] = {
                "mace_hessian": {
                    "active_atoms": active,
                    "partial": len(active) < len(atomic_numbers),
                    "method": hess_method,
                    "fd_step_angstrom": step if hess_method == "fd" else None,
                    "n_atoms": len(atomic_numbers),
                    "layout": "(3 n_active) x (3 n_active), atoms in the order of active_atoms, xyz; symmetrized",
                }
            }

        else:
            raise InputError("MACE only supports the energy, gradient and hessian driver methods.")

        ret_data["input_data"] = input_data
        ret_data["molecule"] = input_data.molecule
        ret_data["provenance"] = Provenance(creator="mace", version=mace.__version__, routine="mace")
        ret_data["schema_name"] = "qcschema_atomic_result"
        ret_data["success"] = True

        # Form up a dict first, then sent to BaseModel to avoid repeat kwargs which don't override each other
        return AtomicResult(**ret_data)
