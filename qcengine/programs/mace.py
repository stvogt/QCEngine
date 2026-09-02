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

    def load_model(self, name: str):
        """Compile and cache the model to make it faster when calling many times in serial"""
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
            comp_mod = run_e3nn_to_cueq(model, return_model=True)
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
        model, r_max, atomic_numbers = self.load_model(name=method)

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
        try:
            config = Configuration(
                atomic_numbers=atomic_numbers,
                positions=positions,
                pbc=pbc,
                cell=cell,
                properties={},
                property_weights={},
            )
        except TypeError:
            config = Configuration(
                atomic_numbers=atomic_numbers,
                positions=positions,
                pbc=pbc,
                cell=cell,
            )

        data_loader = DataLoader(
            dataset=[AtomicData.from_config(config, z_table=z_table, cutoff=r_max)],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        model.to(device)
        # GPU-safe: move the input batch onto the model's device (no-op on CPU).
        # Without this, model on cuda + data on cpu raises
        # "Expected all tensors to be on the same device".
        batch = next(iter(data_loader)).to(device)
        input_dict = batch.to_dict()
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

        else:
            raise InputError("MACE only supports the energy and gradient driver methods.")

        ret_data["input_data"] = input_data
        ret_data["molecule"] = input_data.molecule
        ret_data["provenance"] = Provenance(creator="mace", version=mace.__version__, routine="mace")
        ret_data["schema_name"] = "qcschema_atomic_result"
        ret_data["success"] = True

        # Form up a dict first, then sent to BaseModel to avoid repeat kwargs which don't override each other
        return AtomicResult(**ret_data)
