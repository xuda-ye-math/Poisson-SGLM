"""Single source of truth for the paper experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
PREPARED_DIR = OUTPUT_ROOT / "prepared"
REFERENCE_DIR = OUTPUT_ROOT / "reference"
FORMAL_SAMPLING_DIR = OUTPUT_ROOT / "formal_sampling"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

DIGITS = (4, 9)
VALIDATION_PER_DIGIT = 500
VALIDATION_SEED_BASE = 314159
PRIOR_VARIANCE = 1e-3

BATCHES = (1024, 2048, 4096)
METHODS = ("sgld", "sglm")
REPETITIONS = 20
BURN_IN = 16_384
RETAINED = 131_072
PREFIXES = tuple(8192 * j for j in range(1, 17))
SNAPSHOT_COUNT = 256
SNAPSHOT_INDICES = tuple(range(0, RETAINED, RETAINED // SNAPSHOT_COUNT))

REFERENCE_CHAINS = 4
REFERENCE_WARMUP = 16_384
REFERENCE_RETAINED = 524_288
REFERENCE_TARGET_ACCEPT = 0.9
REFERENCE_MAX_TREE_DEPTH = 10

BOOTSTRAPS = 5000
REFERENCE_BOOTSTRAP_BLOCK = 64

FUNCTION_NAMES = (
    "cos_z1", "cos_z2", "cos_z3", "cos_z4",
    "cos_z5", "cos_z6", "cos_z7", "cos_z8",
    "bump1d_z1", "bump1d_z2", "bump1d_z3", "bump1d_z4",
    "bump2d_z1_z2", "bump2d_z3_z4", "bump2d_z5_z6", "bump2d_z7_z8",
)
FIGURE_FUNCTIONS = (
    "cos_z1", "cos_z4", "bump1d_z2", "bump1d_z3",
    "bump2d_z5_z6", "bump2d_z7_z8",
)
FIGURE_TITLES = (
    r"$f_{C_1}$", r"$f_{C_4}$", r"$f_{G_2}$", r"$f_{G_3}$",
    r"$f_{G_{5,6}}$", r"$f_{G_{7,8}}$",
)

MODEL_FILE = PREPARED_DIR / "model.npz"
PREPARE_MANIFEST = PREPARED_DIR / "manifest.json"
REFERENCE_SUMMARY = REFERENCE_DIR / "reference.json"


def pair_file(batch: int, repetition: int) -> Path:
    return FORMAL_SAMPLING_DIR / f"batch_{batch}" / f"rep_{repetition}.npz"


def pair_manifest(batch: int, repetition: int) -> Path:
    return FORMAL_SAMPLING_DIR / f"batch_{batch}" / f"rep_{repetition}.json"
