from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_SRC_DIR = PACKAGE_DIR.parent


def find_upstream_dir() -> Path:
    """Support both a normal clone and the older locally nested clone."""
    candidates = (
        WORKSPACE_SRC_DIR / "diffusion_policy",
        WORKSPACE_SRC_DIR / "diffusion_policy" / "diffusion_policy",
    )
    marker = Path("diffusion_policy/model/diffusion/conditional_unet1d.py")
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    # Preserve a useful import error while exposing the conventional expected path.
    return candidates[0]


UPSTREAM_DIR = find_upstream_dir()

STATE_DIM = 4
CAMERA_NAMES = ("wrist_d405",)
JOINT_NAMES = ("J1", "J2", "J3", "J4")

DEFAULT_DATASET_DIR = WORKSPACE_SRC_DIR / "scara_act" / "datasets" / "leaf_cutting_experiment_rgb_640x480"
DEFAULT_CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints" / "leaf_cutting_diffusion_h32_a16"

JOINT_LIMITS = {
    "J1": (0.0015, 1.18),
    "J2": (-1.65316, 1.65316),
    "J3": (0.10944, 6.07375),
    "J4": (-16.7552, 16.7552),
}

DEFAULT_JOINT_TARGET = {
    "J1": 0.0015,
    "J2": 0.0,
    "J3": 1.0,
    "J4": 0.0,
}

# Per 30 Hz command: J1 in metres, J2-J4 in radians.
DEFAULT_MAX_DELTA = (0.01, 0.06, 0.06, 0.09)
