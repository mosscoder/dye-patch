"""
Central configuration for dye patch-level classification.
"""

# =============================================================================
# Spatial constants
# =============================================================================
GSD_CM = 0.7
GSD_M = 0.007
PRECROP_SIZE = 512
MODEL_INPUT_SIZE = 384
VIT_PATCH_SIZE = 16
GRID_DIM = MODEL_INPUT_SIZE // VIT_PATCH_SIZE  # 24
NUM_PATCHES = GRID_DIM * GRID_DIM  # 576
JITTER_PX = PRECROP_SIZE - MODEL_INPUT_SIZE  # 128 total range
TILE_METERS = PRECROP_SIZE * GSD_M  # 3.584
CROP_METERS = MODEL_INPUT_SIZE * GSD_M  # 2.688
GRID_SPACING_M = 4.0
EVAL_CROP_OFFSET = (PRECROP_SIZE - MODEL_INPUT_SIZE) // 2  # 64px centre crop offset

# =============================================================================
# Model
# =============================================================================
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
NUM_CLASSES = 3  # ternary: 0=none, 1=red, 2=blue
# Normalization stats from AutoImageProcessor (satellite-pretrained)
NORM_MEAN = [0.430, 0.411, 0.296]
NORM_STD = [0.213, 0.156, 0.143]

# =============================================================================
# Labels
# =============================================================================
LABEL_NONE = 0
LABEL_RED = 1
LABEL_BLUE = 2
COLOR_TO_LABEL = {"red": LABEL_RED, "blue": LABEL_BLUE}
DYE_COLORS = ["red", "blue"]

# HuggingFace dataset
HF_REPO = "mpg-ranch/dye-patch"

# =============================================================================
# Synthetic dye blob generation
# =============================================================================
BLOB_SIZE_RANGE_M = (0.1, 1.0)
BLOB_SIZE_RANGE_PX = (
    int(BLOB_SIZE_RANGE_M[0] / GSD_M),   # 14
    int(BLOB_SIZE_RANGE_M[1] / GSD_M),   # 143
)

# =============================================================================
# Augmentations (match wolverines, exclude grayscale)
# =============================================================================
AUG_PROB = 0.5
AUG_BLUR_KERNEL = 5
AUG_BLUR_SIGMA = (0.1, 2.0)
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.15

# =============================================================================
# Learning rate sweep
# =============================================================================
LR_GRID = [0.000125, 0.00025, 0.0005, 0.001, 0.002]
LR_SEEDS = [0, 1, 2]
WEIGHT_DECAY = 0.01

# =============================================================================
# Negative sampling
# =============================================================================
NEG_MULTIPLIER = 10  # default neg:pos ratio (swept in 00b_sweep_neg)
NEG_MULTIPLIERS = [2, 4, 8, 16]  # sweep grid
NO_DYE_NEG_SAMPLE = 10  # fixed neg count for tiles with no dye

# =============================================================================
# Training configs
# =============================================================================
CONFIGS = ["real_only", "hybrid", "synth_local", "synth_offsite"]
SYNTH_CONFIGS = ["hybrid", "synth_local", "synth_offsite"]

# Tuning subsampling (points, not tiles — each point has 3 monthly tiles)
TUNING_POINTS_PER_SET = 40
TUNING_TEST_FRAC = 0.3  # 70/30 train/test

# Eval
EVAL_SEED = 0
MONTHS = ["may", "july", "october"]
