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
MODEL_NAME = "facebook/dinov3-vit7b16-pretrain-sat493m"
NUM_CLASSES = 2  # binary: 0=none, 1=dye
# Normalization stats from AutoImageProcessor for dinov3-vit7b16-pretrain-sat493m
NORM_MEAN = [0.430, 0.411, 0.296]
NORM_STD = [0.213, 0.156, 0.143]

# =============================================================================
# Labels
# =============================================================================
LABEL_NONE = 0
LABEL_DYE = 1
# Overlay color names (for HSV delta selection, not classification labels)
DYE_COLORS = ["red", "blue"]

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
AUG_ROTATION_DEGREES = 15
AUG_BLUR_KERNEL = 5
AUG_BLUR_SIGMA = (0.1, 2.0)
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.15

# =============================================================================
# Learning rate sweep
# =============================================================================
LR_GRID = [0.0001, 0.0005, 0.001, 0.005]
LR_SEEDS = [0, 1, 2, 3, 4]
WEIGHT_DECAY = 0.01

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

# =============================================================================
# SLURM
# =============================================================================
SLURM_PARTITION = "preempt"
SLURM_CPUS = 16
SLURM_MEM = "32G"
SLURM_GPU = 1
