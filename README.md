# dye-patch

Patch-level herbicide dye detection from drone imagery using a frozen DINOv3 vision transformer with a per-patch linear classification head. The system produces a 24x24 spatial prediction grid over each image tile, identifying the presence of herbicide dye at fine spatial resolution. Synthetic dye overlays are generated during training to augment data diversity, with overlay color properties derived from empirical HSV delta measurements. This builds on prior work demonstrating effective drone-based dye detection using tile-level classification with DINOv2.

## Installation

```bash
git clone https://github.com/mosscoder/dye-patch.git
cd dye-patch
pip install -e .
```

The DINOv3 backbone requires the development version of transformers, which must be installed separately:

```bash
pip install git+https://github.com/huggingface/transformers.git
```
