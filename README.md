## HARP: Human Anchor for Robust Positioning in Monocular Trajectory Annotation

Built on top of [TRAM](https://github.com/yufu-wang/tram), HARP improves the robustness of metric scale estimation for monocular human trajectory annotation. Instead of relying solely on background depth prediction (which fails in low-texture, dark, or stage environments), HARP leverages the metric depth already estimated by pose regression as a geometric anchor.

### Key Features
- **Boundary Sampling**: Extracts SLAM depth at the human-scene boundary to bridge pose regression and SLAM coordinate frames
- **Uncertainty-Aware Fusion**: Fuses human-derived and background-derived scale estimates via inverse-variance weighting (MLE), automatically adapting to scene conditions
- **Temporal Shape Consistency**: Regularizes body shape predictions to stabilize metric depth estimates
- **Drop-in Module**: No retraining required, <1ms overhead per frame

### Results on EMDB-2

| Method | PA-MPJPE | WA-MPJPE<sub>100</sub> | W-MPJPE<sub>100</sub> | RTE(%) |
|--------|----------|------------|-----------|--------|
| TRACE | 58.0 | 529.0 | 1702.3 | 17.7 |
| GLAMR | 56.0 | 280.8 | 726.6 | 11.4 |
| SLAHMR | 61.5 | 326.9 | 776.1 | 10.2 |
| WHAM | 38.2 | 133.3 | 343.9 | 4.6 |
| TRAM | 37.6 | 78.7 | 223.9 | 1.5 |
| **HARP (ours)** | **37.6** | **78.0** | **218.0** | **1.3** |

Under background degradation, HARP reduces scale drift by 50% and trajectory error by 38% compared to TRAM.

## Installation

Follow the TRAM installation steps:

```bash
# Clone with submodules
git clone --recursive https://github.com/XuenYat/HARP

# Create environment
conda create -n tram python=3.10 -y
conda activate tram
bash install.sh

# Compile DROID-SLAM
cd thirdparty/DROID-SLAM
python setup.py install
cd ../..
```

## Prepare Data

Register at [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de) to download SMPL models. Then fetch all models and checkpoints:

```bash
bash scripts/download_models.sh
```

## Inference

### HARP Pipeline (Recommended)
```bash
# Run HARP with uncertainty-aware fusion
python scripts/run_harp.py --video "./video.mov"

# Use TRAM's ZoeDepth scale instead
python scripts/run_harp.py --video "./video.mov" --method tram

# Visualize results
python scripts/visualize_tram.py --video "./video.mov" --method harp
```

### Legacy Step-by-Step Pipeline
```bash
python scripts/estimate_camera.py --video "./video.mov"
python scripts/estimate_humans.py --video "./video.mov"
python scripts/visualize_tram.py --video "./video.mov"
```

## Evaluation on EMDB

```bash
# Full HARP evaluation with per-category breakdown
python scripts/emdb/harp_eval_emdb.py \
    --emdb_root /path/to/emdb \
    --tram_results results/emdb_harp_shapereg \
    --harp_intermediates results/emdb_harp/intermediates \
    --save_dir results/harp_eval

# Background degradation robustness test
python scripts/experiments/harp_degradation.py \
    --emdb_root /path/to/emdb \
    --intermediates_dir results/emdb_harp/intermediates \
    --save_dir results/harp_degradation
```

## Training

```bash
# Train VIMO from HMR2b initialization
python train.py --cfg configs/config_vimo.yaml

# HARP fine-tuning with temporal shape consistency loss
python train.py --cfg configs/config_harp.yaml
```

Download the HMR2b pretrained checkpoint first:
```bash
bash scripts/download_pretrain.sh
```

## Method Overview

HARP computes a human-derived scale from two signals:
1. **Metric depth** (t_z) from VIMO pose regression, grounded in body proportion priors
2. **Boundary SLAM depth** sampled at the human-scene boundary (feet, sides)

These are fused with the background-derived scale via inverse-variance weighting:

```
α* = (α_bg/σ²_bg + α_human/σ²_human) / (1/σ²_bg + 1/σ²_human)
```

When background depth is reliable, the fusion defers to α_bg. In degraded environments, it automatically shifts toward the human anchor.

## Acknowledgements

HARP builds on the excellent [TRAM](https://github.com/yufu-wang/tram) framework. We also benefit from:
- [WHAM](https://github.com/yohanshin/WHAM): visualization and evaluation
- [HMR2.0](https://github.com/shubham-goel/4D-Humans): baseline backbone
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM): baseline SLAM
- [ZoeDepth](https://github.com/isl-org/ZoeDepth): metric depth prediction
- [EMDB](https://github.com/eth-ait/emdb): evaluation dataset

The pipeline also includes [Detectron2](https://github.com/facebookresearch/detectron2), [Segment-Anything](https://github.com/facebookresearch/segment-anything), and [DEVA-Track-Anything](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).

## Citation

```bibtex
@article{xu2025harp,
  title={HARP: Human Anchor for Robust Positioning in Monocular Trajectory Annotation},
  author={Xu, Sicheng and Dou, Huanxin},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```
