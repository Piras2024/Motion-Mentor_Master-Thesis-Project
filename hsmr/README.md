# HSMR — 3D Body Pose Reconstruction

This directory contains the HSMR codebase used to extract 3D body-pose
sequences from video. **Only the code is committed.** The model weights and
body-model files (~2.8 GB) must be downloaded separately.

The original upstream README is preserved as `README_UPSTREAM.md`.

## What this folder contains

The full HSMR source tree as used in the thesis. The original `data_inputs/`
and `data_outputs/` directories are *not* included because they would push
the repository to ~6 GB.

## What you need to download

To run HSMR you need:

1. **The HSMR checkpoint** (`hsmr.ckpt`, ~2.6 GB) — place under:
   `data_inputs/released_models/HSMR-ViTH-r1d1/checkpoints/hsmr.ckpt`

2. **SMPL / SKEL body-model files** (~200 MB) — place under
   `data_inputs/body_models/`:
   - `skel_male.pkl`, `skel_female.pkl`
   - `J_regressor_SMPL_MALE.pkl`, `J_regressor_SKEL_mix_MALE.pkl`
   - `SMPL_to_J19.pkl`

See the upstream HSMR repository for download instructions:
https://github.com/IsshikiHugh/HSMR

## Setup

HSMR requires a separate conda environment. Refer to `requirements.txt` and
`README_UPSTREAM.md`.

```bash
conda create -n hsmr python=3.10
conda activate hsmr
pip install -r requirements.txt
```

## Usage from the demo

The demo's default flow assumes you have either:

- precomputed `_guofeats.npy` files (see `../demo/test_videos/guofeats/` —
  three are shipped with the repo), in which case run the demo with
  `--skip-hsmr`; **or**
- a working HSMR environment plus body models / checkpoint, in which case
  the pipeline scripts under `../pipeline/` can extract features end-to-end.

For the scope of the thesis demo we recommend `--skip-hsmr` to keep the
example self-contained.
