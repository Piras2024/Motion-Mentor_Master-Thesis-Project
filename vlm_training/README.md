# VLM Training

Six fine-tuning scripts, one per (backbone × motion-injection) combination.
All use LoRA on the language-model submodule of the VLM.

## Scripts

| Script | Backbone | Motion injection |
|---|---|---|
| `finetune_gemma4.py`             | Gemma4-E2B-it    | none |
| `finetune_qwen2vl.py`            | Qwen2-VL-2B-Instruct | none |
| `finetune_gemma4_motion_proj.py` | Gemma4-E2B-it    | per-timestep `Linear(512→2048)` from VQ-VAE codebook |
| `finetune_qwen_motion_proj.py`   | Qwen2-VL-2B-Instruct | per-timestep projection |
| `finetune_gemma4_qformer.py`     | Gemma4-E2B-it    | 8-query Q-Former cross-attending the full motion sequence |
| `finetune_qwen_qformer.py`       | Qwen2-VL-2B-Instruct | Q-Former |

## Matched training recipe

All checkpoints in `../checkpoints/` were produced with these settings:

| Setting | Value |
|---|---|
| Epochs | 20 |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| Scheduler | OneCycleLR (pct_start=0.1) |
| Gradient accumulation | 8 |
| Batch size | 1 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| LoRA target | language model q/k/v/o + gate/up/down projections |
| dtype | bfloat16 |
| Q-Former queries (Q-Former variants) | 8 |
| Q-Former layers (Q-Former variants) | 1 |

## What you need before training

1. **Labelled video dataset** placed under `/path/to/videos/{rdls,squat_micc}/`.
2. **Pre-extracted Guo features** for every video, under `/path/to/hsmr_guofeats/`,
   named `HSMR-{video_stem}_guofeats.npy`. See `../pipeline/`.
3. **The in-domain VQ-VAE** (already shipped under
   `../motion_encoder/checkpoints/vqvae_indomain_best.pth`).
4. **The label JSON files** (see `../labels/`).

## Example training command

```bash
python vlm_training/finetune_qwen_qformer.py \
    --labels  labels/labels_5var_reusable.json \
    --vqvae-ckpt motion_encoder/checkpoints/vqvae_indomain_best.pth \
    --output-dir checkpoints/qwen_qformer_v2 \
    --wandb-run  qwen_qformer_v2 \
    --epochs 20
```

The script will:

1. Build the stratified train / val / test split with seed=42.
2. Save the exact split as `{output-dir}/split.json` (so evaluation uses
   the same held-out set).
3. Train for 20 epochs, saving the best-validation checkpoint at
   `{output-dir}/best/`.
4. Save the Q-Former weights as `{output-dir}/best/qformer.pt` and the
   reconstruction config as `{output-dir}/best/qformer_config.json`.

(For motion-projection variants, the projection is saved as
`motion_proj.pt`.)
