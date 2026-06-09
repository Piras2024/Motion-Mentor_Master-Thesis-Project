# MotionLLM — Motion-Only Corrective Feedback

A **video-free** baseline: corrective feedback is generated from 3D motion
alone. The pipeline is `HSMR → Guo features → in-domain VQ-VAE → discrete
motion tokens → Gemma-2-2B (LoRA)`. No image or video signal ever reaches the
language model — it sees only the motion-token sequence.

This is the motion-only counterpart to the VLM track in `../vlm_training/`. It
quantifies how much of the task is solvable from pose alone (answer: a fair
amount in-distribution, very little out-of-distribution — see
`../docs/results.md`).

## Architecture

| Component | Detail |
|---|---|
| LLM backbone | `google/gemma-2-2b-it` (frozen) |
| Motion encoder | HumanVQVAE — encodes `(T-1, 263)` Guo features into discrete tokens |
| Adapters | two LoRA adapters: `t2m` (frozen) and `m2t` (fine-tuned), r=16 α=32 dropout=0.05 |
| Added tokens | 512 motion codebook tokens + 2 motion delimiters, embeddings/lm_head rows trained |
| Instruction | *"Analyze this motion and identify the main technical error…"* |

The VQ-VAE checkpoint is the same in-domain encoder used by the VLM motion
variants: `../motion_encoder/checkpoints/vqvae_indomain_best.pth`.

## Layout

```
motionllm/
├── finetune_corrective.py   training (LoRA fine-tune of the m2t adapter)
├── eval_corrective.py       in-distribution eval (BERTScore classifier)
├── eval_ood.py              out-of-distribution eval against GT labels
├── models/                  MotionLLM + VQ-VAE (self-contained, no external deps)
├── options/option_llm.py    argument/hyper-parameter defaults
├── utils/augmentations.py   temporal motion augmentations (speed/mirror/reverse)
└── meta/                    Guo-feature normalization stats (mean.npy, std.npy)
```

All scripts `chdir` to this directory and add it to `sys.path`, so they run
self-contained — there is no dependency on the original `motion-agent` tree.

## Checkpoints

| File | What |
|---|---|
| `../checkpoints/motionllm_corrective_v1/best.pth` | the shipped best model (LoRA + motion-token rows, ~508 MB) |
| `../checkpoints/motionllm_base/pretrained.pth` | HumanML3D-pretrained MotionLLM, the fine-tuning starting point |

> **Note on size.** `save_model` stores only the trained deltas (LoRA weights +
> the ~514 added embedding/lm_head rows). An upstream slicing bug used to
> serialize the *entire* base vocabulary as a view, bloating checkpoints to
> 2.86 GB; both shipped files were re-saved with the views cloned (`.clone()`),
> bringing them to ~508 MB with identical weights. The fix is also applied in
> `models/mllm.py:save_model`, so newly trained checkpoints are compact.

## Evaluate the shipped model

Run from inside this directory. The checkpoint, VQ-VAE and BERTScore references
all default to the repo paths.

```bash
# In-distribution (needs the test Guo features referenced in splits.json)
python eval_corrective.py \
    --splits ../checkpoints/motionllm_corrective_v1/splits.json \
    --use-bertscore --device cuda:0

# Out-of-distribution (point at a dir of {id}_guofeats.npy + a GT json)
python eval_ood.py \
    --npy-dir /path/to/ood_guofeats \
    --gt-json /path/to/ground_truth.json \
    --use-bertscore --device cuda:0
```

`--do-sample --temperature 0.8 --top-p 0.9` produces more varied feedback at a
small accuracy cost; greedy (default) is reproducible.

## Reproduce training

You need pre-extracted Guo features for the labelled videos (see `../pipeline/`
and `../hsmr/`), one `HSMR-{video_stem}_guofeats.npy` per clip.

```bash
python finetune_corrective.py \
    --guofeats-dir /path/to/hsmr_guofeats \
    --out-dir      experiments/motionllm_corrective_v1 \
    --epochs 50 --batch-size 8 --patience 10 \
    --scheduler onecycle --lr 1e-5 --max-lr 5e-5 \
    --augmentations speed \
    --device cuda:0 --seed 42
```

Defaults already select the in-domain VQ-VAE, the pretrained starting
checkpoint, the per-sample 5-variant labels (`../labels/labels_5var_reusable.json`,
the **same label set the VLM track uses**) and the pooled class labels used as
a fallback / BERTScore reference pool (`../labels/class_labels_pooled_150.json`).
Training writes `splits.json`, `motionllm_corrective_best.pth`, a training
curve and `run.log` to `--out-dir`.

### Recipe that produced the shipped model

| Setting | Value |
|---|---|
| VQ-VAE | in-domain (`vqvae_indomain_best.pth`), no augmentation, no discriminator |
| Motion augmentation | `speed` (temporal resampling ±20 %) |
| Labels | per-sample 5-variant (`labels_5var_reusable.json`) — identical to the VLM track |
| Optimizer | AdamW |
| Scheduler | OneCycleLR (lr 1e-5 → max 5e-5 → 5e-9, pct_start=0.3) |
| Epochs / batch | 50 / 8 |
| LoRA r / α / dropout | 16 / 32 / 0.05 |

Findings from the hyper-parameter search: the in-domain VQ-VAE beats the base
HumanML3D one; `speed` is the best single augmentation (adding mirror/reverse
does not help).

## Results (summary)

| Metric | Value |
|---|---|
| In-distribution accuracy (BERTScore classifier, 55 videos) | **78.2 %** |
| Out-of-distribution accuracy (19 videos) | **10.5 %** |
| Mean BERTScore F1 vs reference labels (in-dist) | 84.3 % |

This checkpoint is trained on the **same per-sample labels as the VLM track**
(`labels_5var_reusable.json`), so it is a like-for-like motion-only comparison
against the VLM models in `../docs/results.md`. A pooled-label variant
(max-LR 3e-5, 40 epochs) trades a little in-distribution accuracy (76.4 %) for
better OOD (21.1 %), but is not directly comparable to the VLMs because of the
different label set. The large in-dist → OOD gap is driven by monocular
pose-estimation noise on unseen subjects/angles and VQ-VAE codebook shift; with
no visual channel there is nothing to fall back on.
