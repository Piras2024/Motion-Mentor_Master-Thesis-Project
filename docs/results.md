# Exercise-Form Feedback — Experiments and Results

This document presents the consolidated results across all evaluation methods,
for both the **VLM track** (video) and the **motion-only track**
(`motionllm_corrective_v1`, no video).

---

## 1. Experimental setup

### 1.1 Dataset

- **Training source**: `/deck/users/mpiras/dataset/rdls` and `/deck/users/mpiras/dataset/squat_micc`
- **Total labeled videos**: 558 (11 classes × ~50 videos each, stratified)
- **Split**: 447 train / 55 val / 55 test (stratified, seed=42)
- **OOD set**: `/deck/users/mpiras/dataset/out_of_distribution/` — 19 videos with hand-labelled ground truth

Splits are persisted to `{output_dir}/split.json` at training time and read
back by the evaluators, so the test set a model is scored on is exactly the one
it never trained on. The OOD videos live in a separate directory that is never
scanned during training.

### 1.2 Training labels

- `labels_5var_reusable.json`: 558 entries, 5 text variants per video file. One variant is sampled per epoch per video (data augmentation on the response side). Used by both the VLM track and the motion-only model.

### 1.3 Matched recipe

After discovering that the original `finetune_gemma4.py` and `finetune_qwen2vl.py` used 15 epochs with no scheduler while the motion variants used 20 epochs + OneCycleLR (unfair comparison), we re-trained all base models with the matched recipe:

| Setting | Value |
|---|---|
| Epochs | 20 |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| Scheduler | OneCycleLR (pct_start=0.1) |
| Gradient accumulation | 8 |
| Batch size | 1 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | language model q/k/v/o + gate/up/down projections |
| dtype | bfloat16 |
| Frame size | 336×336 (Qwen) or native (Gemma4 video processor) |
| N frames sampled | 12 (Qwen) / 32 native (Gemma4) |

All v2/v3 checkpoints use the matched recipe. The motion-only model uses its own
recipe (see `motionllm/README.md`): 50 epochs, OneCycleLR max-LR 5e-5, `speed`
augmentation.

### 1.4 In-domain VQ-VAE

The motion variants (both tracks) use a fine-tuned VQ-VAE — shipped as
`motion_encoder/checkpoints/vqvae_indomain_best.pth` — adapted from the
HumanML3D-trained checkpoint to the exercise-form data.

### 1.5 Architectures evaluated

| ID | Backbone | Motion injection | Trainable params (motion side) |
|---|---|---|---|
| `gemma4_v3` | Gemma4-E2B-it | none | 0 |
| `qwen_v3` | Qwen2-VL-2B-Instruct | none | 0 |
| `gemma4_motion_proj_v2` | Gemma4-E2B-it | per-timestep `Linear(512→2048)` from frozen VQ-VAE codebook, projected vectors prepended as soft prompt tokens | ~1M |
| `qwen_motion_proj_v2` | Qwen2-VL-2B-Instruct | same as above | ~1M |
| `gemma4_qformer_v2` | Gemma4-E2B-it | 8 learnable queries cross-attend the full motion sequence via 1-layer Q-Former; output spliced at `<MQ_0>..<MQ_7>` slots | ~29M |
| `qwen_qformer_v2` | Qwen2-VL-2B-Instruct | same as above | ~29M |
| `motionllm_corrective_v1` | Gemma-2-2B (motion-only, no VLM) | VQ-VAE motion tokens fed to Gemma-2-2B; LoRA `m2t` adapter fine-tuned from the HumanML3D-pretrained MotionLLM. No video. (code: `motionllm/`) | LoRA r=16 + 514 token rows |

---

## 2. Evaluation methodology

Four complementary metrics are reported.

### 2.1 BERTScore classifier accuracy (BS acc)

The original eval pipeline: for each video, compare the model's response against all class-reference sentences from `class_labels_pooled_150.json` (10 sampled per class). The class with the highest max BERTScore F1 wins. Compare to ground-truth class → accuracy.

**Known weakness**: phrasing-sensitive. The model's response *"Bar drifting forward — fix the bar path"* (a correct identification of `rdl_hands_forward`) gets misclassified as `rdl_no_error` because the "bar" tokens have high F1 against the `rdl_no_error` refs ("bar close throughout", "bar stays on legs").

### 2.2 Real accuracy (manual audit)

Each of the 55 in-distribution responses and 19 OOD responses was read by hand. A response is counted "correct" if it identifies the actual form fault described by the ground-truth class, regardless of whether BERTScore could route it to the right class label.

This is the most trustworthy metric in this report. Where BS-acc and Real-acc differ, Real-acc is the truth. (The motion-only model is scored by its BERTScore classifier only — it was not manually audited.)

### 2.3 BERTScore F1 (response vs ground-truth references)

Pure text-similarity metric: compare the generated response against the ground-truth reference text using BERTScore F1 with `distilbert-base-uncased`.

- **In-dist**: F1 vs the 5 GT variants for that specific video (max and mean reported).
- **OOD**: F1 vs 10 sampled references for the true GT class (max and mean reported).

This is a generation-quality metric, not a classification metric. It can saturate (a model that produces text in the right "family" gets a high score even if it identifies the wrong fault).

### 2.4 Reference upper bound (LOO BERTScore)

For each test sample: take one GT reference as the "prediction" and compute BERTScore against the *other* GT references. Averaged across all such leave-one-out pairs, this gives the ceiling BERTScore can reach when the prediction *is* essentially a ground-truth sentence.

Models that score *above* this upper bound are producing text closer to the centroid of the reference cloud than the references are to each other. That's typical of a well-trained generator on this kind of multi-reference task — it doesn't mean the model is "better than ground truth", it means the metric has saturated.

### 2.5 OOD ground-truth label mapping

The OOD ground-truth file uses short class names that need to be mapped to the training class names:

| GT label (OOD) | Training class |
|---|---|
| `no_error` (squat) | `squat_no_errors` |
| `butt_wink` | `squat_butt_wink` |
| `depth_high` | `squat_depth_high` |
| `rdl_no_error` | `rdl_no_error` |
| `rdl_hands_forward` | `rdl_hands_forward` |

---

## 3. Results

### 3.1 In-distribution (55 videos)

The six VLM checkpoints are evaluated on the **same 55 held-out videos** (the test split saved at training time).

| Model | BS acc (Δ vs base) | Real acc (Δ vs base) | F1 max | F1 mean |
|---|---|---|---|---|
| **gemma4_v3** (base) | 83.6 % (46/55) | 92.7 % (51/55) | 0.8818 | 0.8485 |
| **qwen_v3** (base) | 85.5 % (47/55) | 90.9 % (50/55) | 0.8808 | 0.8454 |
| gemma4_motion_proj_v2 | 89.1 % (49/55) +5.5 | **96.4 % (53/55) +3.7** | **0.8857** | 0.8482 |
| qwen_motion_proj_v2 | **96.4 % (53/55) +10.9** | **96.4 % (53/55) +5.5** | 0.8833 | 0.8440 |
| gemma4_qformer_v2 | 92.7 % (51/55) +9.1 | 92.7 % (51/55) +0.0 | 0.8728 | 0.8403 |
| qwen_qformer_v2 | 90.9 % (50/55) +5.4 | 92.7 % (51/55) +1.8 | 0.8739 | 0.8453 |
| motionllm_corrective_v1 (motion only, no video)* | 78.2 % (43/55) | 78.2 % (43/55) | 0.8724 | 0.8382 |
| **Reference (LOO upper bound)** | — | — | **0.8716** | **0.8420** |

\* The motion-only model was evaluated on its own stratified 55-video split (same dataset, same protocol, different specific samples) and scored by its BERTScore classifier (no manual audit), so BS acc = Real acc here. The six VLM rows share the identical 55-video held-out set.

> **Shipped motion-only checkpoint.** This 78.2 % row is the model shipped as
> `checkpoints/motionllm_corrective_v1/best.pth` and documented in
> `motionllm/README.md` (per-sample labels `labels_5var_reusable.json`, max-LR
> 5e-5, 50 epochs). It is trained on the **same label set as the VLM track**,
> making it a like-for-like motion-only comparison: **78.2 % in-dist · 10.5 %
> OOD**. A pooled-label variant (max-LR 3e-5, 40 epochs) generalizes a little
> better (76.4 % in-dist · 21.1 % OOD) but uses a different label set, so it is
> not directly comparable to the VLM rows.

### 3.2 Out-of-distribution (19 videos)

| Model | BS acc (Δ vs base) | Real acc (Δ vs base) | F1 max | F1 mean |
|---|---|---|---|---|
| **gemma4_v3** (base) | **42.1 % (8/19)** | **42.1 % (8/19)** | 0.8488 | 0.7929 |
| **qwen_v3** (base) | 21.1 % (4/19) | 15.8 % (3/19) | 0.8012 | 0.7734 |
| gemma4_motion_proj_v2 | 21.1 % (4/19) −21.0 | 21.1 % (4/19) −21.0 | 0.8213 | 0.7821 |
| qwen_motion_proj_v2 | 15.8 % (3/19) −5.3 | 15.8 % (3/19) +0.0 | 0.8274 | 0.7850 |
| gemma4_qformer_v2 | **42.1 % (8/19) +0.0** | **42.1 % (8/19) +0.0** | **0.8513** | **0.8009** |
| qwen_qformer_v2 | 26.3 % (5/19) +5.2 | 21.1 % (4/19) +5.3 | 0.8297 | 0.7916 |
| motionllm_corrective_v1 (motion only, no video) | 10.5 % (2/19) | 10.5 % (2/19) | — | — |
| **Reference (LOO upper bound)** | — | — | **0.8565** | **0.8177** |

The motion-only model collapses on OOD (10.5 %) despite a strong 78.2 %
in-distribution score. With pose as its only input it has nothing to fall back
on when monocular pose estimation degrades on unseen subjects/angles. Per-class
it gets only `squat_butt_wink` partially right (2/7) and misses everything else.

### 3.3 Per-class OOD breakdown (real / manually-verified)

The `motionllm` column is BERTScore-classifier output (not manually audited);
all others are the manual audit.

```
                            gemma4_v3   qwen_v3   gemma4_mp   qwen_mp   gemma4_qf   qwen_qf   motionllm
squat_no_errors    (5)         2/5         0/5       0/5         0/5       4/5         3/5        0/5
squat_butt_wink    (7)         4/7         3/7       3/7         3/7       4/7         1/7        2/7
squat_depth_high   (4)         0/4         0/4       0/4         0/4       0/4         0/4        0/4
rdl_no_error       (2)         2/2         0/2       1/2         1/2       1/2         0/2        0/2
rdl_hands_forward  (1)         0/1         0/1       0/1         0/1       0/1         0/1        0/1
──────────────────────────────────────────────────────────────────────────────────────────────────────
Total              (19)        8 (42.1%)   3 (15.8%) 4 (21.1%)   3 (15.8%) 8 (42.1%)   4 (21.1%)  2 (10.5%)
```

### 3.4 Headline view (Real accuracy, in-dist · OOD)

| Architecture | Gemma4 (in-dist · OOD) | Qwen2-VL (in-dist · OOD) |
|---|---|---|
| Base (video only) | 92.7 % · **42.1 %** | 90.9 % · 15.8 % |
| Motion projection | **96.4 %** · 21.1 % | **96.4 %** · 15.8 % |
| Q-Former | 92.7 % · **42.1 %** | 92.7 % · **21.1 %** |

**Motion-only baseline** (`motionllm_corrective_v1`, Gemma-2-2B, **no video**):
78.2 % in-dist · 10.5 % OOD. Strong in-distribution but the weakest OOD
generalizer of all — confirming that the video signal, not the motion stream,
is what carries OOD robustness.

---

## 4. Findings

### 4.1 In-distribution

- **Motion projection is the in-dist winner on both backbones** (96.4 % real accuracy on Gemma4 and Qwen). Q-Former trails by ~3–4 pp; base trails by ~2–6 pp.
- **The video signal is necessary**: the motion-only model plateaus at 78.2 %, ~12 pp below the worst VLM variant. Motion alone is not enough.
- **BERTScore F1 cannot discriminate in-dist**: every model is at or above the LOO reference upper bound (0.87 max). The metric has saturated.

### 4.2 Out-of-distribution

- **The in-dist ranking flips completely on OOD.** The best in-dist model (qwen_motion_proj_v2 at 96.4 %) drops to 15.8 % on OOD.
- **Motion projection is a liability on OOD for Gemma4**: real accuracy falls from 42.1 % (base) to 21.1 % (motion_proj), a 21 pp drop. The projection has memorised the training motion manifold and falls apart on out-of-distribution motion features.
- **The motion-only model is the worst OOD generalizer (10.5 %)** — the same failure mode in its purest form: with no pixels, it is entirely at the mercy of pose-estimation drift on unseen subjects.
- **Q-Former preserves base-level OOD on Gemma4 and improves it on Qwen**: it's the only motion-augmented method that doesn't crash on OOD.
- **Gemma4 base is the best generalizer.** Tied with gemma4_qformer_v2 at 42.1 %, well above any motion-projection variant.

### 4.3 Universal failure modes (OOD)

- **`squat_depth_high`** is 0/4 across every architecture (including motion-only). A genuinely fine-grained perceptual task; OOD camera angles seem to make it unsolvable for these models.
- **`rdl_hands_forward`** is 0/1 universally — but n=1, so signal is weak.
- **`squat_no_errors` is catastrophic for Qwen** (0/5 on base, motion_proj) and the motion-only model (0/5). Both Qwen variants and the motion-only model over-predict errors on clean reps. Gemma4 Q-Former is the only model that handles "no error" videos well on OOD (4/5).

### 4.4 Metric-level observations

- **BERTScore F1 max is the cleanest single metric for OOD.** Ranking matches the manual audit. On the in-dist set the metric saturates and is unreliable.
- **Llama-3-8B-Instruct as a judge** was attempted; it under-counted by ~5–10 pp because responses that don't explicitly say "squat" or "RDL" cannot be assigned a prefix. We dropped Llama judging in the final tables in favour of the manual audit.
- **BERTScore classifier accuracy under-counts by 5–9 pp on the in-dist set** because of phrasing mismatches between training-label vocabulary and `class_labels_pooled_150.json` reference vocabulary (the "Bar drifting forward → rdl_no_error" misclassification pattern).

---

## 5. Open questions

1. **`squat_depth_high` is unsolved by every architecture on OOD (0/4 universal).** It may be a fundamentally fine-grained perception problem that OOD camera angles make unsolvable for models of this size.
2. **In-distribution N=55 may be too small to separate methods reliably.** Differences of 1-2 videos swing accuracy by 2-4 pp. Multi-seed runs would tighten conclusions but compute cost may not be worth it for a thesis.
3. **OOD N=19** is even smaller; pattern-level conclusions (Q-Former > motion_proj on OOD) hold but ranking among the bottom three models isn't statistically meaningful.

---

## 6. Reproducibility

VLM-track result files are at `/deck/users/mpiras/paligemma_exercise/`:

- `eval_results_*.json` — in-distribution generations + BERTScore classifications
- `ood_results_*.json` — OOD generations + BERTScore classifications
- `checkpoints_*/split.json` — exact train/val/test split per checkpoint (auto-discovered by eval)
- `results_summary.pdf` — the PDF version of the tables above

Motion-only result files are shipped under `checkpoints/motionllm_corrective_v1/`
(`splits.json`, `test_results.json`, `run.log`, `training_curves.png`).

### Training scripts

- `vlm_training/finetune_gemma4.py`, `finetune_qwen2vl.py` — base trainers (with matched recipe)
- `vlm_training/finetune_gemma4_motion_proj.py`, `finetune_qwen_motion_proj.py` — motion projection
- `vlm_training/finetune_gemma4_qformer.py`, `finetune_qwen_qformer.py` — Q-Former
- `motionllm/finetune_corrective.py` — motion-only model

### Evaluation scripts

- `evaluation/evaluate.py` — VLM in-dist evaluation (BERTScore classifier)
- `evaluation/evaluate_ood.py` — VLM OOD evaluation against ground-truth labels
- `motionllm/eval_corrective.py`, `motionllm/eval_ood.py` — motion-only evaluation
