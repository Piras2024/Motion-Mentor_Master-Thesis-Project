# Evaluation

Two scripts:

| Script | Purpose |
|---|---|
| `evaluate.py`     | In-distribution: uses the test split saved at training time. |
| `evaluate_ood.py` | Out-of-distribution: against an externally-labelled video set with ground-truth class names. |

## How `evaluate.py` works

1. Auto-discovers `{checkpoint}/../split.json` (created by the training
   script). Uses its `test` list as the evaluation set.
2. Loads the VLM + LoRA adapter + (for motion variants) the motion
   projection or Q-Former weights and the in-domain VQ-VAE.
3. Generates a textual form-feedback response for each video.
4. **BERTScore classification** — every response is scored against the
   reference sentences of every class
   (`../labels/class_labels_pooled_150.json`); the class whose top-F1 is
   highest wins. Compared to ground truth → per-class and overall accuracy.

```bash
python evaluation/evaluate.py \
    --base-model qwen_qformer \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --output     eval_results.json
```

## How `evaluate_ood.py` works

Same flow but the test set is built from an OOD ground-truth JSON
(`{video.mp4: short_label}`). Short labels are mapped to training class
names:

| Short GT label | Training class |
|---|---|
| `no_error`          | `squat_no_errors` |
| `butt_wink`         | `squat_butt_wink` |
| `depth_high`        | `squat_depth_high` |
| `rdl_no_error`      | `rdl_no_error` |
| `rdl_hands_forward` | `rdl_hands_forward` |

```bash
python evaluation/evaluate_ood.py \
    --base-model  qwen_qformer \
    --checkpoint  checkpoints/qwen_qformer_v2/best \
    --video-dir   /path/to/ood/videos \
    --guofeats-dir /path/to/ood/guofeats \
    --ground-truth /path/to/ood/ground_truth.json \
    --output      ood_results.json
```

## A note on BERTScore vs. manual evaluation

The BERTScore classifier is phrasing-sensitive: some correct responses get
misrouted because their vocabulary doesn't overlap with the chosen
references. In our experiments, manually auditing the responses
(`real accuracy`) was 3–10 percentage points higher than the BS-classifier
accuracy on most models. The thesis's headline numbers use the manual audit.
