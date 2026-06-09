# Strength-Training Form-Feedback VLM

End-to-end pipeline that takes a short video of a strength-training movement
(squat or Romanian deadlift) and produces natural-language form feedback
identifying the exercise and any execution error.

The system combines:

1. **HSMR** — 3D body pose reconstruction from video.
2. **VQ-VAE motion encoder** — domain-adapted from HumanML3D, encodes pose
   sequences into discrete motion tokens.
3. **Vision-language model** — fine-tuned with LoRA on either Gemma4-E2B-it
   or Qwen2-VL-2B-Instruct, conditioned on the video frames and (optionally)
   the motion tokens.

## Repository layout

```
.
├── hsmr/               3D pose extraction codebase (see hsmr/README.md)
├── motion_encoder/     VQ-VAE for the motion stream (Guo features → tokens)
├── pipeline/           Glue: raw mp4 → HSMR → guofeats → motion tokens
├── vlm_training/       6 finetune scripts (3 architectures × 2 backbones)
├── evaluation/         In-distribution and OOD evaluation
├── checkpoints/        Best LoRA adapters per (backbone × architecture)
├── demo/               End-to-end demo with 3 test videos
├── labels/             Class labels and reference sentences
└── docs/               Results write-up
```

## Architectures evaluated

| ID | Backbone | Motion injection |
|---|---|---|
| `gemma4_v3`            | Gemma4-E2B-it    | none (base) |
| `qwen_v3`              | Qwen2-VL-2B-Instruct | none (base) |
| `gemma4_motion_proj_v2`| Gemma4-E2B-it    | per-timestep `Linear(512→2048)` from VQ-VAE codebook |
| `qwen_motion_proj_v2`  | Qwen2-VL-2B-Instruct | same as above |
| `gemma4_qformer_v2`    | Gemma4-E2B-it    | 8-query Q-Former cross-attending the full motion sequence |
| `qwen_qformer_v2`      | Qwen2-VL-2B-Instruct | same as above |

See [docs/results.md](docs/results.md) for the full results table.

## Quick start: run the demo

```bash
# Install
conda create -n vlm python=3.10
conda activate vlm
pip install -r requirements.txt
git lfs pull       # pull the checkpoints

# Demo on a shipped video (no HSMR extraction needed)
python demo/run_demo.py \
    --video demo/test_videos/squat_butt_wink10.mp4 \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --base-model qwen_qformer \
    --skip-hsmr
```

## Reproducing the training

See `vlm_training/README.md` for the matched training recipe and one-line
commands for each architecture. Note that you need:

- the in-domain VQ-VAE (shipped under `motion_encoder/checkpoints/`),
- a labelled video dataset (see `labels/` for the structure),
- pre-extracted Guo features for every video (see `pipeline/` and `hsmr/`).

## Reproducing the evaluation

```bash
python evaluation/evaluate.py \
    --base-model qwen_qformer \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --output eval_results.json
```

For OOD evaluation, see `evaluation/evaluate_ood.py`.

## Results

See [docs/results.md](docs/results.md).

## Acknowledgements

- **HSMR** — 3D body-pose reconstruction (see `hsmr/LICENSE`).
- **motion-agent / T2M-GPT** — VQ-VAE motion encoder architecture.
- **HuggingFace transformers + PEFT** — VLM backbones and LoRA.

## License

MIT (see `LICENSE`).
