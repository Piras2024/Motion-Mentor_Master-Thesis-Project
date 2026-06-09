# Strength-Training Form Feedback — Video vs. Motion

Given a short video of a strength-training movement (squat or Romanian
deadlift), the system produces natural-language form feedback identifying the
exercise and any execution error (e.g. *"your pelvis is tucking under at the
bottom — that's a butt wink"*).

The repository studies **one question through two model families**: *how much of
this task actually needs the video, versus the 3D motion alone?*

Both families share the same front-end pipeline:

**HSMR** (3D body-pose reconstruction) → **Guo features** → **in-domain VQ-VAE**
(domain-adapted from HumanML3D) → discrete **motion tokens**.

They differ in what the language model sees:

| Track | Code | Input to the LM | Backbone |
|---|---|---|---|
| **VLM** (video) | `vlm_training/`, `evaluation/` | video frames + *optionally* motion tokens | Gemma4-E2B-it / Qwen2-VL-2B-Instruct |
| **Motion-only** (no video) | `motionllm/` | motion tokens only — **no pixels** | Gemma-2-2B |

The headline finding: the motion-only model is usable in-distribution (78.2 %,
though below the 90–96 % VLMs) but collapses out-of-distribution (10.5 %), while
the best VLM holds up far better (42.1 % OOD) — the video signal is what
generalizes. See [docs/results.md](docs/results.md).

## Repository layout

```
.
├── hsmr/               3D pose extraction codebase (see hsmr/README.md)
├── motion_encoder/     VQ-VAE for the motion stream (Guo features → tokens)
├── pipeline/           Glue: raw mp4 → HSMR → guofeats → motion tokens
├── vlm_training/       VLM track — 6 finetune scripts (3 architectures × 2 backbones)
├── motionllm/          Motion-only track — VQ-VAE tokens → Gemma-2-2B (no video)
├── evaluation/         VLM in-distribution and OOD evaluation
├── checkpoints/        Trained checkpoints for both tracks
├── demo/               End-to-end demo with 3 test videos (both tracks)
├── labels/             Class labels and reference sentences (shared by both tracks)
└── docs/               Results write-up
```

## Models

### VLM track (video)

| ID | Backbone | Motion injection |
|---|---|---|
| `gemma4_v3`            | Gemma4-E2B-it    | none (base) |
| `qwen_v3`              | Qwen2-VL-2B-Instruct | none (base) |
| `gemma4_motion_proj_v2`| Gemma4-E2B-it    | per-timestep `Linear(512→2048)` from VQ-VAE codebook |
| `qwen_motion_proj_v2`  | Qwen2-VL-2B-Instruct | same as above |
| `gemma4_qformer_v2`    | Gemma4-E2B-it    | 8-query Q-Former cross-attending the full motion sequence |
| `qwen_qformer_v2`      | Qwen2-VL-2B-Instruct | same as above |

### Motion-only track (no video)

| ID | Backbone | Input |
|---|---|---|
| `motionllm_corrective_v1` | Gemma-2-2B | VQ-VAE motion tokens only — no video (see [motionllm/README.md](motionllm/README.md)) |

## Headline results

Real accuracy (manually audited for the VLM rows), best model per track:

| Model | In-distribution | Out-of-distribution |
|---|---|---|
| `gemma4_v3` (best generalizer, video) | 92.7 % | **42.1 %** |
| `qwen_motion_proj_v2` (best in-dist, video) | **96.4 %** | 15.8 % |
| `motionllm_corrective_v1` (motion only, no video) | 78.2 % | 10.5 % |

Full cross-model tables in [docs/results.md](docs/results.md).

## Environments

The two tracks use different `transformers` versions and are best kept in
separate conda envs:

| Env | Used by | `transformers` |
|---|---|---|
| VLM env (e.g. `vlm`) | `vlm_training/`, `evaluation/`, VLM demo | ≥ 5.x (Gemma4 / Qwen2-VL) |
| Motion env (e.g. `motionagent`) | `motionllm/`, motion-only demo | 4.44.x (Gemma-2-2B + VQ-VAE code) |

```bash
conda create -n vlm python=3.10
conda activate vlm
pip install -r requirements.txt
git lfs pull       # pull the checkpoints (both tracks)
```

See `motionllm/README.md` for the motion-only env. The base LLM/VLM weights are
pulled from HuggingFace on first run.

## Quick start: run the demo

Both demos use the precomputed Guo features shipped with the 3 test videos, so
no HSMR installation is needed (`--skip-hsmr`).

```bash
# VLM track (video) — in the VLM env
python demo/run_demo.py \
    --video demo/test_videos/squat_butt_wink10.mp4 \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --base-model qwen_qformer \
    --skip-hsmr

# Motion-only track (no video) — in the motion env; checkpoint defaults to the
# shipped motionllm_corrective_v1/best.pth
conda run -n motionagent python demo/run_demo.py \
    --video demo/test_videos/squat_butt_wink10.mp4 \
    --base-model motionllm \
    --skip-hsmr
```

See [demo/README.md](demo/README.md) for all options.

## Reproducing training

- **VLM track** — `vlm_training/README.md` has the matched recipe and one-line
  command per architecture.
- **Motion-only track** — `motionllm/README.md` has the recipe for
  `motionllm_corrective_v1`.

Both need: the in-domain VQ-VAE (shipped under `motion_encoder/checkpoints/`),
a labelled video dataset (see `labels/`), and pre-extracted Guo features per
video (see `pipeline/` and `hsmr/`).

## Reproducing evaluation

```bash
# VLM track (in-distribution)
python evaluation/evaluate.py \
    --base-model qwen_qformer \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --output eval_results.json
# OOD: evaluation/evaluate_ood.py

# Motion-only track (run from inside motionllm/)
cd motionllm
python eval_corrective.py \
    --splits ../checkpoints/motionllm_corrective_v1/splits.json \
    --use-bertscore --device cuda:0
# OOD: motionllm/eval_ood.py
```

## Results

See [docs/results.md](docs/results.md).

## Acknowledgements

- **HSMR** — 3D body-pose reconstruction (see `hsmr/LICENSE`).
- **motion-agent / T2M-GPT** — VQ-VAE motion encoder and MotionLLM architecture.
- **HuggingFace transformers + PEFT** — Gemma / Qwen backbones and LoRA.

## License

The **original code in this repository** (the training/eval scripts, the demo,
and the docs) is released under the **MIT License** — see [LICENSE](LICENSE).

This does **not** relicense the bundled third-party code, body models, datasets,
or model weights, which retain their own terms. Several are **non-commercial /
research-only**, so the repository **as a whole is for non-commercial scientific
research use**:

| Component | License / terms |
|---|---|
| `hsmr/thirdparty/SKEL/` (body model) | Max-Planck **non-commercial research only**; redistribution restricted |
| `hsmr/` (HSMR) | MIT |
| `hsmr/detectron2/` | Apache-2.0 |
| Gemma-2 / Gemma4 weights | **Google Gemma Terms** (use restrictions apply) |
| Qwen2-VL-2B weights | Apache-2.0 |
| `motionllm_base/pretrained.pth` (T2M-GPT / HumanML3D-derived) | research-use; see upstream T2M-GPT and AMASS/HumanML3D terms |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details. If you intend
any commercial use, you must obtain the appropriate licenses for the
non-commercial components (notably SKEL) directly from their owners.
