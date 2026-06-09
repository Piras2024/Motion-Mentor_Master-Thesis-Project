# Demo

End-to-end inference on a single video.

## What's here

```
demo/
├── README.md            this file
├── run_demo.py          single-video inference
└── test_videos/
    ├── squat_butt_wink10.mp4
    ├── squat_no_errors10.mp4
    ├── rdl_hands_forward10.mp4
    └── guofeats/
        ├── HSMR-squat_butt_wink10_guofeats.npy
        ├── HSMR-squat_no_errors10_guofeats.npy
        └── HSMR-rdl_hands_forward10_guofeats.npy
```

The three videos come with their precomputed Guo features, so the demo
runs without needing the (large) HSMR installation.

## Run the demo

```bash
# Make sure git LFS has pulled the checkpoints
git lfs pull

# Recommended starting model: Qwen + Q-Former (best motion-injection)
python demo/run_demo.py \
    --video      demo/test_videos/squat_butt_wink10.mp4 \
    --checkpoint checkpoints/qwen_qformer_v2/best \
    --base-model qwen_qformer \
    --skip-hsmr
```

You should get something like:

```
DEMO RESULT
========================================================================

Video:    demo/test_videos/squat_butt_wink10.mp4
Response: Your pelvis is tilting posteriorly at the bottom of the squat.
          Brace your core and maintain a neutral lower back throughout.
Predicted class (BERTScore): squat_butt_wink
```

## Trying other architectures

`--base-model` accepts:

- `gemma4`, `qwen2vl`                — base (video only)
- `gemma4_motion_proj`, `qwen_motion_proj` — per-timestep motion projection
- `gemma4_qformer`, `qwen_qformer`   — Q-Former
- `motionllm`                        — **motion only, no video** (see below)

For the VLM models the `--checkpoint` path must match
(`checkpoints/{name}_v2_or_v3/best`), and the demo runs in the VLM/`paligemma`
conda env.

## Motion-only model (no video)

The `motionllm` base-model runs the video-free track: it feeds VQ-VAE motion
tokens straight into Gemma-2-2B (see `../motionllm/`). It uses the **same
precomputed guofeats** as everything else, but never looks at the pixels. Run
it in the `motionagent` conda env; `--checkpoint` defaults to the shipped
`checkpoints/motionllm_corrective_v1/best.pth`, so it can be omitted:

```bash
conda run -n motionagent python demo/run_demo.py \
    --video      demo/test_videos/squat_butt_wink10.mp4 \
    --base-model motionllm \
    --skip-hsmr
```

```
DEMO RESULT  (motion-only — no video)
========================================================================

Video:    squat_butt_wink10.mp4
Response: Your pelvis is tilting posteriorly at the bottom of the squat. Get
          your core braced and keep your lower back neutral all the way down.
Predicted class (BERTScore): squat_butt_wink
```

Add `--temperature 0.8 --top-p 0.9` for more varied phrasing (sampled instead
of greedy).

## Trying your own video

If you have a video with no precomputed guofeats:

1. Install HSMR (see `../hsmr/README.md` and the upstream HSMR repo).
2. Extract HSMR poses with `../pipeline/extract_hsmr.py`.
3. Convert to Guo features with `../pipeline/hsmr_to_guofeats.py`.
4. Place the resulting `_guofeats.npy` under `demo/test_videos/guofeats/`
   and run `run_demo.py` without `--skip-hsmr`.
