"""
End-to-end demo: video → form-feedback text.

Two modes:
  --skip-hsmr  : use the precomputed guofeats shipped with the repo (fast)
  (default)    : extract HSMR poses from the video first (requires HSMR
                 conda env + body models + hsmr.ckpt; see hsmr/README.md)

There are two model families, each needing its own conda env:

  * VLM models (video, run in the `paligemma`/VLM env):
        gemma4, qwen2vl, gemma4_motion_proj, qwen_motion_proj,
        gemma4_qformer, qwen_qformer

  * Motion-only model (no video, run in the `motionagent` env):
        motionllm  — VQ-VAE motion tokens → Gemma-2-2B (see motionllm/)

Usage:
    # VLM demo — uses precomputed guofeats in demo/test_videos/guofeats
    python demo/run_demo.py \
        --video demo/test_videos/squat_butt_wink10.mp4 \
        --checkpoint checkpoints/qwen_qformer_v2/best \
        --base-model qwen_qformer \
        --skip-hsmr

    # Motion-only demo (no video signal at all). --checkpoint defaults to the
    # shipped motionllm_corrective_v1/best.pth, so it can be omitted.
    conda run -n motionagent python demo/run_demo.py \
        --video demo/test_videos/squat_butt_wink10.mp4 \
        --base-model motionllm \
        --skip-hsmr
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motion_encoder"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import numpy as np
import torch

# ---------------------------------------------------------------------------

def find_guofeats(video_path: Path, skip_hsmr: bool) -> Path:
    """Locate guofeats for the input video.

    If --skip-hsmr was passed, look in demo/test_videos/guofeats/.
    Otherwise the user is expected to have extracted features via HSMR
    and placed them at pipeline/output/<stem>_guofeats.npy
    """
    stem = video_path.stem
    if skip_hsmr:
        shipped = video_path.parent / "guofeats" / f"HSMR-{stem}_guofeats.npy"
        if shipped.exists():
            return shipped
        raise FileNotFoundError(
            f"--skip-hsmr was set but no precomputed guofeats at {shipped}.\n"
            f"Run without --skip-hsmr to extract features, or provide them yourself."
        )
    # Otherwise expect them at pipeline/output/
    out = REPO_ROOT / "pipeline" / "output" / f"{stem}_guofeats.npy"
    if not out.exists():
        raise FileNotFoundError(
            f"Expected guofeats at {out}. Run extract_hsmr.py + hsmr_to_guofeats.py "
            f"or rerun with --skip-hsmr."
        )
    return out


def _motionllm_classify(response: str, device: str, n_refs: int = 10) -> str:
    """BERTScore-classify a motion-only response against the pooled refs."""
    import random
    from bert_score import score as bert_score_fn

    raw = json.load(open(REPO_ROOT / "labels" / "class_labels_pooled_150.json"))
    classes = list(raw.keys())
    cands, refs, cls_idx = [], [], []
    for ci, cls in enumerate(classes):
        for r in random.sample(raw[cls], min(n_refs, len(raw[cls]))):
            cands.append(response); refs.append(r); cls_idx.append(ci)
    _, _, F1 = bert_score_fn(cands, refs, model_type="distilbert-base-uncased",
                             device=device, verbose=False)
    scores = [0.0] * len(classes)
    for f1, ci in zip(F1.tolist(), cls_idx):
        scores[ci] = max(scores[ci], f1)
    return classes[max(range(len(classes)), key=lambda c: scores[c])]


def run_motionllm_demo(args):
    """Motion-only path: VQ-VAE tokens -> Gemma-2-2B, no video.

    Runs entirely from the self-contained motionllm/ package; must be launched
    in the `motionagent` conda env.
    """
    import numpy as np
    import torch

    # Make the motionllm/ package win for `models` / `options` imports.
    sys.path.insert(0, str(REPO_ROOT / "motionllm"))
    for m in list(sys.modules):
        if m == "models" or m.startswith("models.") or m == "options" or m.startswith("options."):
            del sys.modules[m]
    from models.mllm import MotionLLM
    from options.option_llm import get_args_parser as get_llm_args

    CORRECTIVE_INSTRUCTION = (
        "### Instruction:\nAnalyze this motion and identify the main technical "
        "error. Provide corrective feedback. If the technique is correct, "
        "confirm it.\n\n"
    )

    guofeats_path = find_guofeats(args.video, args.skip_hsmr)
    checkpoint = args.checkpoint or (
        REPO_ROOT / "checkpoints" / "motionllm_corrective_v1" / "best.pth"
    )

    # Build MotionLLM args (the trained model uses r=16, alpha=32, dropout=0.05).
    _argv, sys.argv = sys.argv, sys.argv[:1]
    llm_args = get_llm_args()
    sys.argv = _argv
    llm_args.device         = args.device
    llm_args.vq_path        = str(args.vqvae_ckpt)
    llm_args.lora_r_m2t     = 16
    llm_args.lora_alpha_m2t = 32
    llm_args.lora_dropout   = 0.05

    print(f"Loading motionllm from {checkpoint} ...")
    print("(For full eval over a test set, use motionllm/eval_corrective.py instead.)\n")
    model = MotionLLM(llm_args)
    model.load_model(str(checkpoint))
    model.llm.eval()
    model.to(args.device)

    motion = np.load(guofeats_path).astype(np.float32)
    do_sample = (args.temperature != 1.0) or (args.top_p != 1.0)
    response = model.caption(motion, instruction=CORRECTIVE_INSTRUCTION,
                             temperature=args.temperature, top_p=args.top_p,
                             do_sample=do_sample)

    print("Running BERTScore classification...")
    pred_class = _motionllm_classify(response, args.device)

    print("\n" + "=" * 70)
    print("DEMO RESULT  (motion-only — no video)")
    print("=" * 70)
    print(f"\nVideo:    {args.video.name}")
    print(f"Response: {response}")
    print(f"Predicted class (BERTScore): {pred_class}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True, type=Path)
    parser.add_argument("--checkpoint", required=False, type=Path, default=None,
                        help="VLM: path to checkpoints/{model}/best. "
                             "motionllm: path to a .pth (defaults to the shipped "
                             "checkpoints/motionllm_corrective_v1/best.pth).")
    parser.add_argument("--base-model", required=True,
                        choices=["gemma4", "qwen2vl",
                                 "gemma4_motion_proj", "qwen_motion_proj",
                                 "gemma4_qformer", "qwen_qformer",
                                 "motionllm"])
    parser.add_argument("--skip-hsmr",  action="store_true",
                        help="Use the precomputed guofeats shipped under demo/test_videos/guofeats")
    parser.add_argument("--vqvae-ckpt", type=Path,
                        default=REPO_ROOT / "motion_encoder" / "checkpoints" / "vqvae_indomain_best.pth")
    parser.add_argument("--device",      default="cuda:0",
                        help="Device for the motionllm path (VLM path uses device_map=auto)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p",       type=float, default=1.0)
    args = parser.parse_args()

    # Motion-only model takes a completely separate, video-free path.
    if args.base_model == "motionllm":
        run_motionllm_demo(args)
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required for VLM base-models "
                     "(e.g. checkpoints/qwen_qformer_v2/best)")

    # The evaluation script does everything we need (model loading, motion
    # encoding, inference, response decoding). Call it programmatically.
    # The simplest reliable path is to write a single-video split.json and
    # invoke evaluate.py — but for a demo we set up the model in-process.

    import evaluate as ev   # from REPO_ROOT/evaluation/

    print(f"Loading {args.base_model} from {args.checkpoint} ...")
    print(f"(For full eval over a test set, use evaluation/evaluate.py instead.)\n")

    # Build a tiny synthetic test_samples list of one element so we can
    # reuse the evaluate.py setup.
    sys.argv = [
        "evaluate.py",
        "--base-model", args.base_model,
        "--checkpoint", str(args.checkpoint),
        "--output",     "/tmp/_demo_out.json",
        "--vqvae-ckpt", str(args.vqvae_ckpt),
        "--temperature", str(args.temperature),
        "--top-p",       str(args.top_p),
        # the split flag — we'll point to a single-video split we synthesize
        "--split",       "/tmp/_demo_split.json",
    ]

    # synthesize the single-video split
    fname = args.video.name
    cls_guess = "demo"  # not used for inference, only for accounting
    with open("/tmp/_demo_split.json", "w") as f:
        json.dump({"test": [[str(args.video.resolve()), cls_guess]]}, f)

    # Override GUOFEATS_DIR so the eval script finds our guofeats
    guo_dir = find_guofeats(args.video, args.skip_hsmr).parent
    ev.GUOFEATS_DIR = str(guo_dir)

    # evaluate.py's main() runs the full pipeline
    ev.main()

    # Show only the prediction
    out = json.load(open("/tmp/_demo_out.json"))
    print("\n" + "=" * 70)
    print("DEMO RESULT")
    print("=" * 70)
    for r in out["results"]:
        print(f"\nVideo:    {r['video']}")
        print(f"Response: {r['response']}")
        print(f"Predicted class (BERTScore): {r['predicted_class']}\n")


if __name__ == "__main__":
    main()
