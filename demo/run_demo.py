"""
End-to-end demo: video → form-feedback text.

Two modes:
  --skip-hsmr  : use the precomputed guofeats shipped with the repo (fast)
  (default)    : extract HSMR poses from the video first (requires HSMR
                 conda env + body models + hsmr.ckpt; see hsmr/README.md)

Usage:
    # Fastest path — uses precomputed guofeats in demo/test_videos/guofeats
    python demo/run_demo.py \
        --video demo/test_videos/squat_butt_wink10.mp4 \
        --checkpoint checkpoints/qwen_qformer_v2/best \
        --base-model qwen_qformer \
        --skip-hsmr

    # With base-model=gemma4_qformer or any of:
    #   {gemma4, qwen2vl, gemma4_motion_proj, qwen_motion_proj,
    #    gemma4_qformer, qwen_qformer}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to checkpoint/{model}/best  e.g. checkpoints/qwen_qformer_v2/best")
    parser.add_argument("--base-model", required=True,
                        choices=["gemma4", "qwen2vl",
                                 "gemma4_motion_proj", "qwen_motion_proj",
                                 "gemma4_qformer", "qwen_qformer"])
    parser.add_argument("--skip-hsmr",  action="store_true",
                        help="Use the precomputed guofeats shipped under demo/test_videos/guofeats")
    parser.add_argument("--vqvae-ckpt", type=Path,
                        default=REPO_ROOT / "motion_encoder" / "checkpoints" / "vqvae_indomain_best.pth")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p",       type=float, default=1.0)
    args = parser.parse_args()

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
