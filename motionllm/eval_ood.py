"""
eval_ood.py — Evaluate a MotionLLM checkpoint on OOD videos with GT labels.

Usage (run from inside motionllm/; checkpoint, VQ-VAE and refs default to the
shipped best model):
  python eval_ood.py \
    --npy-dir /path/to/ood_guofeats \
    --gt-json /path/to/ground_truth.json \
    --use-bertscore --device cuda:0
"""

import sys, os, json, argparse, random
from collections import defaultdict
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from models.mllm import MotionLLM
from options.option_llm import get_args_parser as get_llm_args
from finetune_corrective import get_instruction
from eval_corrective import bertscore_classify, detect_class

GT_LABEL_MAP = {
    "no_error":        "squat_no_errors",
    "butt_wink":       "squat_butt_wink",
    "depth_high":      "squat_depth_high",
    "rdl_no_error":    "rdl_no_error",
    "rdl_hands_forward": "rdl_hands_forward",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir",    default="pipeline_out/ood",
                        help="Directory of {id}_guofeats.npy (or HSMR-{id}.npy) OOD features")
    parser.add_argument("--gt-json",    required=True,
                        help="JSON dict {video.mp4: short_label} of OOD ground truth")
    parser.add_argument("--ckpt",       default="../checkpoints/motionllm_corrective_v1/best.pth")
    parser.add_argument("--vqvae-ckpt", default="../motion_encoder/checkpoints/vqvae_indomain_best.pth")
    parser.add_argument("--lora-r",     type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--use-bertscore", action="store_true")
    parser.add_argument("--bertscore-refs",
                        default="../labels/class_labels_pooled_150.json")
    parser.add_argument("--bertscore-n-refs", type=int, default=10)
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p",       type=float, default=1.0)
    parser.add_argument("--do-sample",   action="store_true")
    args = parser.parse_args()

    # Load GT labels
    with open(args.gt_json) as f:
        raw_gt = json.load(f)
    # Map short label → full class name
    gt = {vid.replace(".mp4", ""): GT_LABEL_MAP.get(lbl, lbl)
          for vid, lbl in raw_gt.items()}

    # Build list of (sample_id, npy_path, gt_cls)
    samples = []
    for sample_id, gt_cls in gt.items():
        # Prefer precomputed guofeats; fall back to HSMR raw npy
        npy_path = os.path.join(args.npy_dir, f"{sample_id}_guofeats.npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(args.npy_dir, f"HSMR-{sample_id}.npy")
        if not os.path.exists(npy_path):
            print(f"  MISSING npy: {sample_id}")
            continue
        samples.append((sample_id, npy_path, gt_cls))
    print(f"OOD samples found: {len(samples)}/{len(gt)}")

    # Load model
    _argv, sys.argv = sys.argv, sys.argv[:1]
    llm_args = get_llm_args()
    sys.argv = _argv
    llm_args.device         = args.device
    llm_args.vq_path        = args.vqvae_ckpt
    llm_args.lora_r_m2t     = args.lora_r
    llm_args.lora_alpha_m2t = args.lora_alpha
    llm_args.lora_dropout   = args.lora_dropout

    print("Building MotionLLM …")
    model = MotionLLM(llm_args)
    model.load_model(args.ckpt)
    model.llm.eval()
    model.to(args.device)

    # BERTScore refs
    bs_class_refs = None
    if args.use_bertscore:
        raw_refs = json.load(open(args.bertscore_refs))
        bs_class_refs = {cls: random.sample(refs, min(args.bertscore_n_refs, len(refs)))
                         for cls, refs in raw_refs.items()}
        print(f"BERTScore: {len(bs_class_refs)} classes, {args.bertscore_n_refs} refs each")

    # Inference
    raw_results = []
    for sample_id, npy_path, gt_cls in samples:
        motion = np.load(npy_path).astype(np.float32)
        pred = model.caption(motion, instruction=get_instruction(sample_id),
                             temperature=args.temperature, top_p=args.top_p,
                             do_sample=args.do_sample)
        raw_results.append((sample_id, gt_cls, pred))
        print(f"  [{sample_id}] gt={gt_cls} | pred={pred[:80]}…")

    # Classify
    predictions = [r[2] for r in raw_results]
    if args.use_bertscore:
        pred_classes = bertscore_classify(predictions, bs_class_refs,
                                          bs_model=args.bertscore_model,
                                          device=args.device)
    else:
        pred_classes = [detect_class(p) for p in predictions]

    # Report
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    print("\n" + "=" * 70)
    for (sample_id, gt_cls, pred), pred_cls in zip(raw_results, pred_classes):
        correct = (pred_cls == gt_cls)
        class_total[gt_cls]   += 1
        class_correct[gt_cls] += (1 if correct else 0)
        mark = "✓" if correct else "✗"
        print(f"  {mark} [{sample_id}]  GT={gt_cls}  PRED={pred_cls}")
        if not correct:
            print(f"      pred text: {pred[:100]}")

    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    print("\n" + "=" * 70)
    print(f"{'Class':<25} {'Corr':>5} {'Tot':>5} {'Acc':>8}")
    print("-" * 50)
    for cls in sorted(class_total):
        n, d = class_correct[cls], class_total[cls]
        print(f"{cls:<25} {n:>5} {d:>5} {n/d*100:>7.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_correct:>5} {total_samples:>5} "
          f"{total_correct/total_samples*100:>7.1f}%")


if __name__ == "__main__":
    main()
