"""
eval_corrective.py

Evaluate a fine-tuned MotionLLM checkpoint on the held-out test split
produced by finetune_corrective.py.

Supports both label formats:
  - JSONL: {"id": "head_position1", "output": "..."}
  - JSON dict: {"head_position1.mp4": "..."}

Since the new LLM labels are unique per sample (not one fixed string per class),
exact-match accuracy is replaced with keyword-based class detection or BERTScore
classification (--use-bertscore).

Usage (run from inside motionllm/; defaults point at the shipped checkpoint,
in-domain VQ-VAE and pooled BERTScore references):
  python eval_corrective.py \
    --splits ../checkpoints/motionllm_corrective_v1/splits.json \
    --use-bertscore \
    --device cuda:0
"""

import sys
import os
import json
import argparse
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from models.mllm import MotionLLM
from options.option_llm import get_args_parser as get_llm_args
from finetune_corrective import CORRECTIVE_INSTRUCTION, _load_labels, get_instruction

# Keywords used to detect which error class a free-text prediction belongs to.
CLASS_KEYWORDS = {
    "squat_no_errors":  ["great work", "spot on", "well done", "good job", "no error", "nothing to correct",
                         "excellent squat", "good squat", "correct on all", "nothing to fix",
                         "clean squat", "form is correct", "squat looks good", "no corrections",
                         "squat is solid", "movement is correct", "looking great", "keep doing what you are doing"],
    "squat_hands_wide": ["grip is too wide", "hands are too far", "grip too wide",
                         "hands too wide", "hands are too wide", "hands are placed too wide",
                         "grip on the bar is too wide", "wide grip",
                         "bring your hands in", "bring your hands closer", "narrow your grip",
                         "hands in closer", "hands too far apart", "grip width",
                         "bar placement", "hand position on the bar", "slide your hands inward",
                         "upper back tightness", "tightness of your upper back",
                         "shoulder blades pull together", "width of your grip"],
    "squat_high_heel":  ["heel", "heels are coming", "heels come up", "heels lifting", "heels popping",
                         "ankle mobility", "foot contact", "elevating your heel",
                         "heels off", "heel rise", "ankle dorsiflexion",
                         "keep your feet flat", "feet flat", "feet planted",
                         "heels lift", "heels leave", "raising your heels",
                         "weight distributed across", "whole sole", "toes"],
    "squat_depth_high": ["not deep enough", "not squatting deep", "too shallow", "too high",
                         "above parallel", "reach parallel", "femurs", "squat depth",
                         "stopping too early", "shallow", "not reach", "below parallel",
                         "hips descend", "full depth",
                         "squat is too shallow", "not reaching adequate", "insufficient squat depth",
                         "hip crease", "hips need to drop", "sitting deep",
                         "hips sink to that position", "descend further",
                         "transition from down to up should happen at parallel",
                         "pushing the floor away", "squat is too high",
                         "reversing the movement before reaching parallel"],
    "squat_butt_wink":  ["butt wink", "pelvic tilt", "posterior pelvic",
                         "lumbar", "hip flexor", "hamstring mobility",
                         "pelvic tuck", "pelvis tilts",
                         "pelvis tucks", "pelvic tucking", "pelvis is tucking",
                         "neutral arch", "pelvic position",
                         "pelvis is tilting", "pelvis tilting", "posteriorly",
                         "tilting under", "rolling under", "pelvis rolls",
                         "pelvis rolling", "tucking under",
                         "lower back rounds at depth", "lower back rounding at",
                         "squatting with a rounded lower back"],
    "squat_head_position": ["head is not aligned", "head position", "looking too far down",
                         "looking straight down", "looking too much straight down",
                         "misaligns your head", "head misaligned", "head not in line",
                         "head in line with your torso", "head aligned with your torso",
                         "chin", "torso lean", "forward lean",
                         "look more in front", "look in front of you",
                         "looking down", "fix your gaze", "focal point",
                         "head is angled", "head position is causing",
                         "gaze too far down", "gaze forward", "eyes too far down",
                         "looking too far forward during the squat"],
    "rdl_no_error":     ["clean romanian", "romanian deadlift is correct", "romanian deadlift technique",
                         "romanian deadlift looks", "rdl looks", "no errors to point", "nothing to correct",
                         "hip hinge is clean", "hip hinge mechanics", "executed correctly",
                         "bar stays close", "well performed romanian", "well-performed romanian",
                         "well executed romanian", "romanian deadlift is solid",
                         "well performed rdl", "romanian deadlift form is",
                         "rdl looks solid", "nothing to fix", "great romanian",
                         "bar close, back flat", "hinge correct", "bar on your legs",
                         "bar is not drifting", "clean and controlled romanian",
                         "correct mechanics throughout", "performed exactly how it should",
                         "hip hinge is correct", "hip hinge is working",
                         "good romanian deadlift", "bar stays in contact with your legs",
                         "excellent romanian deadlift", "technically correct romanian",
                         "traveling vertically close to your legs",
                         "hamstrings are properly loaded", "bar is close, your back is flat",
                         "bar close, your back is flat"],
    "rdl_hands_forward": ["hands are drifting", "bar is moving away", "bar should stay",
                          "bar is drifting", "drifting forward", "drifting away from your legs",
                          "bar contact", "bar close to",
                          "hands moving away", "bar traveling away", "bar should graze",
                          "keep the bar close", "should remain in contact with your legs",
                          "bar in contact", "drifts away from your body",
                          "dragging along your body", "shaving your shins",
                          "bar dragging", "bar in contact with",
                          "bar drifting", "drifting away", "bar stays close to your legs",
                          "contact with your thighs", "contact with your legs",
                          "bar leaving your legs", "bar off your legs",
                          "bar drift", "bar path is"],
    "rdl_too_much_depth": ["too deep", "descending too much", "descending past", "descent is too long",
                            "going too deep", "descent too long", "past the correct end",
                            "depth is too much", "depth is excessive",
                            "goes too deep", "going past the point",
                            "hips have stopped moving", "stop earlier",
                            "descending too far", "going past", "lumbar spine",
                            "hips can no longer push back", "hips can no longer hinge",
                            "hips have hit their limit", "past the correct finishing",
                            "hips have pushed back as far", "stop when your hips",
                            "movement ends when your hips", "do not go past it",
                            "pushed back to their limit", "hip hinge gives out",
                            "hips pushing back controls the depth", "end the rep before",
                            "more range than this lift", "past that point forces",
                            "hips can no longer travel backward", "hip hinge can support",
                            "correct end point is when", "stop there, not past",
                            "end the movement when your hips"],
    "rdl_head_position": ["head is kept up during the rdl", "head kept up",
                          "keeping your head up", "head up and looking forward",
                          "head should remain in line with your torso",
                          "neck is extended", "looking too far forward",
                          "head should follow", "head in line with your spine",
                          "head and spine aligned", "head move naturally",
                          "head to move naturally", "allow your head to move",
                          "cervical spine", "gaze directed toward the floor",
                          "head position during the rdl", "neck hyperextended",
                          "passive extension of your spine", "head is tilted",
                          "head tilted", "chin lifted", "chin up",
                          "head be a passive", "extend that neutrality through your neck",
                          "final piece of your neutral spine", "head is elevated",
                          "head elevated", "your head should simply follow",
                          "wherever your spine is pointing"],
    "rdl_too_much_knee_bend": ["too much knee bend", "knees flex excessively",
                                "knees are bending too much", "excessive knee",
                                "knee bend in this romanian", "knee bend in the rdl",
                                "hip hinge movement", "knees should have only a slight bend",
                                "shins vertical", "knees nearly straight",
                                "legs nearly straight", "almost straight legs",
                                "knees bending too much", "knees bending throughout",
                                "set a soft bend", "soft bend", "knee angle",
                                "knees stay", "knee flexion", "knees flex",
                                "knees moving too much", "reverse the emphasis",
                                "hips not traveling back enough", "knees moving",
                                "knee bend is excessive"],
}


MULTI_ERROR_PHRASES = ["two error", "two correction", "three error", "three issue",
                        "two issue", "two mistake", "two problem"]


def all_mentioned_classes(text: str) -> set:
    """Return all error classes with at least 1 keyword match, exercise-aware.

    For RDL texts only RDL classes are considered; for squat texts only squat
    classes. Falls back to the other group when the primary group has no match.
    This prevents shared keywords (e.g. 'lower back rounds') from firing across
    exercise types.
    """
    t = text.lower()
    is_rdl = any(kw in t for kw in ["romanian deadlift", "romanian dead", "rdl", "hip hinge",
                                     "during this hinge", "throughout the hinge", "as you hinge",
                                     "the hinge"])

    rdl_classes   = ["rdl_no_error", "rdl_hands_forward", "rdl_too_much_depth",
                      "rdl_head_position", "rdl_too_much_knee_bend"]
    squat_classes = ["squat_no_errors", "squat_hands_wide", "squat_high_heel",
                     "squat_depth_high", "squat_butt_wink", "squat_head_position"]

    primary   = rdl_classes   if is_rdl else squat_classes
    secondary = squat_classes if is_rdl else rdl_classes

    result = {cls for cls in primary if any(kw in t for kw in CLASS_KEYWORDS[cls])}
    if not result:
        result = {cls for cls in secondary if any(kw in t for kw in CLASS_KEYWORDS[cls])}
    return result


def gt_label_classes(gt_text: str, primary_cls: str) -> set:
    """Return the set of error classes the GT label refers to.

    Always includes primary_cls. For labels that explicitly mention multiple
    errors ('two errors', 'two corrections', …) also includes secondary classes
    whose keywords appear in the text — limited to the same exercise group as
    primary_cls to avoid cross-exercise false positives.
    """
    classes = {primary_cls}
    t = gt_text.lower()
    if any(p in t for p in MULTI_ERROR_PHRASES):
        is_rdl  = primary_cls.startswith("rdl")
        exclude = {"squat_no_errors", "rdl_no_error"}
        same_exercise = (
            ["rdl_no_error", "rdl_hands_forward", "rdl_too_much_depth",
             "rdl_head_position", "rdl_too_much_knee_bend"]
            if is_rdl else
            ["squat_no_errors", "squat_hands_wide", "squat_high_heel",
             "squat_depth_high", "squat_butt_wink", "squat_head_position"]
        )
        for cls in same_exercise:
            if cls != primary_cls and cls not in exclude and any(kw in t for kw in CLASS_KEYWORDS[cls]):
                classes.add(cls)
    return classes


def detect_class(text: str) -> str:
    """Return the class with the most keyword matches, using exercise type to break ties.

    Counts how many keywords from each class appear in the text, then returns the
    class with the highest count. RDL classes are prioritised for RDL-labelled text,
    squat classes for everything else; the secondary group is only consulted when no
    primary class has any keyword match.
    """
    t = text.lower()
    is_rdl = any(kw in t for kw in ["romanian deadlift", "romanian dead", "rdl", "hip hinge"])

    rdl_classes   = ["rdl_no_error", "rdl_hands_forward", "rdl_too_much_depth",
                      "rdl_head_position", "rdl_too_much_knee_bend"]
    squat_classes = ["squat_no_errors", "squat_hands_wide", "squat_high_heel",
                     "squat_depth_high", "squat_butt_wink", "squat_head_position"]

    scores = {cls: sum(1 for kw in kws if kw in t) for cls, kws in CLASS_KEYWORDS.items()}

    primary   = rdl_classes   if is_rdl else squat_classes
    secondary = squat_classes if is_rdl else rdl_classes

    for group in [primary, secondary]:
        best_cls   = max(group, key=lambda c: scores[c])
        best_score = scores[best_cls]
        if best_score > 0:
            return best_cls

    return "unknown"


def bertscore_classify(predictions, class_refs, bs_model="distilbert-base-uncased", device="cpu"):
    """Classify each prediction to a class using BERTScore F1 against class references.

    class_refs: {class_name: [ref_sentence, ...]}
    Returns list of predicted class names (one per prediction).
    """
    from bert_score import score as bert_score_fn

    classes = list(class_refs.keys())
    n_preds = len(predictions)

    # Build flat (candidate, reference) pairs for one big batch call
    all_cands, all_refs, pred_idx, cls_idx = [], [], [], []
    for pi, pred in enumerate(predictions):
        for ci, cls in enumerate(classes):
            for ref in class_refs[cls]:
                all_cands.append(pred)
                all_refs.append(ref)
                pred_idx.append(pi)
                cls_idx.append(ci)

    print(f"  BERTScore: scoring {len(all_cands)} (pred, ref) pairs …")
    _, _, F1 = bert_score_fn(all_cands, all_refs, model_type=bs_model,
                              device=device, verbose=False)
    F1 = F1.tolist()

    # Aggregate: for each (pred, class) take the max F1 over refs
    scores = [[0.0] * len(classes) for _ in range(n_preds)]
    for f1, pi, ci in zip(F1, pred_idx, cls_idx):
        if f1 > scores[pi][ci]:
            scores[pi][ci] = f1

    return [classes[int(max(range(len(classes)), key=lambda c: scores[pi][c]))]
            for pi in range(n_preds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",       required=True, help="splits.json from finetune_corrective.py")
    parser.add_argument("--labels-jsonl", default="../labels/labels_5var_reusable.json",
                        help="Labels file (JSONL or JSON dict)")
    parser.add_argument("--ckpt",         default="../checkpoints/motionllm_corrective_v1/best.pth")
    parser.add_argument("--vqvae-ckpt",   default="../motion_encoder/checkpoints/vqvae_indomain_best.pth",
                        help="VQ-VAE checkpoint (must match the one used during training)")
    parser.add_argument("--lora-r",       type=int, default=16)
    parser.add_argument("--lora-alpha",   type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--split",        default="test", choices=["test", "val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--top-p",           type=float, default=1.0)
    parser.add_argument("--do-sample",       action="store_true")
    parser.add_argument("--use-bertscore",   action="store_true",
                        help="Use BERTScore to classify predictions instead of keyword matching")
    parser.add_argument("--bertscore-refs",  default="../labels/class_labels_pooled_150.json",
                        help="JSON file {class: [ref1, ...]} used as BERTScore references")
    parser.add_argument("--bertscore-n-refs", type=int, default=10,
                        help="Number of reference sentences per class to use (default: 10)")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased",
                        help="HuggingFace model for BERTScore (default: distilbert-base-uncased)")
    our_args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    _argv, sys.argv = sys.argv, sys.argv[:1]
    llm_args = get_llm_args()
    sys.argv = _argv
    llm_args.device         = our_args.device
    llm_args.vq_path        = our_args.vqvae_ckpt
    llm_args.lora_r_m2t     = our_args.lora_r
    llm_args.lora_alpha_m2t = our_args.lora_alpha
    llm_args.lora_dropout   = our_args.lora_dropout

    print("Building MotionLLM …")
    model = MotionLLM(llm_args)
    model.load_model(our_args.ckpt)
    model.llm.eval()
    model.to(our_args.device)

    # ── load test paths and labels ────────────────────────────────────────────
    with open(our_args.splits) as f:
        splits = json.load(f)
    key = "val_paths" if our_args.split == "val" else "test_paths"
    test_paths = splits[key]
    print(f"{our_args.split.capitalize()} samples: {len(test_paths)}")

    # Load labels — supports both per-sample dicts and class-level dicts (class → list)
    raw_labels = json.load(open(our_args.labels_jsonl))
    is_class_level = isinstance(list(raw_labels.values())[0], list)
    if is_class_level:
        # class_labels.json: {"squat_butt_wink": ["v1", "v2", ...], ...}
        class_label_map = raw_labels
        gt_map = {}
    else:
        gt_map = _load_labels(our_args.labels_jsonl)
        class_label_map = {}

    # ── load BERTScore references if needed ──────────────────────────────────
    bs_class_refs = None
    if our_args.use_bertscore:
        import random as _random
        raw_refs = json.load(open(our_args.bertscore_refs))
        bs_class_refs = {}
        for cls, refs in raw_refs.items():
            sample = _random.sample(refs, min(our_args.bertscore_n_refs, len(refs)))
            bs_class_refs[cls] = sample
        print(f"BERTScore: {len(bs_class_refs)} classes, "
              f"{our_args.bertscore_n_refs} refs each, model={our_args.bertscore_model}")

    # ── run inference ─────────────────────────────────────────────────────────
    raw_results = []  # collect (npy_path, gt_cls, sample_id, gt, pred) before classification
    for npy_path in test_paths:
        fname    = os.path.basename(npy_path)
        raw_id   = fname.replace("HSMR-", "").replace("_guofeats.npy", "")
        gt_cls    = raw_id.rstrip("0123456789")
        sample_id = raw_id if raw_id in gt_map else raw_id.replace("squat_", "", 1)
        if class_label_map:
            gt = class_label_map.get(gt_cls, [""])[0]
        else:
            gt = gt_map.get(sample_id, "")

        motion = np.load(npy_path).astype(np.float32)
        pred   = model.caption(motion, instruction=get_instruction(sample_id),
                               temperature=our_args.temperature, top_p=our_args.top_p,
                               do_sample=our_args.do_sample)
        raw_results.append((npy_path, gt_cls, sample_id, gt, pred))

    # ── classify predictions ──────────────────────────────────────────────────
    predictions = [r[4] for r in raw_results]
    if our_args.use_bertscore:
        pred_classes = bertscore_classify(predictions, bs_class_refs,
                                          bs_model=our_args.bertscore_model,
                                          device=our_args.device)
    else:
        pred_classes = [detect_class(p) for p in predictions]

    results = []
    for (npy_path, gt_cls, sample_id, gt, pred), pred_cls in zip(raw_results, pred_classes):
        correct  = (pred_cls == gt_cls)

        gt_cls_set   = gt_label_classes(gt, gt_cls)
        pred_cls_set = all_mentioned_classes(pred) or {"unknown"}
        tp           = len(gt_cls_set & pred_cls_set)
        ml_precision = tp / len(pred_cls_set)
        ml_recall    = tp / len(gt_cls_set)
        ml_f1        = (2 * ml_precision * ml_recall / (ml_precision + ml_recall)
                        if (ml_precision + ml_recall) > 0 else 0.0)

        results.append({
            "id": sample_id, "gt_cls": gt_cls, "pred_cls": pred_cls,
            "correct": correct, "gt": gt, "pred": pred,
            "gt_cls_set": sorted(gt_cls_set), "pred_cls_set": sorted(pred_cls_set),
            "ml_precision": ml_precision, "ml_recall": ml_recall, "ml_f1": ml_f1,
        })

        mark = "✓" if correct else "✗"
        print(f"\n[{sample_id}]  GT class: {gt_cls}  →  PRED class: {pred_cls}  {mark}")
        print(f"  GT labels:  {sorted(gt_cls_set)}  PRED labels: {sorted(pred_cls_set)}")
        print(f"  P={ml_precision:.2f}  R={ml_recall:.2f}  F1={ml_f1:.2f}")
        print(f"  GT:   {gt}")
        print(f"  PRED: {pred}")

    # ── per-class summary ─────────────────────────────────────────────────────
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for r in results:
        class_total[r["gt_cls"]] += 1
        if r["correct"]:
            class_correct[r["gt_cls"]] += 1

    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())

    avg_p = np.mean([r["ml_precision"] for r in results])
    avg_r = np.mean([r["ml_recall"]    for r in results])
    avg_f = np.mean([r["ml_f1"]        for r in results])
    primary_in_pred = sum(1 for r in results if r["gt_cls"] in all_mentioned_classes(r["pred"]))

    print("\n" + "=" * 65)
    print(f"{'Class':<25} {'Correct':>7} {'Total':>7} {'Accuracy':>9}  (primary class)")
    print("-" * 65)
    for cls in sorted(class_total):
        n, d = class_correct[cls], class_total[cls]
        print(f"{cls:<25} {n:>7} {d:>7} {n/d*100:>8.1f}%")
    print("-" * 65)
    print(f"{'TOTAL':<25} {total_correct:>7} {total_samples:>7} "
          f"{total_correct/total_samples*100:>8.1f}%")
    print("=" * 65)

    print(f"\nMulti-label metrics (avg over {total_samples} samples):")
    print(f"  Primary class in prediction: {primary_in_pred}/{total_samples} = "
          f"{primary_in_pred/total_samples*100:.1f}%")
    print(f"  Precision:  {avg_p*100:.1f}%")
    print(f"  Recall:     {avg_r*100:.1f}%")
    print(f"  F1:         {avg_f*100:.1f}%")

    # ── save ──────────────────────────────────────────────────────────────────
    summary = {
        cls: {"correct": class_correct[cls], "total": class_total[cls],
              "accuracy": class_correct[cls] / class_total[cls]}
        for cls in class_total
    }
    ml_summary = {"avg_precision": avg_p, "avg_recall": avg_r, "avg_f1": avg_f,
                  "primary_in_pred": primary_in_pred, "total": total_samples}
    out_path = our_args.ckpt.replace(".pth", "_test_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "ml_summary": ml_summary, "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
