"""
Out-of-distribution evaluation against /deck/users/mpiras/dataset/out_of_distribution.

Mirrors evaluate.py (same inference branches, same BERTScore classification head)
but pulls the test set from ground_truth.json and uses the OOD guofeats dir.

Usage:
    python evaluate_ood.py --base-model gemma4         --checkpoint checkpoints_gemma4_v3/best
    python evaluate_ood.py --base-model qwen2vl        --checkpoint checkpoints_qwen_v3/best
    python evaluate_ood.py --base-model gemma4_motion_proj --checkpoint checkpoints_gemma4_motion_proj_v2/best \
        --vqvae-ckpt /deck/users/mpiras/motion-agent/experiments/v2_vqvae_noaug_nocls/vqvae_aug_cls_best.pth
    python evaluate_ood.py --base-model qwen_motion_proj   --checkpoint checkpoints_qwen_motion_proj_v2/best \
        --vqvae-ckpt /deck/users/mpiras/motion-agent/experiments/v2_vqvae_noaug_nocls/vqvae_aug_cls_best.pth
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from bert_score import score as bert_score_fn
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/deck/users/mpiras/motion-agent")
from models import vqvae as vqvae_module

# ── paths ─────────────────────────────────────────────────────────────────────
OOD_VIDEO_DIR    = "/deck/users/mpiras/dataset/out_of_distribution"
OOD_GUO_DIR      = "/deck/users/mpiras/motion-agent/pipeline_out/ood"
GROUND_TRUTH     = "/deck/users/mpiras/dataset/out_of_distribution/ground_truth.json"
REFS_PATH        = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"
DEFAULT_VQVAE    = "/deck/users/mpiras/motion-agent/experiments/v2_vqvae_noaug_nocls/vqvae_aug_cls_best.pth"
MEAN_PATH        = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH         = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

NB_CODE     = 512
FRAME_SIZE  = 336
N_FRAMES    = 12

BASE_MODELS = {
    "qwen2vl":              "Qwen/Qwen2-VL-2B-Instruct",
    "qwen_motion_proj":     "Qwen/Qwen2-VL-2B-Instruct",
    "qwen_qformer":         "Qwen/Qwen2-VL-2B-Instruct",
    "gemma4":               "google/gemma-4-E2B-it",
    "gemma4_motion_proj":   "google/gemma-4-E2B-it",
    "gemma4_qformer":       "google/gemma-4-E2B-it",
}

# Ground-truth label normalization (GT uses short names; training uses long).
GT_TO_CLASS = {
    "no_error":          "squat_no_errors",
    "butt_wink":         "squat_butt_wink",
    "depth_high":        "squat_depth_high",
    "rdl_no_error":      "rdl_no_error",
    "rdl_hands_forward": "rdl_hands_forward",
}

PROMPT_QWEN = (
    "The images above are frames sampled uniformly from a single strength exercise "
    "repetition, in chronological order. First identify the exercise being performed, "
    "then carefully analyze the person's form throughout the movement. Identify any "
    "execution errors or form faults. If the form looks correct, say so. Be specific "
    "about what you observe and at which phase of the movement it occurs."
)

PROMPT_GEMMA4 = (
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)

PROMPT_MOTION_PROJ_TEMPLATE = (
    "Motion token sequence: {motion_tokens}\n\n"
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)

PROMPT_QWEN_MOTION_TEMPLATE = (
    "Motion token sequence: {motion_tokens}\n\n"
    "The images above are frames sampled uniformly from a single strength exercise "
    "repetition, in chronological order. First identify the exercise being performed, "
    "then carefully analyze the person's form throughout the movement. Identify any "
    "execution errors or form faults. If the form looks correct, say so. Be specific "
    "about what you observe and at which phase of the movement it occurs."
)


def make_qformer_slot_str(n_queries):
    return "".join(f"<MQ_{i}>" for i in range(n_queries))


def make_qformer_prompt_gemma4(n_queries):
    return (
        f"Motion summary: {make_qformer_slot_str(n_queries)}\n\n"
        "First identify the strength exercise being performed, then carefully analyze "
        "the person's form throughout the movement. Identify any execution errors or "
        "form faults. If the form looks correct, say so. Be specific about what you "
        "observe and at which phase of the movement it occurs."
    )


def make_qformer_prompt_qwen(n_queries):
    return (
        f"Motion summary: {make_qformer_slot_str(n_queries)}\n\n"
        "The images above are frames sampled uniformly from a single strength exercise "
        "repetition, in chronological order. First identify the exercise being performed, "
        "then carefully analyze the person's form throughout the movement. Identify any "
        "execution errors or form faults. If the form looks correct, say so. Be specific "
        "about what you observe and at which phase of the movement it occurs."
    )


# ── motion-aware embedding (same as evaluate.py) ──────────────────────────────

class MotionAwareEmbedding(nn.Module):
    def __init__(self, orig_embed, codebook, orig_vocab_size):
        super().__init__()
        self.orig_embed      = orig_embed
        self.orig_vocab_size = orig_vocab_size
        hidden_dim = orig_embed.weight.shape[1]
        self.motion_proj = nn.Linear(codebook.shape[1], hidden_dim, bias=False,
                                     dtype=orig_embed.weight.dtype)
        self.register_buffer("codebook", codebook.to(orig_embed.weight.dtype))
        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.orig_embed.weight

    def forward(self, input_ids):
        is_motion = input_ids >= self.orig_vocab_size
        safe_ids  = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds    = self.orig_embed(safe_ids)
        if is_motion.any():
            code_idx   = (input_ids[is_motion] - self.orig_vocab_size).clamp(0)
            motion_emb = self.motion_proj(self.codebook[code_idx])
            embeds = embeds.clone()
            embeds[is_motion] = motion_emb.to(embeds.dtype)
        return embeds


class MotionQFormer(nn.Module):
    def __init__(self, motion_dim, hidden_dim, n_queries, n_layers, n_heads, dtype):
        super().__init__()
        self.n_queries = n_queries
        self.queries   = nn.Parameter(torch.zeros(n_queries, hidden_dim, dtype=dtype))
        self.in_proj   = nn.Linear(motion_dim, hidden_dim, dtype=dtype)
        self.layers    = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_norm_q":  nn.LayerNorm(hidden_dim, dtype=dtype),
                "cross_norm_kv": nn.LayerNorm(hidden_dim, dtype=dtype),
                "cross_attn":    nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dtype=dtype),
                "self_norm":     nn.LayerNorm(hidden_dim, dtype=dtype),
                "self_attn":     nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dtype=dtype),
                "ffn_norm":      nn.LayerNorm(hidden_dim, dtype=dtype),
                "ffn":           nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype),
                ),
            }))

    def forward(self, motion_feats):
        B = motion_feats.shape[0]
        kv = self.in_proj(motion_feats)
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)
        for blk in self.layers:
            qn  = blk["cross_norm_q"](q)
            kvn = blk["cross_norm_kv"](kv)
            attn_out, _ = blk["cross_attn"](qn, kvn, kvn, need_weights=False)
            q = q + attn_out
            qn = blk["self_norm"](q)
            attn_out, _ = blk["self_attn"](qn, qn, qn, need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](blk["ffn_norm"](q))
        return q


class QFormerAwareEmbedding(nn.Module):
    def __init__(self, orig_embed, qformer, orig_vocab_size, n_queries):
        super().__init__()
        self.orig_embed      = orig_embed
        self.qformer         = qformer
        self.orig_vocab_size = orig_vocab_size
        self.n_queries       = n_queries
        self._current_q      = None
        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.orig_embed.weight

    def set_motion_features(self, motion_feats):
        self._current_q = self.qformer(motion_feats)

    def forward(self, input_ids):
        is_motion = input_ids >= self.orig_vocab_size
        safe_ids  = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds    = self.orig_embed(safe_ids)
        if is_motion.any() and self._current_q is not None:
            slot_idx = (input_ids - self.orig_vocab_size).clamp(0, self.n_queries - 1)
            b_idx, l_idx = torch.where(is_motion)
            slot_at_pos = slot_idx[b_idx, l_idx]
            qf_emb = self._current_q[b_idx, slot_at_pos]
            embeds = embeds.clone()
            embeds[b_idx, l_idx] = qf_emb.to(embeds.dtype)
        return embeds


# ── VQ-VAE helpers ────────────────────────────────────────────────────────────

def load_vqvae_for_eval(device, ckpt_path):
    import argparse as ap
    a = ap.Namespace(dataname="hml3d", nb_code=NB_CODE, code_dim=512, output_emb_width=512,
                     down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                     vq_act="relu", vq_norm=None, quantizer="ema_reset", mu=0.99)
    net = vqvae_module.HumanVQVAE(a, a.nb_code, a.code_dim, a.output_emb_width,
                                  a.down_t, a.stride_t, a.width, a.depth,
                                  a.dilation_growth_rate, a.vq_act, a.vq_norm)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)
    codebook = ckpt["net"]["vqvae.quantizer.codebook"]
    mean = np.load(MEAN_PATH); std = np.load(STD_PATH)
    return net, codebook, mean, std


@torch.no_grad()
def encode_motion(vqvae, guo_path, mean, std, device):
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    return vqvae.encode(t).squeeze(0).cpu().tolist()


@torch.no_grad()
def encode_motion_features(vqvae, codebook, guo_path, mean, std, device):
    """Continuous codebook vectors (T, motion_dim) for Q-Former cross-attention."""
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    indices = vqvae.encode(t).squeeze(0)
    return codebook[indices.to(codebook.device)]


def find_guofeats(stem, guo_dir):
    for name in (f"{stem}_guofeats.npy", f"HSMR-{stem}_guofeats.npy"):
        p = os.path.join(guo_dir, name)
        if os.path.exists(p):
            return p
    return None


# ── video helpers ─────────────────────────────────────────────────────────────

def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, N_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR))
    cap.release()
    return frames


def load_video_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


# ── BERTScore classifier (mirrors evaluate.py) ────────────────────────────────

def bertscore_classify(predictions, class_refs, bs_model, device):
    classes = list(class_refs.keys())
    all_cands, all_refs, pred_idx, cls_idx = [], [], [], []
    for pi, pred in enumerate(predictions):
        for ci, cls in enumerate(classes):
            for ref in class_refs[cls]:
                all_cands.append(pred); all_refs.append(ref)
                pred_idx.append(pi);    cls_idx.append(ci)
    print(f"  BERTScore: scoring {len(all_cands)} pairs...")
    _, _, F1 = bert_score_fn(all_cands, all_refs, model_type=bs_model,
                             device=device, verbose=False)
    F1 = F1.tolist()
    scores = [[0.0] * len(classes) for _ in predictions]
    for f1, pi, ci in zip(F1, pred_idx, cls_idx):
        if f1 > scores[pi][ci]:
            scores[pi][ci] = f1
    return [classes[max(range(len(classes)), key=lambda c: scores[pi][c])]
            for pi in range(len(predictions))]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--base-model",   default="gemma4", choices=list(BASE_MODELS.keys()))
    parser.add_argument("--output",       required=True)
    parser.add_argument("--n-refs",       type=int, default=10)
    parser.add_argument("--bs-model",     default="distilbert-base-uncased")
    parser.add_argument("--video-dir",    default=OOD_VIDEO_DIR)
    parser.add_argument("--guofeats-dir", default=OOD_GUO_DIR)
    parser.add_argument("--ground-truth", default=GROUND_TRUTH)
    parser.add_argument("--vqvae-ckpt",   default=DEFAULT_VQVAE)
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Sampling temperature (default: 1.0 = greedy)")
    parser.add_argument("--top-p",        type=float, default=1.0,
                        help="Nucleus sampling top-p (default: 1.0 = off)")
    args = parser.parse_args()

    # ── ground truth ──────────────────────────────────────────────────────────
    with open(args.ground_truth) as f:
        gt = json.load(f)

    test_samples = []
    for fname, gt_label in gt.items():
        true_class = GT_TO_CLASS.get(gt_label, gt_label)
        if true_class not in {*GT_TO_CLASS.values()}:
            # Allow unmapped labels through, but warn — they won't match any class ref
            print(f"WARNING: GT label '{gt_label}' for {fname} has no mapping → "
                  f"using as-is. BERTScore will likely never predict this.")
        vpath = os.path.join(args.video_dir, fname)
        if not os.path.exists(vpath):
            print(f"WARNING: video missing: {vpath} — skipping.")
            continue
        test_samples.append((vpath, true_class, fname))
    print(f"OOD test set: {len(test_samples)} videos")

    # ── BERTScore references ──────────────────────────────────────────────────
    with open(REFS_PATH) as f:
        all_refs = json.load(f)
    rng = random.Random(0)
    class_refs = {cls: rng.sample(refs, min(args.n_refs, len(refs)))
                  for cls, refs in all_refs.items()}
    print(f"BERTScore refs: {args.n_refs} per class")

    # ── model + processor ─────────────────────────────────────────────────────
    base_model_id = BASE_MODELS[args.base_model]
    print(f"Loading model {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id)

    is_motion_proj      = args.base_model == "gemma4_motion_proj"
    is_qwen_motion_proj = args.base_model == "qwen_motion_proj"
    is_gemma4_qformer   = args.base_model == "gemma4_qformer"
    is_qwen_qformer     = args.base_model == "qwen_qformer"
    is_qformer          = is_gemma4_qformer or is_qwen_qformer
    is_qwen             = args.base_model in ("qwen2vl", "qwen_motion_proj", "qwen_qformer")
    needs_motion_proj   = is_motion_proj or is_qwen_motion_proj

    qf_cfg = None
    if is_qformer:
        with open(os.path.join(args.checkpoint, "qformer_config.json")) as f:
            qf_cfg = json.load(f)

    if is_qformer:
        processor.tokenizer.add_tokens([f"<MQ_{i}>" for i in range(qf_cfg["n_queries"])])

    if needs_motion_proj:
        processor.tokenizer.add_tokens([f"<Motion_{i}>" for i in range(NB_CODE)])

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if needs_motion_proj:
        base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        orig_vocab_size_mp = base.get_input_embeddings().weight.shape[0] - NB_CODE
        orig_embed_mp      = base.get_input_embeddings()

    if is_qformer:
        base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        orig_vocab_size_qf = base.get_input_embeddings().weight.shape[0] - qf_cfg["n_queries"]
        orig_embed_qf      = base.get_input_embeddings()

    model = PeftModel.from_pretrained(base, args.checkpoint).eval()
    device = next(model.parameters()).device

    if needs_motion_proj:
        print("Loading VQ-VAE...")
        vqvae, codebook, mean_npy, std_npy = load_vqvae_for_eval(device, args.vqvae_ckpt)
        motion_embed = MotionAwareEmbedding(orig_embed_mp, codebook.to(device), orig_vocab_size_mp).to(device)
        motion_embed.motion_proj.load_state_dict(
            torch.load(os.path.join(args.checkpoint, "motion_proj.pt"), map_location=device)
        )
        motion_embed.motion_proj = motion_embed.motion_proj.to(device, torch.bfloat16)
        model.set_input_embeddings(motion_embed)
        print("MotionAwareEmbedding installed.")

    if is_qformer:
        print("Loading VQ-VAE + Q-Former...")
        vqvae, codebook, mean_npy, std_npy = load_vqvae_for_eval(device, args.vqvae_ckpt)
        codebook = codebook.to(device)
        qformer = MotionQFormer(
            motion_dim=qf_cfg["motion_dim"], hidden_dim=qf_cfg["hidden_dim"],
            n_queries=qf_cfg["n_queries"],   n_layers=qf_cfg["qf_layers"],
            n_heads=qf_cfg["qf_heads"],      dtype=torch.bfloat16,
        ).to(device)
        qformer.load_state_dict(
            torch.load(os.path.join(args.checkpoint, "qformer.pt"), map_location=device)
        )
        qformer = qformer.to(device, torch.bfloat16).eval()
        qf_embed = QFormerAwareEmbedding(
            orig_embed_qf, qformer, orig_vocab_size_qf, qf_cfg["n_queries"]
        ).to(device)
        model.set_input_embeddings(qf_embed)
        print(f"Q-Former installed ({qf_cfg['n_queries']} queries).")
    print("Model loaded.\n")

    # ── inference loop ────────────────────────────────────────────────────────
    records = []
    for i, (video_path, true_class, fname) in enumerate(test_samples):
        mot_str = None
        if needs_motion_proj:
            stem = Path(video_path).stem
            guo_path = find_guofeats(stem, args.guofeats_dir)
            if guo_path is None:
                print(f"  WARNING: no guofeats for {fname}, skipping.")
                continue
            indices = encode_motion(vqvae, guo_path, mean_npy, std_npy, device)
            mot_str = "".join(f"<Motion_{i}>" for i in indices)

        if is_qformer:
            stem = Path(video_path).stem
            guo_path = find_guofeats(stem, args.guofeats_dir)
            if guo_path is None:
                print(f"  WARNING: no guofeats for {fname}, skipping.")
                continue
            motion_feats = encode_motion_features(vqvae, codebook, guo_path,
                                                   mean_npy, std_npy, device)
            qf_embed.set_motion_features(motion_feats.unsqueeze(0).to(device, torch.bfloat16))

        if is_qwen_qformer:
            frames   = sample_frames(video_path)
            prompt   = make_qformer_prompt_qwen(qf_cfg["n_queries"])
            content  = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs   = processor(text=[text], images=frames, return_tensors="pt", padding=True)
        elif is_gemma4_qformer:
            prompt   = make_qformer_prompt_gemma4(qf_cfg["n_queries"])
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_array = load_video_array(video_path)
            inputs      = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_qwen_motion_proj:
            frames   = sample_frames(video_path)
            prompt   = PROMPT_QWEN_MOTION_TEMPLATE.format(motion_tokens=mot_str)
            content  = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs   = processor(text=[text], images=frames, return_tensors="pt", padding=True)
        elif is_motion_proj:
            prompt   = PROMPT_MOTION_PROJ_TEMPLATE.format(motion_tokens=mot_str)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_array = load_video_array(video_path)
            inputs      = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_qwen:
            frames   = sample_frames(video_path)
            content  = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": PROMPT_QWEN})
            messages = [{"role": "user", "content": content}]
            text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs   = processor(text=[text], images=frames, return_tensors="pt", padding=True)
        else:  # gemma4 plain
            video_array = load_video_array(video_path)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": PROMPT_GEMMA4}
            ]}]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, videos=video_array, return_tensors="pt")

        inputs = inputs.to(device, torch.bfloat16)

        do_sample = args.temperature != 1.0 or args.top_p != 1.0
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 1.0,
                top_p=args.top_p if do_sample else 1.0,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        response  = processor.decode(generated, skip_special_tokens=True)

        print(f"[{i+1:2d}/{len(test_samples)}] {true_class:<22s}  {response[:80]}...")
        records.append({"video": fname, "true_class": true_class, "response": response})

    # ── BERTScore classification ──────────────────────────────────────────────
    print("\nRunning BERTScore classification...")
    predictions  = [r["response"] for r in records]
    pred_classes = bertscore_classify(predictions, class_refs, args.bs_model, str(device))

    # ── accuracy ──────────────────────────────────────────────────────────────
    per_class_correct, per_class_total = defaultdict(int), defaultdict(int)
    for r, pc in zip(records, pred_classes):
        r["predicted_class"] = pc
        r["correct"] = pc == r["true_class"]
        per_class_total[r["true_class"]] += 1
        if r["correct"]:
            per_class_correct[r["true_class"]] += 1

    total = len(records)
    correct_all = sum(per_class_correct.values())

    print(f"\n{'CLASS':<25s} {'CORRECT':>7}  {'TOTAL':>5}  {'ACC':>6}")
    print("-" * 55)
    for cls in sorted(per_class_total.keys()):
        c, t = per_class_correct[cls], per_class_total[cls]
        print(f"{cls:<25s} {c:>7d}  {t:>5d}  {c/t:>5.1%}")
    print("-" * 55)
    print(f"{'OVERALL':<25s} {correct_all:>7d}  {total:>5d}  {correct_all/total:>5.1%}")

    out = {
        "summary": {
            "overall_acc": correct_all / total if total else 0.0,
            "per_class":   {k: per_class_correct[k] / per_class_total[k]
                            for k in per_class_total},
        },
        "results": records,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
