"""
Evaluation of finetuned VLM checkpoints on the held-out test set.

Classification via BERTScore: each prediction is scored against N reference
sentences per class (sampled from class_labels_pooled_150.json). The class
with the highest max F1 wins.

Usage:
    # Qwen2-VL checkpoint
    python evaluate.py --checkpoint checkpoints/best

    # Gemma4 checkpoint
    python evaluate.py --checkpoint checkpoints_gemma4/best --base-model gemma4 \
        --output eval_results_gemma4.json
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from bert_score import score as bert_score_fn
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/deck/users/mpiras/motion-agent")
from models import vqvae as vqvae_module

# ── paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "/deck/users/mpiras/paligemma_exercise/checkpoints/best"
VIDEO_DIRS = [
    "/deck/users/mpiras/dataset/rdls",
    "/deck/users/mpiras/dataset/squat_micc",
]
LABELS_PATH     = "/deck/users/mpiras/dataset/LLM_lables/class_labels.json"
REFS_PATH       = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"
N_TEST_PER_CLASS = 5
N_VAL_PER_CLASS  = 5
FRAME_SIZE       = 336
N_FRAMES         = 12

BASE_MODELS = {
    "qwen2vl":              "Qwen/Qwen2-VL-2B-Instruct",
    "qwen_motion":          "Qwen/Qwen2-VL-2B-Instruct",
    "qwen_motion_proj":     "Qwen/Qwen2-VL-2B-Instruct",
    "qwen_qformer":         "Qwen/Qwen2-VL-2B-Instruct",
    "gemma3":               "google/gemma-3-4b-it",
    "gemma4":               "google/gemma-4-E2B-it",
    "gemma4_motion":        "google/gemma-4-E2B-it",
    "gemma4_motion_proj":   "google/gemma-4-E2B-it",
    "gemma4_qformer":       "google/gemma-4-E2B-it",
    "gemma4_hsmr":          "google/gemma-4-E2B-it",
}

N_FRAMES_GEMMA3 = 10

VQVAE_CKPT   = "/deck/users/mpiras/motion-agent/ckpt/vqvae.pth"
GUOFEATS_DIR = "/deck/users/mpiras/dataset/hsmr_guofeats"
MEAN_PATH    = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH     = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
NB_CODE      = 512

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

QFORMER_DEFAULT_N_QUERIES = 8


def make_qformer_slot_str(n_queries: int) -> str:
    return "".join(f"<MQ_{i}>" for i in range(n_queries))


def make_qformer_prompt_gemma4(n_queries: int) -> str:
    return (
        f"Motion summary: {make_qformer_slot_str(n_queries)}\n\n"
        "First identify the strength exercise being performed, then carefully analyze "
        "the person's form throughout the movement. Identify any execution errors or "
        "form faults. If the form looks correct, say so. Be specific about what you "
        "observe and at which phase of the movement it occurs."
    )


def make_qformer_prompt_qwen(n_queries: int) -> str:
    return (
        f"Motion summary: {make_qformer_slot_str(n_queries)}\n\n"
        "The images above are frames sampled uniformly from a single strength exercise "
        "repetition, in chronological order. First identify the exercise being performed, "
        "then carefully analyze the person's form throughout the movement. Identify any "
        "execution errors or form faults. If the form looks correct, say so. Be specific "
        "about what you observe and at which phase of the movement it occurs."
    )

PROMPT = (
    "The images above are frames sampled uniformly from a single strength exercise repetition, "
    "in chronological order. First identify the exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or form faults. "
    "If the form looks correct, say so. Be specific about what you observe and at which phase "
    "of the movement it occurs."
)

PROMPT_GEMMA4 = (
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)

PROMPT_HSMR_TEMPLATE = (
    "HSMR pose sequence: {hsmr_tokens}\n\n"
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)

HSMR_DIR  = "/deck/users/mpiras/dataset/hsmr"
POSE_DIM  = 46
K_FRAMES  = 16
import torch.nn as nn


class MotionAwareEmbedding(nn.Module):
    def __init__(self, orig_embed: nn.Module, codebook: torch.Tensor, orig_vocab_size: int):
        super().__init__()
        self.orig_embed      = orig_embed
        self.orig_vocab_size = orig_vocab_size
        hidden_dim           = orig_embed.weight.shape[1]
        vqvae_dim            = codebook.shape[1]
        self.motion_proj     = nn.Linear(vqvae_dim, hidden_dim, bias=False,
                                         dtype=orig_embed.weight.dtype)
        self.register_buffer("codebook", codebook.to(orig_embed.weight.dtype))
        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.orig_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_motion = input_ids >= self.orig_vocab_size
        safe_ids  = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds    = self.orig_embed(safe_ids)
        if is_motion.any():
            code_idx   = (input_ids[is_motion] - self.orig_vocab_size).clamp(0)
            motion_emb = self.motion_proj(self.codebook[code_idx])
            embeds = embeds.clone()
            embeds[is_motion] = motion_emb.to(embeds.dtype)
        return embeds


class HSMRAwareEmbedding(nn.Module):
    def __init__(self, orig_embed, orig_vocab_size, k_frames, pose_dim=POSE_DIM):
        super().__init__()
        self.orig_embed      = orig_embed
        self.orig_vocab_size = orig_vocab_size
        self.k_frames        = k_frames
        hidden_dim           = orig_embed.weight.shape[1]
        self.motion_proj     = nn.Linear(pose_dim, hidden_dim, bias=False,
                                         dtype=orig_embed.weight.dtype)
        self._current_features = None
        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.orig_embed.weight

    def set_features(self, features: torch.Tensor):
        self._current_features = features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_hsmr  = input_ids >= self.orig_vocab_size
        safe_ids = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds   = self.orig_embed(safe_ids)
        if is_hsmr.any() and self._current_features is not None:
            frame_idx = (input_ids[is_hsmr] - self.orig_vocab_size).clamp(0, self.k_frames - 1)
            hsmr_emb  = self.motion_proj(self._current_features[frame_idx])
            embeds = embeds.clone()
            embeds[is_hsmr] = hsmr_emb.to(embeds.dtype)
        return embeds


class MotionQFormer(nn.Module):
    """Re-construction at eval time of the Q-Former trained by finetune_*_qformer.py."""

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
    """Drop-in embed_tokens replacement that routes <MQ_i> slots to Q-Former output."""

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


def load_vqvae_for_eval(device: str, ckpt_path: str = VQVAE_CKPT):
    import argparse as ap
    a = ap.Namespace(
        dataname="hml3d", nb_code=NB_CODE, code_dim=512, output_emb_width=512,
        down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
        vq_act="relu", vq_norm=None, quantizer="ema_reset", mu=0.99,
    )
    net = vqvae_module.HumanVQVAE(
        a, a.nb_code, a.code_dim, a.output_emb_width, a.down_t, a.stride_t,
        a.width, a.depth, a.dilation_growth_rate, a.vq_act, a.vq_norm,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)
    codebook = ckpt["net"]["vqvae.quantizer.codebook"]
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    return net, codebook, mean, std


@torch.no_grad()
def encode_motion(vqvae, guo_path: str, mean: np.ndarray, std: np.ndarray, device: str) -> list[int]:
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    return vqvae.encode(t).squeeze(0).cpu().tolist()


@torch.no_grad()
def encode_motion_features(vqvae, codebook: torch.Tensor, guo_path: str,
                            mean: np.ndarray, std: np.ndarray, device: str) -> torch.Tensor:
    """For Q-Former: return the per-timestep codebook vectors (T, motion_dim)."""
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    indices = vqvae.encode(t).squeeze(0)
    return codebook[indices.to(codebook.device)]


def load_hsmr_poses(video_stem: str, hsmr_dir: str, k: int) -> np.ndarray | None:
    path = os.path.join(hsmr_dir, f"HSMR-{video_stem}.npy")
    if not os.path.exists(path):
        return None
    frames  = np.load(path, allow_pickle=True)
    T       = len(frames)
    indices = np.linspace(0, T - 1, k, dtype=int)
    return np.stack([frames[i]["poses"][0] for i in indices]).astype(np.float32)


# ── dataset helpers ───────────────────────────────────────────────────────────

def extract_class(filename: str, class_keys: list[str]) -> str | None:
    stem = re.sub(r"\d+$", "", Path(filename).stem)
    for key in sorted(class_keys, key=len, reverse=True):
        if stem == key:
            return key
    return None


def build_by_class(class_keys: list[str], labeled_fnames: set | None = None) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {k: [] for k in class_keys}
    for video_dir in VIDEO_DIRS:
        for fname in sorted(os.listdir(video_dir)):
            if not fname.endswith(".mp4"):
                continue
            if labeled_fnames is not None and fname not in labeled_fnames:
                continue
            key = extract_class(fname, class_keys)
            if key:
                by_class[key].append(os.path.join(video_dir, fname))
    return by_class


def get_test_samples(by_class: dict[str, list[str]]) -> list[tuple[str, str]]:
    rng = random.Random(42)
    test = []
    for class_key, paths in by_class.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        for p in shuffled[:N_TEST_PER_CLASS]:
            test.append((p, class_key))
    return test


def sample_frames(video_path: str) -> list[Image.Image]:
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


def load_video_array(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


# ── BERTScore classification ──────────────────────────────────────────────────

def bertscore_classify(predictions: list[str],
                       class_refs: dict[str, list[str]],
                       bs_model: str = "distilbert-base-uncased",
                       device: str = "cuda") -> list[str]:
    """
    For each prediction, compute BERTScore F1 against every reference sentence
    of every class. Assign the class whose max F1 is highest.
    """
    classes = list(class_refs.keys())
    all_cands, all_refs, pred_idx, cls_idx = [], [], [], []

    for pi, pred in enumerate(predictions):
        for ci, cls in enumerate(classes):
            for ref in class_refs[cls]:
                all_cands.append(pred)
                all_refs.append(ref)
                pred_idx.append(pi)
                cls_idx.append(ci)

    print(f"  BERTScore: scoring {len(all_cands)} (pred, ref) pairs...")
    _, _, F1 = bert_score_fn(all_cands, all_refs, model_type=bs_model,
                             device=device, verbose=False)
    F1 = F1.tolist()

    n_preds = len(predictions)
    scores = [[0.0] * len(classes) for _ in range(n_preds)]
    for f1, pi, ci in zip(F1, pred_idx, cls_idx):
        if f1 > scores[pi][ci]:
            scores[pi][ci] = f1

    return [classes[max(range(len(classes)), key=lambda c: scores[pi][c])]
            for pi in range(n_preds)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default="qwen2vl", choices=list(BASE_MODELS.keys()))
    parser.add_argument("--output", default="/deck/users/mpiras/paligemma_exercise/eval_results.json")
    parser.add_argument("--n-refs", type=int, default=10,
                        help="Reference sentences per class for BERTScore (default: 10)")
    parser.add_argument("--bs-model", default="distilbert-base-uncased",
                        help="BERTScore backbone model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling top-p (default: 1.0 = off)")
    parser.add_argument("--labels", default=LABELS_PATH,
                        help="Labels JSON (class-keyed or file-keyed) used during training")
    parser.add_argument("--split", default=None,
                        help="Path to split.json saved at training time. If omitted, "
                             "auto-discovered at {checkpoint}/../split.json")
    parser.add_argument("--hsmr-dir", default=HSMR_DIR,
                        help="Directory with HSMR-{stem}.npy files (for gemma4_hsmr)")
    parser.add_argument("--vqvae-ckpt", default=VQVAE_CKPT,
                        help="VQ-VAE checkpoint path (for gemma4_motion_proj)")
    args = parser.parse_args()

    with open(args.labels) as f:
        class_labels = json.load(f)

    # Detect file-keyed format and derive class keys + filtered filenames
    first_key = next(iter(class_labels))
    if first_key.endswith(".mp4"):
        labeled_fnames = set(class_labels.keys())
        class_keys = sorted({re.sub(r"\d+$", "", Path(k).stem) for k in class_labels})
    else:
        labeled_fnames = None
        class_keys = list(class_labels.keys())

    # load BERTScore references
    with open(REFS_PATH) as f:
        all_refs = json.load(f)
    rng = random.Random(0)
    class_refs = {
        cls: rng.sample(refs, min(args.n_refs, len(refs)))
        for cls, refs in all_refs.items()
    }
    print(f"BERTScore refs: {args.n_refs} per class, model={args.bs_model}")

    # Resolve test set: prefer split.json saved at training time (guarantees disjointness)
    split_path = args.split or os.path.join(os.path.dirname(args.checkpoint.rstrip("/")), "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        test_samples = [(p, c) for p, c in split["test"]]
        print(f"Loaded test set from {split_path}: {len(test_samples)} samples")
    else:
        print(f"WARNING: no split.json at {split_path} — falling back to recomputed split. "
              f"Results may include training samples if the checkpoint was trained before split.json was introduced.")
        by_class = build_by_class(class_keys, labeled_fnames)
        test_samples = get_test_samples(by_class)
        print(f"Test samples: {len(test_samples)} ({N_TEST_PER_CLASS} per class)")

    print("Loading model...")
    base_model_id = BASE_MODELS[args.base_model]
    processor = AutoProcessor.from_pretrained(base_model_id)

    is_hsmr             = args.base_model == "gemma4_hsmr"
    is_motion_proj      = args.base_model == "gemma4_motion_proj"
    is_motion           = args.base_model == "gemma4_motion"
    is_qwen_motion      = args.base_model == "qwen_motion"
    is_qwen_motion_proj = args.base_model == "qwen_motion_proj"
    is_gemma4_qformer   = args.base_model == "gemma4_qformer"
    is_qwen_qformer     = args.base_model == "qwen_qformer"
    is_qformer          = is_gemma4_qformer or is_qwen_qformer
    is_gemma4           = "gemma4" in args.base_model
    is_gemma3           = args.base_model == "gemma3"
    needs_motion_tokens = is_motion or is_motion_proj or is_qwen_motion or is_qwen_motion_proj
    needs_motion_proj   = is_motion_proj or is_qwen_motion_proj
    needs_motion_emb    = is_motion or is_qwen_motion

    # Read Q-Former config (n_queries needed before tokenizer extension)
    qf_cfg = None
    if is_qformer:
        qf_cfg_path = os.path.join(args.checkpoint, "qformer_config.json")
        with open(qf_cfg_path) as f:
            qf_cfg = json.load(f)

    if is_hsmr:
        hsmr_tokens = [f"<HSMR_{i}>" for i in range(K_FRAMES)]
        processor.tokenizer.add_tokens(hsmr_tokens)

    if needs_motion_tokens:
        motion_tokens = [f"<Motion_{i}>" for i in range(NB_CODE)]
        processor.tokenizer.add_tokens(motion_tokens)

    if is_qformer:
        slot_tokens = [f"<MQ_{i}>" for i in range(qf_cfg["n_queries"])]
        processor.tokenizer.add_tokens(slot_tokens)

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if is_hsmr:
        base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        orig_vocab_size = base.get_input_embeddings().weight.shape[0] - K_FRAMES
        orig_embed      = base.get_input_embeddings()

    if needs_motion_tokens:
        base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        orig_vocab_size_mp = base.get_input_embeddings().weight.shape[0] - NB_CODE
        orig_embed_mp      = base.get_input_embeddings()

    if is_qformer:
        base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        orig_vocab_size_qf = base.get_input_embeddings().weight.shape[0] - qf_cfg["n_queries"]
        orig_embed_qf      = base.get_input_embeddings()

    model = PeftModel.from_pretrained(base, args.checkpoint).eval()
    device = next(model.parameters()).device

    def _install_input_embeds(peft_model, new_embed):
        # Works for both Gemma4 (model.model.language_model.embed_tokens) and Qwen2VL
        # (model.embed_tokens) — PeftModel.set_input_embeddings delegates to base.
        peft_model.set_input_embeddings(new_embed)

    if is_hsmr:
        hsmr_embed = HSMRAwareEmbedding(orig_embed, orig_vocab_size, K_FRAMES).to(device)
        hsmr_embed.motion_proj.load_state_dict(
            torch.load(os.path.join(args.checkpoint, "motion_proj.pt"), map_location=device)
        )
        hsmr_embed.motion_proj = hsmr_embed.motion_proj.to(device, torch.bfloat16)
        _install_input_embeds(model, hsmr_embed)
        hsmr_str = "".join(f"<HSMR_{i}>" for i in range(K_FRAMES))
        print("HSMR embedding installed.")

    if needs_motion_proj:
        print("Loading VQ-VAE...")
        vqvae, codebook, mean_npy, std_npy = load_vqvae_for_eval(device, args.vqvae_ckpt)
        motion_embed = MotionAwareEmbedding(orig_embed_mp, codebook.to(device), orig_vocab_size_mp).to(device)
        motion_embed.motion_proj.load_state_dict(
            torch.load(os.path.join(args.checkpoint, "motion_proj.pt"), map_location=device)
        )
        motion_embed.motion_proj = motion_embed.motion_proj.to(device, torch.bfloat16)
        _install_input_embeds(model, motion_embed)
        print("MotionAwareEmbedding installed.")

    if needs_motion_emb:
        print("Loading VQ-VAE for motion-token encoding...")
        vqvae, codebook, mean_npy, std_npy = load_vqvae_for_eval(device, args.vqvae_ckpt)
        motion_emb_path = os.path.join(args.checkpoint, "motion_embeddings.pt")
        if not os.path.exists(motion_emb_path):
            raise FileNotFoundError(
                f"{motion_emb_path} not found — checkpoint trained before "
                f"motion_embeddings.pt was being saved. Retrain with the current "
                f"finetune_{'gemma4' if is_motion else 'qwen'}_motion.py."
            )
        new_rows = torch.load(motion_emb_path, map_location=device)
        emb_w = model.get_input_embeddings().weight
        with torch.no_grad():
            emb_w[orig_vocab_size_mp:].copy_(new_rows.to(emb_w.dtype).to(emb_w.device))
        print(f"Loaded motion embeddings ({new_rows.shape[0]} rows) from {motion_emb_path}")

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
        _install_input_embeds(model, qf_embed)
        print(f"Q-Former installed ({qf_cfg['n_queries']} queries, {qf_cfg['qf_layers']} layer(s)).")

    print("Model loaded.\n")

    # ── run inference, collect responses ─────────────────────────────────────
    records = []
    for i, (video_path, true_class) in enumerate(test_samples):
        fname = os.path.basename(video_path)

        # Resolve motion-token string once for any variant that uses them
        mot_str = None
        if needs_motion_tokens:
            stem     = Path(video_path).stem
            guo_path = os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")
            if not os.path.exists(guo_path):
                guo_path = os.path.join(GUOFEATS_DIR, f"{stem}_guofeats.npy")
            if not os.path.exists(guo_path):
                print(f"  WARNING: no guofeats for {fname}, skipping.")
                continue
            indices = encode_motion(vqvae, guo_path, mean_npy, std_npy, device)
            mot_str = "".join(f"<Motion_{i}>" for i in indices)

        # Q-Former branches need continuous codebook vectors; set them on the embed wrapper.
        if is_qformer:
            stem     = Path(video_path).stem
            guo_path = os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")
            if not os.path.exists(guo_path):
                guo_path = os.path.join(GUOFEATS_DIR, f"{stem}_guofeats.npy")
            if not os.path.exists(guo_path):
                print(f"  WARNING: no guofeats for {fname}, skipping.")
                continue
            motion_feats = encode_motion_features(vqvae, codebook, guo_path,
                                                   mean_npy, std_npy, device)
            qf_embed.set_motion_features(motion_feats.unsqueeze(0).to(device, torch.bfloat16))

        if is_hsmr:
            stem  = Path(video_path).stem
            poses = load_hsmr_poses(stem, args.hsmr_dir, K_FRAMES)
            if poses is None:
                print(f"  WARNING: no HSMR file for {fname}, skipping.")
                continue
            hsmr_embed.set_features(torch.from_numpy(poses).to(device, torch.bfloat16))
            prompt  = PROMPT_HSMR_TEMPLATE.format(hsmr_tokens=hsmr_str)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_array = load_video_array(video_path)
            inputs      = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_gemma4_qformer:
            prompt   = make_qformer_prompt_gemma4(qf_cfg["n_queries"])
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_array = load_video_array(video_path)
            inputs      = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_qwen_qformer:
            frames   = sample_frames(video_path)
            prompt   = make_qformer_prompt_qwen(qf_cfg["n_queries"])
            content  = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs   = processor(text=[text], images=frames, return_tensors="pt", padding=True)
        elif is_motion or is_motion_proj:
            # Gemma4 motion variants: native video + motion tokens
            prompt   = PROMPT_MOTION_PROJ_TEMPLATE.format(motion_tokens=mot_str)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]}]
            text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_array = load_video_array(video_path)
            inputs      = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_qwen_motion or is_qwen_motion_proj:
            # Qwen2-VL motion variants: sampled frames + motion tokens
            frames   = sample_frames(video_path)
            prompt   = PROMPT_QWEN_MOTION_TEMPLATE.format(motion_tokens=mot_str)
            content  = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs   = processor(text=[text], images=frames, return_tensors="pt", padding=True)
        elif is_gemma4:
            video_array = load_video_array(video_path)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": PROMPT_GEMMA4}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, videos=video_array, return_tensors="pt")
        elif is_gemma3:
            frames = sample_frames(video_path)
            frames = [f.resize((224, 224), Image.BILINEAR) for f in frames[:N_FRAMES_GEMMA3]]
            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": PROMPT})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=frames, return_tensors="pt")
        else:
            frames = sample_frames(video_path)
            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": PROMPT})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True)

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
        response = processor.decode(generated, skip_special_tokens=True)

        print(f"[{i+1:2d}/{len(test_samples)}] {true_class:<30s}  {response[:80]}...")
        records.append({"video": fname, "true_class": true_class, "response": response})

    # ── BERTScore classification ──────────────────────────────────────────────
    print("\nRunning BERTScore classification...")
    predictions = [r["response"] for r in records]
    pred_classes = bertscore_classify(predictions, class_refs,
                                      bs_model=args.bs_model,
                                      device=str(device))

    # ── compute accuracy ──────────────────────────────────────────────────────
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    for r, pred_cls in zip(records, pred_classes):
        r["predicted_class"] = pred_cls
        r["correct"] = pred_cls == r["true_class"]
        per_class_total[r["true_class"]] += 1
        if r["correct"]:
            per_class_correct[r["true_class"]] += 1

    total = len(records)
    correct_all = sum(per_class_correct.values())

    print(f"\n{'CLASS':<30s} {'CORRECT':>7}  {'TOTAL':>5}  {'ACC':>6}")
    print("-" * 60)
    for cls in sorted(class_keys):
        t = per_class_total[cls]
        c = per_class_correct[cls]
        print(f"{cls:<30s} {c:>7d}  {t:>5d}  {c/t:>5.1%}" if t else f"{cls:<30s} {'—':>7}  {0:>5d}  {'—':>6}")
    print("-" * 60)
    print(f"{'OVERALL':<30s} {correct_all:>7d}  {total:>5d}  {correct_all/total:>5.1%}")

    out = {
        "summary": {
            "overall_acc": correct_all / total,
            "per_class": {k: per_class_correct[k] / per_class_total[k]
                          for k in class_keys if per_class_total[k]},
        },
        "results": records,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
