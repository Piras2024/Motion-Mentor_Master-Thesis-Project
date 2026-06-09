"""
LoRA finetuning of Gemma4-E2B-it with video + Motion Q-Former (Option 4).

Architecture
------------
The motion stream is summarized by a small Q-Former (BLIP-2 style) rather than
concatenated as a long string of motion tokens:

    motion_indices  →  codebook[indices]  →  motion_features (T, 512)
                                                  │
                                                  ▼
    ┌─────────────────────────────────────────────────┐
    │ MotionQFormer                                   │
    │  - N learnable queries (e.g. 8)                 │
    │  - cross-attn: queries Q,  keys/values = motion │
    │  - self-attn over queries                       │
    │  - FFN                                          │
    └─────────────────────────────────────────────────┘
                          │
                          ▼  (N, LLM_hidden)
                spliced into input_embeds at the
                N placeholder positions <MQ_0>..<MQ_{N-1}>

Compared to finetune_gemma4_motion_proj.py:
  - Variable-length motion sequence → fixed N tokens (compression)
  - Queries *attend* to the motion sequence → learned temporal alignment
  - Trainable params: Q-Former (~1–2M) + LoRA (~24M).  Codebook frozen.

Usage
-----
    python finetune_gemma4_qformer.py [--epochs N] [--n-queries N] [--qf-layers L]
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

sys.path.insert(0, "/deck/users/mpiras/motion-agent")
from models import vqvae as vqvae_module
from options.option_llm import get_args_parser as get_vq_args

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "google/gemma-4-E2B-it"
VQVAE_CKPT    = "/deck/users/mpiras/motion-agent/experiments/v2_vqvae_noaug_nocls/vqvae_aug_cls_best.pth"
GUOFEATS_DIR  = "/deck/users/mpiras/dataset/hsmr_guofeats"
VIDEO_DIRS    = [
    "/deck/users/mpiras/dataset/rdls",
    "/deck/users/mpiras/dataset/squat_micc",
]
LABELS_PATH   = "/deck/users/mpiras/dataset/LLM_lables/labels_5var_reusable.json"
MEAN_PATH     = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH      = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

NB_CODE          = 512
N_QUERIES        = 8       # number of Q-Former slots prepended to the LLM input
QF_LAYERS        = 1       # cross-attn+self-attn+FFN blocks
QF_HEADS         = 8
N_VAL_PER_CLASS  = 5
N_TEST_PER_CLASS = 5

# Prompt: N consecutive placeholder tokens that get overwritten by Q-Former output
def make_prompt(n_queries: int) -> str:
    slots = "".join(f"<MQ_{i}>" for i in range(n_queries))
    return (
        f"Motion summary: {slots}\n\n"
        "First identify the strength exercise being performed, then carefully analyze "
        "the person's form throughout the movement. Identify any execution errors or "
        "form faults. If the form looks correct, say so. Be specific about what you "
        "observe and at which phase of the movement it occurs."
    )


# ── MotionQFormer ─────────────────────────────────────────────────────────────

class MotionQFormer(nn.Module):
    """
    Compresses a variable-length motion feature sequence into N fixed soft tokens
    via learnable queries that cross-attend to the motion stream.
    """

    def __init__(self, motion_dim: int, hidden_dim: int,
                 n_queries: int = N_QUERIES, n_layers: int = QF_LAYERS,
                 n_heads: int = QF_HEADS, dtype=torch.float32):
        super().__init__()
        self.n_queries = n_queries
        self.queries   = nn.Parameter(torch.randn(n_queries, hidden_dim, dtype=dtype) * 0.02)
        self.in_proj   = nn.Linear(motion_dim, hidden_dim, dtype=dtype)
        self.layers    = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_norm_q": nn.LayerNorm(hidden_dim, dtype=dtype),
                "cross_norm_kv": nn.LayerNorm(hidden_dim, dtype=dtype),
                "cross_attn":   nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dtype=dtype),
                "self_norm":    nn.LayerNorm(hidden_dim, dtype=dtype),
                "self_attn":    nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dtype=dtype),
                "ffn_norm":     nn.LayerNorm(hidden_dim, dtype=dtype),
                "ffn":          nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype),
                ),
            }))

    def forward(self, motion_feats: torch.Tensor) -> torch.Tensor:
        # motion_feats: (B, T, motion_dim)
        B = motion_feats.shape[0]
        kv = self.in_proj(motion_feats)                       # (B, T, hidden)
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)      # (B, N, hidden)

        for blk in self.layers:
            # cross-attn: queries ←─ motion
            qn  = blk["cross_norm_q"](q)
            kvn = blk["cross_norm_kv"](kv)
            attn_out, _ = blk["cross_attn"](qn, kvn, kvn, need_weights=False)
            q = q + attn_out

            # self-attn over queries
            qn = blk["self_norm"](q)
            attn_out, _ = blk["self_attn"](qn, qn, qn, need_weights=False)
            q = q + attn_out

            # FFN
            q = q + blk["ffn"](blk["ffn_norm"](q))

        return q                                              # (B, N, hidden)


# ── QFormerAwareEmbedding ─────────────────────────────────────────────────────

class QFormerAwareEmbedding(nn.Module):
    """
    Drop-in replacement for Gemma4's embed_tokens.

    Routing:
      - Normal token IDs       → frozen orig_embed
      - Motion-slot IDs        → corresponding Q-Former query output

    A pre-batch call to `set_motion_features(motion_feats)` computes the
    Q-Former output for the current batch, which is then spliced into the
    embedding output wherever motion-slot tokens appear in input_ids.
    """

    def __init__(self, orig_embed: nn.Module, qformer: MotionQFormer,
                 orig_vocab_size: int, n_queries: int):
        super().__init__()
        self.orig_embed      = orig_embed
        self.qformer         = qformer
        self.orig_vocab_size = orig_vocab_size
        self.n_queries       = n_queries
        self._current_q      = None  # (B, n_queries, hidden) — set per-batch

        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        # PEFT / resize_token_embeddings introspect this — return the underlying table
        return self.orig_embed.weight

    def set_motion_features(self, motion_feats: torch.Tensor):
        # motion_feats: (B, T, motion_dim) — continuous codebook vectors
        self._current_q = self.qformer(motion_feats)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_motion = input_ids >= self.orig_vocab_size                # (B, L)
        safe_ids  = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds    = self.orig_embed(safe_ids)                         # (B, L, hidden)

        if is_motion.any() and self._current_q is not None:
            slot_idx = (input_ids - self.orig_vocab_size).clamp(0, self.n_queries - 1)
            b_idx, l_idx = torch.where(is_motion)
            slot_at_pos = slot_idx[b_idx, l_idx]                      # (n_pos,)
            qf_emb = self._current_q[b_idx, slot_at_pos]              # (n_pos, hidden)
            embeds = embeds.clone()
            embeds[b_idx, l_idx] = qf_emb.to(embeds.dtype)
        return embeds


# ── VQ-VAE helpers ────────────────────────────────────────────────────────────

def load_vqvae(device: str, ckpt_path: str = VQVAE_CKPT):
    _argv, sys.argv = sys.argv, sys.argv[:1]
    a = get_vq_args()
    sys.argv = _argv
    a.nb_joints = 22; a.dataname = "t2m"
    a.nb_code = NB_CODE; a.code_dim = 512; a.output_emb_width = 512
    a.down_t = 2; a.stride_t = 2; a.width = 512; a.depth = 3
    a.dilation_growth_rate = 3; a.vq_act = "relu"; a.vq_norm = None

    net = vqvae_module.HumanVQVAE(a, NB_CODE, 512, 512, 2, 2, 512, 3, 3, "relu", None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)
    codebook = ckpt["net"]["vqvae.quantizer.codebook"]   # (512, 512)
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    return net, codebook, mean, std


@torch.no_grad()
def encode_motion_features(vqvae, codebook: torch.Tensor, guo_path: str,
                            mean: np.ndarray, std: np.ndarray, device: str) -> torch.Tensor:
    """Return the continuous per-timestep codebook vectors for a motion clip."""
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    indices = vqvae.encode(t).squeeze(0)                 # (T',)
    return codebook[indices.to(codebook.device)]          # (T', 512)


# ── dataset ───────────────────────────────────────────────────────────────────

def extract_class(filename, class_keys):
    stem = re.sub(r"\d+$", "", Path(filename).stem)
    for key in sorted(class_keys, key=len, reverse=True):
        if stem == key:
            return key
    return None


def build_by_class(class_labels):
    first_key = next(iter(class_labels))
    if first_key.endswith(".mp4"):
        labeled_fnames = set(class_labels.keys())
        class_keys = sorted({re.sub(r"\d+$", "", Path(k).stem) for k in class_labels})
    else:
        labeled_fnames = None
        class_keys = list(class_labels.keys())
    by_class = {k: [] for k in class_keys}
    for video_dir in VIDEO_DIRS:
        for fname in sorted(os.listdir(video_dir)):
            if not fname.endswith(".mp4"):
                continue
            if labeled_fnames is not None and fname not in labeled_fnames:
                continue
            stem = Path(fname).stem
            if not os.path.exists(os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")):
                continue
            key = extract_class(fname, class_keys)
            if key:
                by_class[key].append(os.path.join(video_dir, fname))
    return by_class


def stratified_split(by_class, n_val, n_test, seed=42):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for class_key, paths in by_class.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        test  += [(p, class_key) for p in shuffled[:n_test]]
        val   += [(p, class_key) for p in shuffled[n_test:n_test + n_val]]
        train += [(p, class_key) for p in shuffled[n_test + n_val:]]
    return train, val, test


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


class ExerciseDataset(Dataset):
    def __init__(self, samples, class_labels, processor, vqvae, codebook,
                 mean, std, device, prompt: str, jitter: bool = True):
        self.samples      = samples
        self.class_labels = class_labels
        self.processor    = processor
        self.vqvae        = vqvae
        self.codebook     = codebook
        self.mean         = mean
        self.std          = std
        self.device       = device
        self.prompt       = prompt
        self.jitter       = jitter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_key = self.samples[idx]
        fname = os.path.basename(video_path)
        label_list = self.class_labels.get(fname) or self.class_labels.get(class_key, [])
        response = random.choice(label_list)

        stem = Path(video_path).stem
        guo_path = os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")
        motion_feats = encode_motion_features(
            self.vqvae, self.codebook, guo_path, self.mean, self.std, self.device
        )                                                # (T, 512)

        video = load_video_array(video_path)

        full_conv = [
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": self.prompt}
            ]},
            {"role": "assistant", "content": response},
        ]
        prompt_conv = [
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": self.prompt}
            ]},
        ]

        full_text   = self.processor.apply_chat_template(full_conv,   tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)

        full_enc   = self.processor(text=full_text,   videos=video, return_tensors="pt")
        prompt_enc = self.processor(text=prompt_text, videos=video, return_tensors="pt")

        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[-1]
        labels     = input_ids.clone()
        labels[:prompt_len] = -100

        out = {"input_ids": input_ids, "labels": labels,
               "motion_features": motion_feats.cpu()}  # collated separately
        for k in full_enc:
            if k != "input_ids":
                out[k] = full_enc[k][0] if full_enc[k].dim() > 1 else full_enc[k]
        return out


def collate_fn(batch):
    # batch_size=1 throughout — keep motion_features as a separate leaf tensor
    out = {}
    for k, v in batch[0].items():
        if k == "motion_features":
            out[k] = v.unsqueeze(0)               # (1, T, 512)
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0) if v.dim() >= 1 else v
        else:
            out[k] = v
    return out


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--output-dir",  default="/deck/users/mpiras/paligemma_exercise/checkpoints_gemma4_qformer")
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--n-queries",   type=int,   default=N_QUERIES)
    parser.add_argument("--qf-layers",   type=int,   default=QF_LAYERS)
    parser.add_argument("--qf-heads",    type=int,   default=QF_HEADS)
    parser.add_argument("--wandb-project", default="exercise-form-vlm")
    parser.add_argument("--wandb-run",   default=None)
    parser.add_argument("--labels",      default=LABELS_PATH)
    parser.add_argument("--vqvae-ckpt",  default=VQVAE_CKPT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.labels) as f:
        class_labels = json.load(f)

    print("Loading VQ-VAE...")
    vqvae, codebook, mean, std = load_vqvae(device, args.vqvae_ckpt)
    codebook = codebook.to(device)
    print(f"VQ-VAE loaded. Codebook shape: {codebook.shape}")

    by_class = build_by_class(class_labels)
    total = sum(len(v) for v in by_class.values())
    print(f"Total videos with Guo features: {total}")
    train_samples, val_samples, test_samples = stratified_split(
        by_class, N_VAL_PER_CLASS, N_TEST_PER_CLASS
    )
    print(f"Split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test")

    split_path = os.path.join(args.output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({
            "seed": 42,
            "n_val_per_class": N_VAL_PER_CLASS,
            "n_test_per_class": N_TEST_PER_CLASS,
            "labels_file": args.labels,
            "n_queries": args.n_queries,
            "train": [[p, c] for p, c in train_samples],
            "val":   [[p, c] for p, c in val_samples],
            "test":  [[p, c] for p, c in test_samples],
        }, f, indent=2)
    print(f"Saved split → {split_path}")

    wandb.init(project=args.wandb_project, name=args.wandb_run,
               config=vars(args) | {"model": MODEL_ID, "nb_code": NB_CODE,
                                    "method": "qformer"})

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    motion_slot_tokens = [f"<MQ_{i}>" for i in range(args.n_queries)]
    n_added = processor.tokenizer.add_tokens(motion_slot_tokens)
    print(f"Added {n_added} motion-slot tokens to tokenizer.")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model_device = next(model.parameters()).device
    orig_vocab_size = model.model.language_model.embed_tokens.weight.shape[0]
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)

    # ── construct Q-Former + install awareness wrapper ────────────────────────
    orig_embed = model.model.language_model.embed_tokens
    hidden_dim = orig_embed.weight.shape[1]

    qformer = MotionQFormer(
        motion_dim=codebook.shape[1],
        hidden_dim=hidden_dim,
        n_queries=args.n_queries,
        n_layers=args.qf_layers,
        n_heads=args.qf_heads,
        dtype=torch.bfloat16,
    ).to(model_device)

    qf_embed = QFormerAwareEmbedding(
        orig_embed=orig_embed, qformer=qformer,
        orig_vocab_size=orig_vocab_size, n_queries=args.n_queries,
    ).to(model_device)
    model.model.language_model.embed_tokens = qf_embed
    print(f"Q-Former installed. Params: {sum(p.numel() for p in qformer.parameters()):,}")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Q-Former isn't a LoRA target → ensure it's trainable
    for p in model.model.model.language_model.embed_tokens.qformer.parameters():
        p.requires_grad_(True)

    model.print_trainable_parameters()

    prompt = make_prompt(args.n_queries)
    train_ds = ExerciseDataset(train_samples, class_labels, processor, vqvae, codebook,
                               mean, std, device, prompt, jitter=True)
    val_ds   = ExerciseDataset(val_samples,   class_labels, processor, vqvae, codebook,
                               mean, std, device, prompt, jitter=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    global_step = 0

    def _run_forward(batch):
        # Pull motion_features out, push into the Q-Former-aware embedding, then
        # forward the model on the remaining inputs.
        motion_feats = batch.pop("motion_features").to(model_device, torch.bfloat16)
        model.model.model.language_model.embed_tokens.set_motion_features(motion_feats)
        return model(**batch)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, n_steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = _run_forward(batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            train_loss += outputs.loss.item()
            n_steps += 1
            global_step += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                wandb.log({"train/loss_step": outputs.loss.item(),
                           "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}",
                             avg=f"{train_loss / n_steps:.4f}")

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                val_loss += _run_forward(batch).loss.item()

        avg_train = train_loss / max(n_steps, 1)
        avg_val   = val_loss   / max(len(val_loader), 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        wandb.log({"train/loss_epoch": avg_train, "val/loss": avg_val, "epoch": epoch},
                  step=global_step)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            # Q-Former is not part of the PEFT adapter — save it separately.
            torch.save(
                model.model.model.language_model.embed_tokens.qformer.state_dict(),
                os.path.join(save_path, "qformer.pt"),
            )
            # Also save the config needed to rebuild the module at eval time.
            with open(os.path.join(save_path, "qformer_config.json"), "w") as f:
                json.dump({
                    "n_queries":  args.n_queries,
                    "qf_layers":  args.qf_layers,
                    "qf_heads":   args.qf_heads,
                    "motion_dim": codebook.shape[1],
                    "hidden_dim": hidden_dim,
                }, f, indent=2)
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="train")
    ax.plot(range(1, args.epochs + 1), history["val_loss"],   marker="s", label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Gemma4-E2B + Motion Q-Former LoRA")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150); plt.close(fig)
    wandb.log({"training_curve": wandb.Image(plot_path)})
    wandb.finish()

    print(f"\nBest checkpoint: {os.path.join(args.output_dir, 'best')}  (val_loss={best_val_loss:.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
