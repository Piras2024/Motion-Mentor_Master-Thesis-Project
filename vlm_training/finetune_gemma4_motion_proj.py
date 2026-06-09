"""
LoRA finetuning of Gemma4-E2B-it with video + VQ-VAE motion tokens (Option 3).

A learnable projection layer maps frozen VQ-VAE codebook embeddings (512-dim)
into Gemma4's token embedding space. Motion token IDs in the input are routed
through this projection instead of the standard embed_tokens table, so the
model receives continuous, VQVAE-informed representations from the start —
rather than random initialised vectors as in Option 2.

Architecture change:
    embed_tokens  →  MotionAwareEmbedding
        - original token IDs  →  original frozen embed_tokens (unchanged)
        - motion token IDs    →  motion_proj( codebook[id - orig_vocab_size] )

Only motion_proj + LoRA adapters are trained. Original embeddings are frozen.
lm_head is not resized (the model never generates motion tokens).

Usage:
    python finetune_gemma4_motion_proj.py [--epochs N]
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

# ── motion-agent imports ──────────────────────────────────────────────────────
sys.path.insert(0, "/deck/users/mpiras/motion-agent")
from models import vqvae as vqvae_module
from options.option_llm import get_args_parser as get_vq_args

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "google/gemma-4-E2B-it"
VQVAE_CKPT    = "/deck/users/mpiras/motion-agent/ckpt/vqvae.pth"
GUOFEATS_DIR  = "/deck/users/mpiras/dataset/hsmr_guofeats"
VIDEO_DIRS    = [
    "/deck/users/mpiras/dataset/rdls",
    "/deck/users/mpiras/dataset/squat_micc",
]
LABELS_PATH   = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"
MEAN_PATH     = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH      = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

NB_CODE          = 512
N_VAL_PER_CLASS  = 5
N_TEST_PER_CLASS = 5

PROMPT_TEMPLATE = (
    "Motion token sequence: {motion_tokens}\n\n"
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)


# ── MotionAwareEmbedding ──────────────────────────────────────────────────────

class MotionAwareEmbedding(nn.Module):
    """
    Drop-in replacement for Gemma4's embed_tokens.

    - Normal token IDs  →  original frozen embedding table (behaviour unchanged)
    - Motion token IDs  →  motion_proj( codebook[id - orig_vocab_size] )

    The VQVAE codebook is registered as a frozen buffer.
    Only motion_proj is trainable.
    """

    def __init__(self, orig_embed: nn.Module, codebook: torch.Tensor,
                 orig_vocab_size: int):
        super().__init__()
        self.orig_embed     = orig_embed
        self.orig_vocab_size = orig_vocab_size

        hidden_dim = orig_embed.weight.shape[1]
        vqvae_dim  = codebook.shape[1]

        # trainable projection: VQ-VAE embedding space → Gemma4 token space
        self.motion_proj = nn.Linear(vqvae_dim, hidden_dim, bias=False,
                                     dtype=orig_embed.weight.dtype)
        nn.init.normal_(self.motion_proj.weight, std=0.02)

        # frozen VQ-VAE codebook
        self.register_buffer("codebook", codebook.to(orig_embed.weight.dtype))

        # freeze original embeddings entirely
        for p in self.orig_embed.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        """Expose weight attribute so PEFT / resize_token_embeddings can inspect it."""
        return self.orig_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_motion = input_ids >= self.orig_vocab_size

        # safe lookup — clamp motion IDs so orig_embed never sees OOB indices
        safe_ids = input_ids.clamp(max=self.orig_vocab_size - 1)
        embeds   = self.orig_embed(safe_ids)             # uses original module (incl. any scaling)

        if is_motion.any():
            code_idx     = (input_ids[is_motion] - self.orig_vocab_size).clamp(0)
            motion_vecs  = self.codebook[code_idx]       # (N, vqvae_dim)
            motion_emb   = self.motion_proj(motion_vecs) # (N, hidden_dim)

            # non-inplace replacement (preserves autograd graph)
            embeds = embeds.clone()
            embeds[is_motion] = motion_emb.to(embeds.dtype)

        return embeds


# ── VQ-VAE helpers ────────────────────────────────────────────────────────────

def load_vqvae(device: str, ckpt_path: str = VQVAE_CKPT) -> tuple[nn.Module, torch.Tensor, np.ndarray, np.ndarray]:
    _argv, sys.argv = sys.argv, sys.argv[:1]
    args = get_vq_args()
    sys.argv = _argv

    args.nb_joints = 22; args.dataname = "t2m"
    args.nb_code = NB_CODE; args.code_dim = 512; args.output_emb_width = 512
    args.down_t = 2; args.stride_t = 2; args.width = 512; args.depth = 3
    args.dilation_growth_rate = 3; args.vq_act = "relu"; args.vq_norm = None

    net = vqvae_module.HumanVQVAE(args, NB_CODE, 512, 512, 2, 2, 512, 3, 3, "relu", None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    codebook = ckpt["net"]["vqvae.quantizer.codebook"]  # (512, 512)
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    return net, codebook, mean, std


@torch.no_grad()
def encode_motion(vqvae: nn.Module, guo_path: str,
                  mean: np.ndarray, std: np.ndarray, device: str) -> list[int]:
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    return vqvae.encode(t).squeeze(0).cpu().tolist()


def motion_tokens_str(indices: list[int]) -> str:
    return "".join(f"<Motion_{i}>" for i in indices)


# ── dataset ───────────────────────────────────────────────────────────────────

def extract_class(filename: str, class_keys: list[str]) -> str | None:
    stem = re.sub(r"\d+$", "", Path(filename).stem)
    for key in sorted(class_keys, key=len, reverse=True):
        if stem == key:
            return key
    return None


def build_by_class(class_labels: dict) -> dict[str, list[str]]:
    # Support file-keyed {"video1.mp4": [labels]} and class-keyed {"class": [labels]}
    first_key = next(iter(class_labels))
    if first_key.endswith(".mp4"):
        labeled_fnames = set(class_labels.keys())
        class_keys = sorted({re.sub(r"\d+$", "", Path(k).stem) for k in class_labels})
    else:
        labeled_fnames = None
        class_keys = list(class_labels.keys())
    by_class: dict[str, list[str]] = {k: [] for k in class_keys}
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


class ExerciseDataset(Dataset):
    def __init__(self, samples, class_labels, processor, vqvae, mean, std, device, jitter=True):
        self.samples      = samples
        self.class_labels = class_labels
        self.processor    = processor
        self.vqvae        = vqvae
        self.mean         = mean
        self.std          = std
        self.device       = device
        self.jitter       = jitter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_key = self.samples[idx]
        fname = os.path.basename(video_path)
        label_list = self.class_labels.get(fname) or self.class_labels.get(class_key, [])
        response = random.choice(label_list)

        stem     = Path(video_path).stem
        guo_path = os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")
        indices  = encode_motion(self.vqvae, guo_path, self.mean, self.std, self.device)
        mot_str  = motion_tokens_str(indices)
        prompt   = PROMPT_TEMPLATE.format(motion_tokens=mot_str)

        video = load_video_array(video_path)

        full_conv = [
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": response},
        ]
        prompt_conv = [
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt}
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

        out = {"input_ids": input_ids, "labels": labels}
        for k in full_enc:
            if k != "input_ids":
                out[k] = full_enc[k][0] if full_enc[k].dim() > 1 else full_enc[k]
        return out


def collate_fn(batch):
    return {k: batch[0][k].unsqueeze(0) if batch[0][k].dim() >= 1 else batch[0][k]
            for k in batch[0]}


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--output-dir",   default="/deck/users/mpiras/paligemma_exercise/checkpoints_gemma4_motion_proj")
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--wandb-project", default="exercise-form-vlm")
    parser.add_argument("--wandb-run",    default=None)
    parser.add_argument("--labels",       default=LABELS_PATH)
    parser.add_argument("--vqvae-ckpt",   default=VQVAE_CKPT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.labels) as f:
        class_labels = json.load(f)

    print("Loading VQ-VAE...")
    vqvae, codebook, mean, std = load_vqvae(device, args.vqvae_ckpt)
    print(f"VQ-VAE loaded. Codebook shape: {codebook.shape}")

    print("Building sample list...")
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
            "train": [[p, c] for p, c in train_samples],
            "val":   [[p, c] for p, c in val_samples],
            "test":  [[p, c] for p, c in test_samples],
        }, f, indent=2)
    print(f"Saved split → {split_path}")

    wandb.init(project=args.wandb_project, name=args.wandb_run,
               config=vars(args) | {"model": MODEL_ID, "nb_code": NB_CODE,
                                    "method": "projection"})

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # add motion tokens to tokenizer so they can appear in prompts
    motion_token_strs = [f"<Motion_{i}>" for i in range(NB_CODE)]
    n_added = processor.tokenizer.add_tokens(motion_token_strs)
    print(f"Added {n_added} motion tokens to tokenizer.")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model_device = next(model.parameters()).device

    # record original vocab size BEFORE any resize
    orig_vocab_size = model.model.language_model.embed_tokens.weight.shape[0]

    # resize so the tokenizer's new IDs are valid (embed_tokens gets extra rows,
    # but we'll override them via MotionAwareEmbedding)
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)

    # ── replace embed_tokens with MotionAwareEmbedding ───────────────────────
    orig_embed = model.model.language_model.embed_tokens
    motion_embed = MotionAwareEmbedding(
        orig_embed    = orig_embed,
        codebook      = codebook.to(model_device),
        orig_vocab_size = orig_vocab_size,
    ).to(model_device)
    model.model.language_model.embed_tokens = motion_embed
    print(f"MotionAwareEmbedding installed. "
          f"motion_proj params: {motion_embed.motion_proj.weight.numel():,}")

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

    # motion_proj is not a LoRA target — ensure it's trainable
    for p in model.model.model.language_model.embed_tokens.motion_proj.parameters():
        p.requires_grad_(True)

    model.print_trainable_parameters()

    train_ds = ExerciseDataset(train_samples, class_labels, processor,
                               vqvae, mean, std, device, jitter=True)
    val_ds   = ExerciseDataset(val_samples,   class_labels, processor,
                               vqvae, mean, std, device, jitter=False)

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
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, n_steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
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
                val_loss += model(**batch).loss.item()

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
            # also save motion_proj separately (not part of LoRA checkpoint)
            torch.save(
                model.model.model.language_model.embed_tokens.motion_proj.state_dict(),
                os.path.join(save_path, "motion_proj.pt")
            )
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

    # ── training curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="train")
    ax.plot(range(1, args.epochs + 1), history["val_loss"],   marker="s", label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Gemma4-E2B + Motion Projection LoRA")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150); plt.close(fig)
    wandb.log({"training_curve": wandb.Image(plot_path)})
    wandb.finish()

    print(f"\nBest checkpoint: {os.path.join(args.output_dir, 'best')}  (val_loss={best_val_loss:.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
