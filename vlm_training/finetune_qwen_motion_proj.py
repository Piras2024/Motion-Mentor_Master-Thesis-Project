"""
LoRA finetuning of Qwen2-VL-2B with video frames + VQ-VAE projection (Option 3).

Motion token IDs are routed through a frozen VQ-VAE codebook and a learned
Linear(512, hidden_dim) projection instead of a trained embedding table.
Only motion_proj + LoRA adapters are trained. Saves motion_proj.pt separately.

Usage:
    python finetune_qwen_motion_proj.py [--epochs N] [--vqvae-ckpt PATH]
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
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.insert(0, "/deck/users/mpiras/motion-agent")
from models import vqvae as vqvae_module

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"
VQVAE_CKPT   = "/deck/users/mpiras/motion-agent/ckpt/vqvae.pth"
GUOFEATS_DIR = "/deck/users/mpiras/dataset/hsmr_guofeats"
MEAN_PATH    = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH     = "/deck/users/mpiras/motion-agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
VIDEO_DIRS   = ["/deck/users/mpiras/dataset/rdls", "/deck/users/mpiras/dataset/squat_micc"]
LABELS_PATH  = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"

NB_CODE          = 512
N_FRAMES         = 12
FRAME_SIZE       = 336
N_VAL_PER_CLASS  = 5
N_TEST_PER_CLASS = 5

PROMPT_TEMPLATE = (
    "Motion token sequence: {motion_tokens}\n\n"
    "The images above are frames sampled uniformly from a single strength exercise "
    "repetition, in chronological order. First identify the exercise being performed, "
    "then carefully analyze the person's form throughout the movement. Identify any "
    "execution errors or form faults. If the form looks correct, say so. Be specific "
    "about what you observe and at which phase of the movement it occurs."
)


# ── MotionAwareEmbedding ──────────────────────────────────────────────────────

class MotionAwareEmbedding(nn.Module):
    def __init__(self, orig_embed: nn.Module, codebook: torch.Tensor, orig_vocab_size: int):
        super().__init__()
        self.orig_embed      = orig_embed
        self.orig_vocab_size = orig_vocab_size
        hidden_dim           = orig_embed.weight.shape[1]
        self.motion_proj     = nn.Linear(codebook.shape[1], hidden_dim, bias=False,
                                         dtype=orig_embed.weight.dtype)
        nn.init.normal_(self.motion_proj.weight, std=0.02)
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


# ── VQ-VAE ────────────────────────────────────────────────────────────────────

def load_vqvae(device: str, ckpt_path: str = VQVAE_CKPT):
    import argparse as ap
    a = ap.Namespace(dataname="hml3d", nb_code=NB_CODE, code_dim=512,
                     output_emb_width=512, down_t=2, stride_t=2, width=512,
                     depth=3, dilation_growth_rate=3, vq_act="relu",
                     vq_norm=None, quantizer="ema_reset", mu=0.99)
    net = vqvae_module.HumanVQVAE(a, NB_CODE, 512, 512, 2, 2, 512, 3, 3, "relu", None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    codebook = ckpt["net"]["vqvae.quantizer.codebook"]
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    return net.eval().to(device), codebook, mean, std


@torch.no_grad()
def encode_motion(vqvae, guo_path, mean, std, device):
    feats = (np.load(guo_path).astype(np.float32) - mean) / std
    t = torch.from_numpy(feats).unsqueeze(0).to(device)
    return vqvae.encode(t).squeeze(0).cpu().tolist()


# ── dataset helpers ───────────────────────────────────────────────────────────

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

    def _sample_frames(self, video_path):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base  = np.linspace(0, total - 1, N_FRAMES)
        if self.jitter:
            gap     = (total - 1) / max(N_FRAMES - 1, 1)
            indices = np.clip(base + np.random.uniform(-gap * 0.15, gap * 0.15, N_FRAMES),
                              0, total - 1).astype(int)
        else:
            indices = base.astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR))
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path, class_key = self.samples[idx]
        fname_key = os.path.basename(video_path)
        label_list = self.class_labels.get(fname_key) or self.class_labels.get(class_key, [])
        response = random.choice(label_list)
        frames   = self._sample_frames(video_path)

        stem     = Path(video_path).stem
        guo_path = os.path.join(GUOFEATS_DIR, f"HSMR-{stem}_guofeats.npy")
        indices  = encode_motion(self.vqvae, guo_path, self.mean, self.std, self.device)
        mot_str  = "".join(f"<Motion_{i}>" for i in indices)
        prompt   = PROMPT_TEMPLATE.format(motion_tokens=mot_str)

        image_content = [{"type": "image", "image": img} for img in frames]
        full_conv = [
            {"role": "user",      "content": image_content + [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": response},
        ]
        prompt_conv = [
            {"role": "user",      "content": image_content + [{"type": "text", "text": prompt}]},
        ]

        full_text   = self.processor.apply_chat_template(full_conv,   tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)

        full_enc   = self.processor(text=[full_text],   images=frames, return_tensors="pt", padding=False)
        prompt_enc = self.processor(text=[prompt_text], images=frames, return_tensors="pt", padding=False)

        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[-1]
        labels     = input_ids.clone()
        labels[:prompt_len] = -100

        mm = full_enc.get("mm_token_type_ids")
        return {
            "input_ids":         input_ids,
            "labels":            labels,
            "pixel_values":      full_enc["pixel_values"],
            "image_grid_thw":    full_enc.get("image_grid_thw"),
            "mm_token_type_ids": mm[0] if mm is not None else None,
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(b["input_ids"], (0, max_len - b["input_ids"].shape[0]))
        for b in batch])
    labels = torch.stack([
        torch.nn.functional.pad(b["labels"], (0, max_len - b["labels"].shape[0]), value=-100)
        for b in batch])
    out = {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": (input_ids != 0).long(),
        "pixel_values":   torch.cat([b["pixel_values"] for b in batch], dim=0),
    }
    if batch[0]["image_grid_thw"] is not None:
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    if batch[0]["mm_token_type_ids"] is not None:
        out["mm_token_type_ids"] = torch.stack([
            torch.nn.functional.pad(b["mm_token_type_ids"], (0, max_len - b["mm_token_type_ids"].shape[0]))
            for b in batch])
    return out


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--output-dir",   default="/deck/users/mpiras/paligemma_exercise/checkpoints_qwen_motion_proj")
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

    by_class = build_by_class(class_labels)
    print(f"Total videos with guofeats: {sum(len(v) for v in by_class.values())}")
    train_samples, val_samples, test_samples = stratified_split(by_class, N_VAL_PER_CLASS, N_TEST_PER_CLASS)
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
               config=vars(args) | {"model": MODEL_ID, "nb_code": NB_CODE})

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.add_tokens([f"<Motion_{i}>" for i in range(NB_CODE)])

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.resize_token_embeddings(len(processor.tokenizer))
    orig_vocab_size = model.get_input_embeddings().weight.shape[0] - NB_CODE
    orig_embed      = model.get_input_embeddings()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model_device = next(model.parameters()).device

    motion_embed = MotionAwareEmbedding(orig_embed, codebook.to(model_device), orig_vocab_size).to(model_device)
    model.set_input_embeddings(motion_embed)
    print(f"MotionAwareEmbedding installed. motion_proj params: {motion_embed.motion_proj.weight.numel():,}")

    # ensure motion_proj is trainable (not frozen by PEFT)
    for p in model.get_input_embeddings().motion_proj.parameters():
        p.requires_grad_(True)

    model.print_trainable_parameters()

    train_ds = ExerciseDataset(train_samples, class_labels, processor, vqvae, mean, std, device, jitter=True)
    val_ds   = ExerciseDataset(val_samples,   class_labels, processor, vqvae, mean, std, device, jitter=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1)

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
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            train_loss += loss.item() * args.grad_accum
            n_steps += 1; global_step += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                wandb.log({"train/loss_step": loss.item() * args.grad_accum,
                           "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}",
                             avg=f"{train_loss/n_steps:.4f}")

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()

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
        wandb.log({"train/loss_epoch": avg_train, "val/loss": avg_val, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            torch.save(model.get_input_embeddings().motion_proj.state_dict(),
                       os.path.join(save_path, "motion_proj.pt"))
            print(f"  → saved best (val={best_val_loss:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="train")
    ax.plot(range(1, args.epochs + 1), history["val_loss"],   marker="s", label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Qwen2-VL + Motion Proj LoRA"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)
    plt.close(fig)
    wandb.finish()
    print(f"Done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
