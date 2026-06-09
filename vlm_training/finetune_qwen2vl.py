"""
LoRA finetuning of Qwen2-VL-2B-Instruct on exercise form videos.

Dataset: videos in rdls/ and squat_micc/ with class labels from class_labels.json.
Each training step pairs a video's sampled frames with a randomly chosen response
from the 5 available for its class (text augmentation). Frames are sampled with
slight temporal jitter each epoch (temporal augmentation).

Loss is computed only on the response tokens — prompt + image tokens are masked.

Usage:
    python finetune_qwen2vl.py [--epochs N] [--lr LR] [--output-dir DIR]
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

import cv2
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── paths ─────────────────────────────────────────────────────────────────────
VIDEO_DIRS = [
    "/deck/users/mpiras/dataset/rdls",
    "/deck/users/mpiras/dataset/squat_micc",
]
LABELS_PATH = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT = (
    "The images above are frames sampled uniformly from a single strength exercise repetition, "
    "in chronological order. First identify the exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or form faults. "
    "If the form looks correct, say so. Be specific about what you observe and at which phase "
    "of the movement it occurs."
)

N_FRAMES = 12


# ── dataset ───────────────────────────────────────────────────────────────────

def extract_class(filename: str, class_keys: list[str]) -> str | None:
    stem = re.sub(r"\d+$", "", Path(filename).stem)  # strip trailing number
    # try exact match first, then prefix match on sorted-by-length keys
    for key in sorted(class_keys, key=len, reverse=True):
        if stem == key:
            return key
    return None


FRAME_SIZE = 336  # resize all frames to this before the processor (controls token count)
# 336×336 with Qwen2-VL patch_size=14, merge_size=2 → 144 tokens/frame
# 12 frames × 144 = ~1728 visual tokens, well within the 32k context limit


class ExerciseDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], class_labels: dict, processor,
                 n_frames: int = N_FRAMES, jitter: bool = True):
        self.samples = samples        # list of (video_path, class_key)
        self.class_labels = class_labels
        self.processor = processor
        self.n_frames = n_frames
        self.jitter = jitter

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path: str) -> list[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base = np.linspace(0, total - 1, self.n_frames)

        if self.jitter:
            gap = (total - 1) / max(self.n_frames - 1, 1)
            noise = np.random.uniform(-gap * 0.15, gap * 0.15, size=self.n_frames)
            indices = np.clip(base + noise, 0, total - 1).astype(int)
        else:
            indices = base.astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
                frames.append(img)
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path, class_key = self.samples[idx]
        fname = os.path.basename(video_path)
        label_list = self.class_labels.get(fname) or self.class_labels.get(class_key, [])
        response = random.choice(label_list)
        frames = self._sample_frames(video_path)

        # build full conversation (prompt + response) for loss computation
        full_conv = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames]
                           + [{"type": "text", "text": PROMPT}],
            },
            {"role": "assistant", "content": response},
        ]
        # build prompt-only to measure where response tokens start
        prompt_conv = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames]
                           + [{"type": "text", "text": PROMPT}],
            }
        ]

        full_text = self.processor.apply_chat_template(
            full_conv, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )

        full_enc = self.processor(
            text=[full_text], images=frames, return_tensors="pt", padding=False
        )
        prompt_enc = self.processor(
            text=[prompt_text], images=frames, return_tensors="pt", padding=False
        )

        input_ids = full_enc["input_ids"][0]
        pixel_values = full_enc["pixel_values"]
        image_grid_thw = full_enc.get("image_grid_thw")
        mm_token_type_ids = full_enc.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[0]

        prompt_len = prompt_enc["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt + image tokens

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }


def collate_fn(batch):
    # pad input_ids and labels to the longest sequence in the batch
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 0

    input_ids = torch.stack([
        torch.nn.functional.pad(b["input_ids"], (0, max_len - b["input_ids"].shape[0]), value=pad_id)
        for b in batch
    ])
    labels = torch.stack([
        torch.nn.functional.pad(b["labels"], (0, max_len - b["labels"].shape[0]), value=-100)
        for b in batch
    ])
    attention_mask = (input_ids != pad_id).long()

    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0) \
        if batch[0]["image_grid_thw"] is not None else None
    mm_token_type_ids = torch.stack([
        torch.nn.functional.pad(b["mm_token_type_ids"], (0, max_len - b["mm_token_type_ids"].shape[0]), value=0)
        for b in batch
    ]) if batch[0]["mm_token_type_ids"] is not None else None

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
    if image_grid_thw is not None:
        out["image_grid_thw"] = image_grid_thw
    if mm_token_type_ids is not None:
        out["mm_token_type_ids"] = mm_token_type_ids
    return out


# ── build sample list ─────────────────────────────────────────────────────────

def build_samples(class_labels: dict) -> dict[str, list[str]]:
    """Returns {class_key: [video_path, ...]}. Handles both class-keyed and file-keyed formats."""
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
            class_key = extract_class(fname, class_keys)
            if class_key is None:
                continue
            by_class[class_key].append(os.path.join(video_dir, fname))
    return by_class


def stratified_split(
    by_class: dict[str, list[str]],
    n_val_per_class: int,
    n_test_per_class: int,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    For each class, randomly pick n_test_per_class for test, n_val_per_class
    for val, and use the rest for train. Returns flat lists of (path, class_key).
    """
    rng = random.Random(seed)
    train, val, test = [], [], []
    for class_key, paths in by_class.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        test_paths = shuffled[:n_test_per_class]
        val_paths = shuffled[n_test_per_class:n_test_per_class + n_val_per_class]
        train_paths = shuffled[n_test_per_class + n_val_per_class:]
        test  += [(p, class_key) for p in test_paths]
        val   += [(p, class_key) for p in val_paths]
        train += [(p, class_key) for p in train_paths]
    return train, val, test


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--output-dir", default="/deck/users/mpiras/paligemma_exercise/checkpoints_qwen_150")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--val-per-class", type=int, default=5,
                        help="Validation examples per class (default: 5)")
    parser.add_argument("--test-per-class", type=int, default=5,
                        help="Test examples per class (default: 5)")
    parser.add_argument("--wandb-project", default="exercise-form-vlm",
                        help="W&B project name (default: exercise-form-vlm)")
    parser.add_argument("--wandb-run", default=None,
                        help="W&B run name (default: auto)")
    parser.add_argument("--labels", default=LABELS_PATH,
                        help="Labels JSON (class-keyed or file-keyed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.labels) as f:
        class_labels = json.load(f)

    print("Building sample list...")
    by_class = build_samples(class_labels)
    total = sum(len(v) for v in by_class.values())
    print(f"Total videos: {total}")

    train_samples, val_samples, test_samples = stratified_split(
        by_class, args.val_per_class, args.test_per_class
    )
    print(f"Split: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test")
    print(f"       ({args.val_per_class} val + {args.test_per_class} test per class, all classes covered)")

    split_path = os.path.join(args.output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({
            "seed": 42,
            "n_val_per_class": args.val_per_class,
            "n_test_per_class": args.test_per_class,
            "labels_file": args.labels,
            "train": [[p, c] for p, c in train_samples],
            "val":   [[p, c] for p, c in val_samples],
            "test":  [[p, c] for p, c in test_samples],
        }, f, indent=2)
    print(f"Saved split → {split_path}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "model": MODEL_ID,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "n_frames": N_FRAMES,
            "frame_size": FRAME_SIZE,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
        },
    )

    print(f"Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    train_ds = ExerciseDataset(list(train_samples), class_labels, processor, jitter=True)
    val_ds = ExerciseDataset(list(val_samples), class_labels, processor, jitter=False)
    test_ds = ExerciseDataset(list(test_samples), class_labels, processor, jitter=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
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

        # flush remaining gradients
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_train = train_loss / max(n_steps, 1)
        avg_val = val_loss / max(len(val_loader), 1)
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
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

    # ── training curve ────────────────────────────────────────────────────────
    epochs_range = range(1, args.epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_range, history["train_loss"], marker="o", label="train loss")
    ax.plot(epochs_range, history["val_loss"],   marker="s", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve — Qwen2-VL-2B LoRA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to {plot_path}")
    wandb.log({"training_curve": wandb.Image(plot_path)})

    wandb.finish()
    print(f"\nBest checkpoint: {os.path.join(args.output_dir, 'best')}  (val_loss={best_val_loss:.4f})")
    print("Run evaluate.py to get keyword-based accuracy on the held-out test set.")
    print("Done.")


if __name__ == "__main__":
    main()
