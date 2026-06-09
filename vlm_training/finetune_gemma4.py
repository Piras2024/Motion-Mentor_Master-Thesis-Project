"""
LoRA finetuning of Gemma4-E2B-it on exercise form videos.

Gemma4 has a native video processor (32 frames, do_sample_frames=True) so we
pass the full video as a numpy array and let the processor handle sampling.
Loss is masked on prompt tokens using the prompt-length approach.

Usage:
    python finetune_gemma4.py [--epochs N] [--output-dir DIR]
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E2B-it"
VIDEO_DIRS = [
    "/deck/users/mpiras/dataset/rdls",
    "/deck/users/mpiras/dataset/squat_micc",
]
LABELS_PATH = "/deck/users/mpiras/dataset/LLM_lables/class_labels_pooled_150.json"

N_VAL_PER_CLASS  = 5
N_TEST_PER_CLASS = 5

PROMPT = (
    "First identify the strength exercise being performed, then carefully analyze "
    "the person's form throughout the movement. Identify any execution errors or "
    "form faults. If the form looks correct, say so. Be specific about what you "
    "observe and at which phase of the movement it occurs."
)


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


def load_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)  # (T, H, W, 3)


class ExerciseDataset(Dataset):
    def __init__(self, samples, class_labels, processor, jitter=True):
        self.samples = samples
        self.class_labels = class_labels
        self.processor = processor
        self.jitter = jitter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_key = self.samples[idx]
        fname = os.path.basename(video_path)
        label_list = self.class_labels.get(fname) or self.class_labels.get(class_key, [])
        response = random.choice(label_list)
        video = load_video(video_path)  # (T, H, W, 3)

        full_conv = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": PROMPT}],
            },
            {"role": "assistant", "content": response},
        ]
        prompt_conv = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": PROMPT}],
            }
        ]

        full_text   = self.processor.apply_chat_template(full_conv,   tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)

        full_enc   = self.processor(text=full_text,   videos=video, return_tensors="pt")
        prompt_enc = self.processor(text=prompt_text, videos=video, return_tensors="pt")

        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        out = {"input_ids": input_ids, "labels": labels}
        for k in full_enc:
            if k not in ("input_ids",):
                out[k] = full_enc[k][0] if full_enc[k].dim() > 1 else full_enc[k]
        return out


def collate_fn(batch):
    # batch_size=1, no padding needed
    return {k: batch[0][k].unsqueeze(0) if batch[0][k].dim() >= 1 else batch[0][k]
            for k in batch[0]}


# ── training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--output-dir", default="/deck/users/mpiras/paligemma_exercise/checkpoints_gemma4")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--wandb-project", default="exercise-form-vlm")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--labels", default=LABELS_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.labels) as f:
        class_labels = json.load(f)

    print("Building sample list...")
    by_class = build_by_class(class_labels)
    total = sum(len(v) for v in by_class.values())
    print(f"Total videos: {total}")

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

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args) | {"model": MODEL_ID},
    )

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # regex matched against full module path — restricts LoRA to LLM only,
        # avoiding Gemma4ClippableLinear layers in the vision tower
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = ExerciseDataset(train_samples, class_labels, processor, jitter=True)
    val_ds   = ExerciseDataset(val_samples,   class_labels, processor, jitter=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    device = next(model.parameters()).device
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, n_steps = 0.0, 0

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

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
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
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            processor.save_pretrained(os.path.join(args.output_dir, "best"))
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

    # ── training curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="train loss")
    ax.plot(range(1, args.epochs + 1), history["val_loss"],   marker="s", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve — Gemma4-E2B LoRA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    wandb.log({"training_curve": wandb.Image(plot_path)})
    wandb.finish()

    print(f"\nBest checkpoint: {os.path.join(args.output_dir, 'best')}  (val_loss={best_val_loss:.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
