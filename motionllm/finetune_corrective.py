"""
finetune_corrective.py

Fine-tune the MotionLLM m2t LoRA adapter on squat corrective-feedback labels.

The dataset is expected as:
  --guofeats-dir   directory of *_guofeats.npy files (T-1, 263)
  --labels-jsonl   JSONL with {"id": "...", "output": "corrective text"}
                   where each id maps to HSMR-{id}_guofeats.npy in guofeats-dir

Loads the pretrained motionllm.pth, freezes the t2m LoRA and VQ-VAE,
then fine-tunes the m2t LoRA on the corrective labels.

Reproduces the best motion-only model (`motionllm_corrective_v1`). Run from
inside the `motionllm/` directory; defaults resolve to the repo's in-domain
VQ-VAE, pretrained checkpoint and pooled labels. Only --guofeats-dir (the
pre-extracted training features) must be supplied.

Usage:
  python finetune_corrective.py \\
    --guofeats-dir /path/to/guofeats \\
    --out-dir      experiments/corrective \\
    --epochs       40 \\
    --batch-size   8 \\
    --scheduler    onecycle --lr 1e-5 --max-lr 3e-5 \\
    --augmentations speed \\
    --device       cuda:0
"""

import sys
import os
import json
import logging
import argparse
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── run from the motion-agent directory ───────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from models.mllm import MotionLLM
from options.option_llm import get_args_parser as get_llm_args

CORRECTIVE_INSTRUCTION = "### Instruction:\nAnalyze this motion and identify the main technical error. Provide corrective feedback. If the technique is correct, confirm it.\n\n"

def get_instruction(sample_id: str) -> str:
    return CORRECTIVE_INSTRUCTION


# ── augmentation ──────────────────────────────────────────────────────────────
import utils.augmentations as _augs

# Default augmentation set used when --augmentations is not specified
_DEFAULT_AUG_FNS = [_augs.mirror, _augs.reverse]

# Augmentation probabilities (per function, applied independently)
_AUG_PROBS = {
    "mirror":  0.5,
    "reverse": 0.2,
    "speed":   0.5,
    "scale":   0.5,
    "noise":   0.5,
    "crop":    0.3,
}

_ACTIVE_AUG_FNS: list = []   # set at startup from --augmentations arg
_SUSTAINED_CLASSES: set = set()  # classes that additionally receive reverse aug


def _build_aug_fns(aug_names: list) -> list:
    return [getattr(_augs, name) for name in aug_names if hasattr(_augs, name)]


def augment_motion(motion: np.ndarray, cls: str = "") -> np.ndarray:
    """Apply active augmentations stochastically.

    Classes in _SUSTAINED_CLASSES additionally receive reverse augmentation
    because their error is present throughout the movement (reversing is a valid
    sample of the same error). Phase-dependent classes only get the base set.
    """
    fns = _ACTIVE_AUG_FNS if _ACTIVE_AUG_FNS else _DEFAULT_AUG_FNS
    for fn in fns:
        prob = _AUG_PROBS.get(fn.__name__, 0.5)
        if np.random.rand() < prob:
            motion = fn(motion)
    # extra reverse for sustained-error classes
    if cls in _SUSTAINED_CLASSES:
        prob = _AUG_PROBS.get("reverse", 0.2)
        if np.random.rand() < prob:
            motion = _augs.reverse(motion)
    return motion


# ── dataset ───────────────────────────────────────────────────────────────────

def truncate_sentences(text: str, n: int) -> str:
    """Keep only the first n sentences (period-terminated)."""
    if n <= 0:
        return text
    parts = text.split(". ")
    kept = parts[:n]
    tail = ". ".join(kept)
    return tail if tail.endswith(".") else tail + "."


def _load_labels(path: str) -> dict:
    """
    Load labels from either format:
      - JSONL: {"id": "head_position1", "output": "..."}
      - JSON dict: {"head_position1.mp4": "..."}
    Returns a dict mapping sample_id (no extension, no path) → caption.
    """
    with open(path) as f:
        raw = f.read().strip()

    # JSON dict format
    if raw.startswith("{"):
        data = json.loads(raw)
        return {k.replace(".mp4", "").replace("HSMR-", ""): v for k, v in data.items()}

    # JSONL format
    result = {}
    for line in raw.splitlines():
        item = json.loads(line)
        result[item["id"]] = item["output"]
    return result


class SquatCorrectiveDataset(Dataset):
    """
    Pairs each *_guofeats.npy with its corrective-feedback string.
    Accepts both JSONL and JSON-dict label files.
    Returns (motion_array, caption) where motion_array is already normalised.

    Label priority (highest first):
      1. sample_labels  — per-sample dict {sample_id: [v1, v2, ...]}  (new)
      2. class_labels   — per-class  dict {class: [v1, v2, ...]}
      3. original label from labels_jsonl
    """

    def __init__(self, guofeats_dir: str, labels_jsonl: str, mean: np.ndarray, std: np.ndarray,
                 augment: bool = False, label_sentences: int = 0,
                 class_labels: dict = None, class_labels_random: bool = True,
                 sample_labels: dict = None):
        self.guofeats_dir        = guofeats_dir
        self.mean                = mean
        self.std                 = std
        self.augment             = augment
        self.class_labels        = class_labels
        self.class_labels_random = class_labels_random
        self.sample_labels       = sample_labels   # {sample_id_with_ext: [v1..v5]}
        self.samples = []  # list of (npy_path, caption_str, class_str)

        for sample_id, caption in _load_labels(labels_jsonl).items():
            if label_sentences > 0:
                caption = truncate_sentences(caption, label_sentences)
            # class = ID with trailing digits stripped  e.g. "head_position1" → "head_position"
            cls = sample_id.rstrip("0123456789")
            # try bare name first, then squat_ prefix (hsmr_guofeats uses squat_ prefix)
            candidates = [
                os.path.join(guofeats_dir, f"HSMR-{sample_id}_guofeats.npy"),
                os.path.join(guofeats_dir, f"HSMR-squat_{sample_id}_guofeats.npy"),
            ]
            npy_path = next((p for p in candidates if os.path.exists(p)), None)
            if npy_path:
                self.samples.append((npy_path, caption, cls))
            else:
                print(f"[warn] missing guofeats for id={sample_id}: {candidates[0]}")

        classes = sorted({s[2] for s in self.samples})
        print(f"Dataset: {len(self.samples)} samples, {len(classes)} classes: {classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, caption, cls = self.samples[idx]
        motion = np.load(npy_path).astype(np.float32)          # (T-1, 263)
        if self.augment:
            motion = augment_motion(motion, cls)
        motion = (motion - self.mean) / self.std                # normalise

        # label priority: per-sample variants > per-class variants > original
        sample_id = os.path.basename(npy_path).replace("HSMR-", "").replace("_guofeats.npy", "")
        sample_key = sample_id + ".mp4"
        if self.sample_labels and sample_key in self.sample_labels:
            variants = self.sample_labels[sample_key]
            caption = random.choice(variants) if self.class_labels_random else variants[0]
        elif self.class_labels and cls in self.class_labels:
            variants = self.class_labels[cls]
            caption = random.choice(variants) if self.class_labels_random else variants[0]

        return motion, caption, sample_id


def collate_fn(batch):
    """Return a list of motions (variable-length), captions, and sample_ids."""
    motions, captions, sample_ids = zip(*batch)
    return list(motions), list(captions), list(sample_ids)


# ── logging ───────────────────────────────────────────────────────────────────

def get_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("finetune_corrective")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.addHandler(logging.FileHandler(os.path.join(out_dir, "run.log")))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MotionLLM m2t on corrective feedback")
    parser.add_argument("--guofeats-dir", required=True,
                        help="Directory of per-sample (T, 263) Guo-feature .npy files (training data)")
    parser.add_argument("--labels-jsonl", default="../labels/labels_5var_reusable.json",
                        help="Per-sample feedback labels (JSON dict id -> [variants] or JSONL)")
    parser.add_argument("--pretrained",   default="../checkpoints/motionllm_base/pretrained.pth",
                        help="Pretrained MotionLLM starting point (HumanML3D-pretrained)")
    parser.add_argument("--vqvae-ckpt",   default="../motion_encoder/checkpoints/vqvae_indomain_best.pth",
                        help="In-domain VQ-VAE checkpoint (best results)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip loading pretrained MotionLLM weights (train from base LLM)")
    parser.add_argument("--out-dir",      default="experiments/corrective")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--val-split",    type=float, default=0.1,
                        help="Fraction of data held out for validation (0 = no val)")
    parser.add_argument("--test-split",   type=float, default=0.1,
                        help="Fraction of data held out as a clean test set, never used during training")
    parser.add_argument("--class-labels", type=str, default="../labels/class_labels_pooled_150.json",
                        help="JSON file mapping class name → list of label variants (picked randomly at train time)")
    parser.add_argument("--sample-labels", type=str, default="",
                        help="JSON file mapping sample_id.mp4 → list of 5 label variants (per-sample, higher priority than --class-labels)")
    parser.add_argument("--augmentations", type=str, default="",
                        help="Comma-separated augmentation names to apply, e.g. 'speed' or 'mirror,reverse'. "
                             "Empty = default (mirror+reverse).")
    parser.add_argument("--sustained-classes", type=str, default="",
                        help="Comma-separated class names that additionally receive reverse augmentation "
                             "(error is present throughout, so reversed clip is still a valid training sample). "
                             "E.g. 'rdl_too_much_depth,rdl_head_position,rdl_no_error,squat_no_errors'")
    parser.add_argument("--oversample-keyword", type=str, default="",
                        help="Oversample train samples whose label contains this string")
    parser.add_argument("--oversample-factor", type=int, default=1,
                        help="Repeat matched samples this many times (1=no oversampling)")
    parser.add_argument("--label-sentences", type=int, default=0,
                        help="Truncate labels to first N sentences (0=all). "
                             "Curriculum: 2=depth only, 3=depth+head, 4=all")
    parser.add_argument("--no-augment",   action="store_true",
                        help="Disable training-time data augmentation")
    parser.add_argument("--max-tgt-len",  type=int, default=350,
                        help="Max token sequence length (increase for longer labels)")
    parser.add_argument("--reset-lora",   action="store_true",
                        help="Reinitialise m2t LoRA weights before training instead of "
                             "loading pretrained weights. Use when the VQ-VAE token "
                             "distribution has changed significantly.")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--wandb-project", default="motion-llm")
    parser.add_argument("--wandb-run",     default=None)
    parser.add_argument("--patience",      type=int, default=0,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--lora-r",        type=int, default=32,
                        help="LoRA rank for m2t adapter (default: 32)")
    parser.add_argument("--lora-alpha",    type=int, default=32,
                        help="LoRA alpha for m2t adapter (default: 32)")
    parser.add_argument("--lora-dropout",  type=float, default=0.1,
                        help="LoRA dropout (default: 0.1)")
    parser.add_argument("--scheduler",     default="cosine", choices=["cosine", "onecycle"],
                        help="LR scheduler: cosine (default) or onecycle")
    parser.add_argument("--max-lr",        type=float, default=None,
                        help="Peak LR for onecycle scheduler (default: 3x --lr)")
    parser.add_argument("--no-wandb",      action="store_true")
    our_args = parser.parse_args()

    random.seed(our_args.seed)
    torch.manual_seed(our_args.seed)
    os.makedirs(our_args.out_dir, exist_ok=True)
    logger = get_logger(our_args.out_dir)
    logger.info(json.dumps(vars(our_args), indent=2))

    # configure augmentations
    global _ACTIVE_AUG_FNS, _SUSTAINED_CLASSES
    if our_args.augmentations:
        names = [n.strip() for n in our_args.augmentations.split(",") if n.strip()]
        _ACTIVE_AUG_FNS = _build_aug_fns(names)
        logger.info(f"Augmentations: {[f.__name__ for f in _ACTIVE_AUG_FNS]}")
    else:
        _ACTIVE_AUG_FNS = []
        logger.info("Augmentations: default (mirror + reverse)")

    if our_args.sustained_classes:
        _SUSTAINED_CLASSES = {c.strip() for c in our_args.sustained_classes.split(",") if c.strip()}
        logger.info(f"Sustained classes (extra reverse): {sorted(_SUSTAINED_CLASSES)}")

    use_wandb = WANDB_AVAILABLE and not our_args.no_wandb
    if use_wandb:
        wandb.init(project=our_args.wandb_project, name=our_args.wandb_run,
                   config=vars(our_args))
        logger.info(f"wandb run: {wandb.run.name}")

    # ── build MotionLLM (clears sys.argv so get_args_parser() doesn't choke) ──
    _argv, sys.argv = sys.argv, sys.argv[:1]
    llm_args = get_llm_args()
    sys.argv = _argv
    llm_args.device        = our_args.device
    llm_args.vq_path       = our_args.vqvae_ckpt
    llm_args.lora_r_m2t    = our_args.lora_r
    llm_args.lora_alpha_m2t = our_args.lora_alpha
    llm_args.lora_dropout  = our_args.lora_dropout
    logger.info(f"LoRA m2t: r={our_args.lora_r}  alpha={our_args.lora_alpha}  "
                f"dropout={our_args.lora_dropout}  scaling={our_args.lora_alpha/our_args.lora_r:.2f}")

    logger.info("Building MotionLLM …")
    model = MotionLLM(llm_args)
    if our_args.no_pretrained:
        logger.info("Skipping pretrained weights — training from base LLM.")
    else:
        model.load_model(our_args.pretrained)
        logger.info(f"Loaded pretrained weights from {our_args.pretrained}")

    if our_args.reset_lora:
        # The pretrained m2t LoRA was trained with a different VQ-VAE token
        # distribution. Reinitialise it so the LLM learns from scratch on the
        # new discriminative tokens rather than trying to adapt stale weights.
        reset_count = 0
        for name, module in model.llm.named_modules():
            if "m2t" in name and hasattr(module, "reset_parameters"):
                module.reset_parameters()
                reset_count += 1
        # Also zero the LoRA A/B matrices directly where reset_parameters isn't available
        for name, param in model.llm.named_parameters():
            if "lora" in name and "m2t" in name:
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif "lora_B" in name:
                    torch.nn.init.zeros_(param)
        logger.info("m2t LoRA weights reset to random initialisation")

    # ── freeze everything; then unfreeze only the m2t LoRA ────────────────────
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.llm.named_parameters():
        if "lora" in name and "m2t" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    model.training_task = "m2t"
    model.llm.train()
    model.to(our_args.device)

    # ── label variants ────────────────────────────────────────────────────────
    class_labels = None
    if our_args.class_labels:
        with open(our_args.class_labels) as f:
            class_labels = json.load(f)
        logger.info(f"Loaded class labels from {our_args.class_labels} "
                    f"({len(class_labels)} classes, {list(class_labels.values())[0].__len__()} variants each)")

    sample_labels = None
    if our_args.sample_labels:
        with open(our_args.sample_labels) as f:
            sample_labels = json.load(f)
        n_covered = len(sample_labels)
        n_var = len(list(sample_labels.values())[0])
        logger.info(f"Loaded sample labels from {our_args.sample_labels} "
                    f"({n_covered} samples, {n_var} variants each)")

    # ── dataset & dataloader ──────────────────────────────────────────────────
    # base dataset (no augmentation) — used for val/test subsets
    dataset = SquatCorrectiveDataset(
        our_args.guofeats_dir,
        our_args.labels_jsonl,
        mean=model.mean,
        std=model.std,
        augment=False,
        label_sentences=our_args.label_sentences,
        class_labels=class_labels,
        class_labels_random=False,  # val always uses variant 0 for consistent loss
        sample_labels=sample_labels,
    )
    # augmented version of the same data — used for the train subset
    dataset_aug = SquatCorrectiveDataset(
        our_args.guofeats_dir,
        our_args.labels_jsonl,
        mean=model.mean,
        std=model.std,
        augment=not our_args.no_augment,
        label_sentences=our_args.label_sentences,
        class_labels=class_labels,
        class_labels_random=True,   # train randomly picks from variants
        sample_labels=sample_labels,
    )

    # ── stratified train / val / test split ──────────────────────────────────
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for i, (_, _, cls) in enumerate(dataset.samples):
        class_to_indices[cls].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in class_to_indices.items():
        random.shuffle(idxs)
        # carve test first (never seen during training or model selection)
        n_test = max(1, int(len(idxs) * our_args.test_split)) if our_args.test_split > 0 else 0
        n_val  = max(1, int(len(idxs) * our_args.val_split))  if our_args.val_split  > 0 else 0
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    # save test indices so they can be reproduced later
    test_ids = [dataset.samples[i][0] for i in test_idx]  # npy paths
    val_ids  = [dataset.samples[i][0] for i in val_idx]
    split_path = os.path.join(our_args.out_dir, "splits.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx,
                   "test_paths": test_ids, "val_paths": val_ids}, f, indent=2)
    logger.info(f"Splits saved → {split_path}")

    # ── optional oversampling of label-keyword-matched train samples ─────────
    if our_args.oversample_factor > 1 and our_args.oversample_keyword:
        kw = our_args.oversample_keyword.lower()
        extra = [i for i in train_idx
                 if kw in dataset_aug.samples[i][1].lower()]
        added = extra * (our_args.oversample_factor - 1)
        train_idx = train_idx + added
        random.shuffle(train_idx)
        logger.info(f"Oversampled {len(extra)} samples matching '{our_args.oversample_keyword}' "
                    f"×{our_args.oversample_factor}  (+{len(added)} extra)")

    train_set = torch.utils.data.Subset(dataset_aug, train_idx)
    logger.info(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}  "
                f"(stratified across {len(class_to_indices)} classes)"
                + ("  [augmentation ON]" if not our_args.no_augment else "  [augmentation OFF]"))

    if val_idx:
        val_set    = torch.utils.data.Subset(dataset, val_idx)
        val_loader = DataLoader(val_set, batch_size=our_args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    # Class-balanced sampling: each class is sampled equally regardless of size,
    # preventing the model from collapsing to the most common label pattern.
    train_classes = [dataset_aug.samples[i][2] for i in train_idx]
    class_counts  = {c: train_classes.count(c) for c in set(train_classes)}
    sample_weights = torch.tensor([1.0 / class_counts[c] for c in train_classes])
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_set, batch_size=our_args.batch_size,
                              sampler=sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=our_args.lr,
        weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader)
    if our_args.scheduler == "onecycle":
        max_lr = our_args.max_lr if our_args.max_lr else our_args.lr * 3
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            epochs=our_args.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.3, div_factor=10, final_div_factor=1000,
        )
        logger.info(f"Scheduler: OneCycleLR  max_lr={max_lr:.2e}  "
                    f"start_lr={max_lr/10:.2e}  end_lr={max_lr/10000:.2e}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=our_args.epochs, eta_min=our_args.lr * 0.01
        )
        logger.info(f"Scheduler: CosineAnnealingLR  lr={our_args.lr:.2e}  eta_min={our_args.lr*0.01:.2e}")

    best_loss    = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "lr": []}

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, our_args.epochs + 1):
        model.llm.train()
        epoch_losses, epoch_accs = [], []

        for motions, captions, sample_ids in train_loader:
            # Encode each motion with the frozen VQ-VAE and reindex tokens
            motion_tokens = []
            for m in motions:
                m_tensor = torch.from_numpy(m).float().unsqueeze(0).to(our_args.device)  # (1, T-1, 263)
                with torch.no_grad():
                    tokens = model.net.encode(m_tensor).squeeze(0)                       # (L,)
                    for j in range(tokens.shape[0]):
                        tokens[j] = model.motion_token_indices[tokens[j]]
                motion_tokens.append(tokens)

            instructions = [get_instruction(sid) for sid in sample_ids]
            optimizer.zero_grad()
            loss, acc, _, _ = model.forward(captions, motion_tokens,
                                            instruction=instructions,
                                            max_tgt_len=our_args.max_tgt_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )
            optimizer.step()
            if our_args.scheduler == "onecycle":
                scheduler.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        if our_args.scheduler == "cosine":
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = np.mean(epoch_losses)
        avg_acc  = np.mean(epoch_accs)
        logger.info(f"Epoch {epoch}/{our_args.epochs}  loss={avg_loss:.4f}  acc={avg_acc:.4f}  lr={current_lr:.2e}")
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        history["lr"].append(current_lr)
        if use_wandb:
            wandb.log({"train/loss": avg_loss, "train/acc": avg_acc, "train/lr": current_lr}, step=epoch)

        # ── optional validation ────────────────────────────────────────────────
        if val_loader is not None:
            model.llm.eval()
            val_losses = []
            with torch.no_grad():
                for motions, captions, sample_ids in val_loader:
                    motion_tokens = []
                    for m in motions:
                        m_tensor = torch.from_numpy(m).float().unsqueeze(0).to(our_args.device)
                        tokens = model.net.encode(m_tensor).squeeze(0)
                        for j in range(tokens.shape[0]):
                            tokens[j] = model.motion_token_indices[tokens[j]]
                        motion_tokens.append(tokens)
                    instructions = [get_instruction(sid) for sid in sample_ids]
                    loss, _, _, _ = model.forward(captions, motion_tokens,
                                                  instruction=instructions,
                                                  max_tgt_len=our_args.max_tgt_len)
                    val_losses.append(loss.item())
            avg_val = np.mean(val_losses)
            history["val_loss"].append(avg_val)
            logger.info(f"           val_loss={avg_val:.4f}")
            if use_wandb:
                wandb.log({"val/loss": avg_val}, step=epoch)

            if avg_val < best_loss:
                best_loss = avg_val
                epochs_no_improve = 0
                out_path  = os.path.join(our_args.out_dir, "motionllm_corrective_best.pth")
                model.save_model(out_path)
                logger.info(f"  ✓ new best val_loss={best_loss:.4f} → {out_path}")
            else:
                epochs_no_improve += 1
                if our_args.patience > 0 and epochs_no_improve >= our_args.patience:
                    logger.info(f"  Early stopping: no improvement for {epochs_no_improve} epochs.")
                    break
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                out_path  = os.path.join(our_args.out_dir, "motionllm_corrective_best.pth")
                model.save_model(out_path)
                logger.info(f"  ✓ new best loss={best_loss:.4f} → {out_path}")

        # always save latest
        model.save_model(os.path.join(our_args.out_dir, "motionllm_corrective_latest.pth"))

    logger.info("Fine-tuning complete.")
    if use_wandb:
        wandb.finish()
    logger.info(f"Best checkpoint: {os.path.join(our_args.out_dir, 'motionllm_corrective_best.pth')}")

    # ── save training curves ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_loss"], label="train loss")
        if history["val_loss"]:
            ax1.plot(epochs, history["val_loss"], label="val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history["train_acc"], label="train acc", color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Train Accuracy")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle(os.path.basename(our_args.out_dir))
        fig.tight_layout()
        curve_path = os.path.join(our_args.out_dir, "training_curves.png")
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        logger.info(f"Training curves saved → {curve_path}")
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")


if __name__ == "__main__":
    main()
