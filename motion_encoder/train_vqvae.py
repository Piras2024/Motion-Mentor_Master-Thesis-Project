"""
finetune_vqvae.py

Fine-tune the pretrained HumanVQVAE on domain-specific motion data,
combining data augmentation with a discriminative classification loss.

Augmentations (applied stochastically per sample per epoch):
  reverse  — flip temporal order
  scale    — randomly scale position / velocity magnitudes
  speed    — temporal resampling (faster / slower)
  crop     — random contiguous sub-sequence
  combo    — random combination of the above

Training objective:
  loss = L2_recon + λ_vel * L2_velocity + λ_commit * commit_loss
       + λ_cls * CrossEntropy(mean_pool(encoder_output), class_label)

Usage:
  python finetune_vqvae.py \\
    --guofeats-dir /deck/users/mpiras/dataset/hsmr_guofeats \\
    --pretrained   ckpt/vqvae.pth \\
    --out-dir      experiments/vqvae_aug_cls \\
    --epochs       100 --batch-size 32 --lr 1e-4 --lambda-cls 1.0
"""

import sys
import os
import json
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import models.vqvae as vqvae_module
from options.option_llm import get_args_parser as get_llm_args
import utils.augmentations as _augs

_DEFAULT_AUG_FNS = [_augs.mirror, _augs.reverse]
_AUG_PROBS = {"mirror": 0.5, "reverse": 0.2, "speed": 0.5,
              "scale": 0.5, "noise": 0.5, "crop": 0.3}
_ACTIVE_AUG_FNS: list = []


def _build_aug_fns(aug_names: list) -> list:
    return [getattr(_augs, name) for name in aug_names if hasattr(_augs, name)]


def augment_motion(motion: np.ndarray) -> np.ndarray:
    fns = _ACTIVE_AUG_FNS if _ACTIVE_AUG_FNS else _DEFAULT_AUG_FNS
    for fn in fns:
        prob = _AUG_PROBS.get(fn.__name__, 0.5)
        if np.random.rand() < prob:
            motion = fn(motion)
    return motion

MEAN_PATH = "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH  = "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

# Class names as they appear in the dataset filenames
CLASSES = [
    "rdl_hands_forward", "rdl_no_error", "rdl_too_much_depth",
    "rdl_head_position", "rdl_too_much_knee_bend",
    "squat_butt_wink", "squat_depth_high", "squat_hands_wide",
    "squat_head_position", "squat_high_heel", "squat_no_errors",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def class_from_path(path: str) -> int:
    fname = os.path.basename(path)
    name  = fname.replace("HSMR-", "").replace("_guofeats.npy", "")
    cls   = name.rstrip("0123456789")
    return CLASS_TO_IDX.get(cls, -1)


# ── classification head ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, emb_dim, T_enc) → mean pool → (B, emb_dim) → (B, num_classes)
        return self.fc(x.mean(dim=-1))


# ── dataset ───────────────────────────────────────────────────────────────────

class GuoFeatsDataset(Dataset):
    def __init__(self, guofeats_dir: str, mean: np.ndarray, std: np.ndarray,
                 max_len: int = 196, augment: bool = False):
        self.mean    = mean
        self.std     = std
        self.max_len = max_len
        self.augment = augment

        all_paths = sorted([
            os.path.join(guofeats_dir, f)
            for f in os.listdir(guofeats_dir)
            if f.endswith("_guofeats.npy")
        ])
        # drop samples with unrecognised class names
        self.paths  = [p for p in all_paths if class_from_path(p) >= 0]
        skipped = len(all_paths) - len(self.paths)
        if skipped:
            print(f"  [dataset] skipped {skipped} samples with unknown class")
        print(f"VQ-VAE dataset: {len(self.paths)} samples  augment={augment}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path   = self.paths[idx]
        label  = class_from_path(path)
        motion = np.load(path).astype(np.float32)       # (T, 263)
        if self.augment:
            motion = augment_motion(motion)
        motion = (motion - self.mean) / self.std
        T = motion.shape[0]
        if T >= self.max_len:
            motion = motion[:self.max_len]
        else:
            pad = np.zeros((self.max_len - T, motion.shape[1]), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        return torch.from_numpy(motion), label


# ── helpers ───────────────────────────────────────────────────────────────────

def get_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("finetune_vqvae")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in [logging.FileHandler(os.path.join(out_dir, "run.log")),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, 1:] - pred[:, :-1],
                      target[:, 1:] - target[:, :-1])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guofeats-dir", required=True)
    parser.add_argument("--pretrained",   default="ckpt/vqvae.pth")
    parser.add_argument("--out-dir",      default="experiments/vqvae_aug_cls")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--loss-vel",     type=float, default=0.1)
    parser.add_argument("--commit",       type=float, default=0.02)
    parser.add_argument("--lambda-cls",   type=float, default=1.0,
                        help="Weight for the classification loss")
    parser.add_argument("--val-split",    type=float, default=0.1)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--wandb-project", default="motion-vqvae")
    parser.add_argument("--wandb-run",     default=None)
    parser.add_argument("--no-augment",    action="store_true",
                        help="Disable augmentation on the training set")
    parser.add_argument("--augmentations", type=str, default="",
                        help="Comma-separated augmentation names, e.g. 'speed' or 'mirror,reverse'. "
                             "Empty = default (mirror+reverse).")
    parser.add_argument("--no-wandb",      action="store_true")
    our_args = parser.parse_args()

    random.seed(our_args.seed)
    torch.manual_seed(our_args.seed)
    logger = get_logger(our_args.out_dir)
    logger.info(json.dumps(vars(our_args), indent=2))

    global _ACTIVE_AUG_FNS
    if our_args.augmentations:
        names = [n.strip() for n in our_args.augmentations.split(",") if n.strip()]
        _ACTIVE_AUG_FNS = _build_aug_fns(names)
        logger.info(f"Augmentations: {[f.__name__ for f in _ACTIVE_AUG_FNS]}")
    else:
        _ACTIVE_AUG_FNS = []
        logger.info("Augmentations: default (mirror + reverse)")

    use_wandb = WANDB_AVAILABLE and not our_args.no_wandb
    if use_wandb:
        wandb.init(project=our_args.wandb_project, name=our_args.wandb_run,
                   config=vars(our_args))
        logger.info(f"wandb run: {wandb.run.name}")

    device = our_args.device

    # ── build VQ-VAE ──────────────────────────────────────────────────────────
    _argv, sys.argv = sys.argv, sys.argv[:1]
    llm_args = get_llm_args()
    sys.argv = _argv
    llm_args.device   = device
    llm_args.dataname = 't2m'

    net = vqvae_module.HumanVQVAE(
        llm_args,
        llm_args.nb_code, llm_args.code_dim, llm_args.output_emb_width,
        llm_args.down_t,  llm_args.stride_t, llm_args.width,
        llm_args.depth,   llm_args.dilation_growth_rate,
        llm_args.vq_act,  llm_args.vq_norm,
    )
    ckpt = torch.load(our_args.pretrained, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.to(device)
    logger.info(f"Loaded VQ-VAE from {our_args.pretrained}")

    # ── classification head ───────────────────────────────────────────────────
    # Probe encoder output dim with a dummy input
    with torch.no_grad():
        dummy    = torch.zeros(1, 4, 263, device=device)
        x_in_d   = net.vqvae.preprocess(dummy)
        enc_out  = net.vqvae.encoder(x_in_d)
        emb_dim  = enc_out.shape[1]
    cls_head = ClassificationHead(emb_dim, len(CLASSES)).to(device)
    logger.info(f"Encoder emb_dim={emb_dim}  num_classes={len(CLASSES)}")
    logger.info(f"Classes: {CLASSES}")

    # ── dataset & split ───────────────────────────────────────────────────────
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)

    all_paths = sorted([
        os.path.join(our_args.guofeats_dir, f)
        for f in os.listdir(our_args.guofeats_dir)
        if f.endswith("_guofeats.npy") and class_from_path(
            os.path.join(our_args.guofeats_dir, f)) >= 0
    ])
    indices = list(range(len(all_paths)))
    random.shuffle(indices)
    n_val     = max(1, int(len(indices) * our_args.val_split))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_dataset = GuoFeatsDataset(our_args.guofeats_dir, mean, std, augment=not our_args.no_augment)
    val_dataset   = GuoFeatsDataset(our_args.guofeats_dir, mean, std, augment=False)

    train_set = torch.utils.data.Subset(train_dataset, train_idx)
    val_set   = torch.utils.data.Subset(val_dataset,   val_idx)

    train_loader = DataLoader(train_set, batch_size=our_args.batch_size,
                              shuffle=True, drop_last=True, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=our_args.batch_size,
                              shuffle=False, num_workers=2)
    logger.info(f"Train: {len(train_set)}  Val: {len(val_set)}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(net.vqvae.encoder.parameters()) +
        list(net.vqvae.decoder.parameters()) +
        list(cls_head.parameters()),
        lr=our_args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=our_args.epochs, eta_min=our_args.lr * 0.1
    )

    best_val = float("inf")
    history = {"train_recon": [], "val_recon": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, our_args.epochs + 1):
        net.train()
        cls_head.train()
        t_recon, t_cls, t_correct, t_total = [], [], 0, 0

        for batch, labels in train_loader:
            batch  = batch.to(device).float()
            labels = labels.to(device)

            # Single encoder forward used for both reconstruction and classification
            x_in      = net.vqvae.preprocess(batch)
            x_enc     = net.vqvae.encoder(x_in)                        # (B, emb_dim, T_enc)
            x_quant, commit_loss, perplexity = net.vqvae.quantizer(x_enc)
            x_dec     = net.vqvae.decoder(x_quant)
            x_out     = net.vqvae.postprocess(x_dec)

            recon  = F.mse_loss(x_out.float(), batch)
            vel    = velocity_loss(x_out.float(), batch)
            logits = cls_head(x_enc)
            cls    = F.cross_entropy(logits, labels)

            loss = (recon
                    + our_args.loss_vel * vel
                    + our_args.commit   * commit_loss.float()
                    + our_args.lambda_cls * cls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(cls_head.parameters()), max_norm=1.0)
            optimizer.step()

            t_recon.append(recon.item())
            t_cls.append(cls.item())
            preds = logits.argmax(dim=1)
            t_correct += (preds == labels).sum().item()
            t_total   += labels.size(0)

        scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        net.eval()
        cls_head.eval()
        v_recon, v_cls, v_correct, v_total = [], [], 0, 0

        with torch.no_grad():
            for batch, labels in val_loader:
                batch  = batch.to(device).float()
                labels = labels.to(device)

                x_in     = net.vqvae.preprocess(batch)
                x_enc    = net.vqvae.encoder(x_in)
                x_quant, commit_loss, _ = net.vqvae.quantizer(x_enc)
                x_dec    = net.vqvae.decoder(x_quant)
                x_out    = net.vqvae.postprocess(x_dec)

                recon  = F.mse_loss(x_out.float(), batch)
                vel    = velocity_loss(x_out.float(), batch)
                logits = cls_head(x_enc)
                cls    = F.cross_entropy(logits, labels)

                v_recon.append(recon.item())
                v_cls.append(cls.item())
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        avg_t_recon = np.mean(t_recon)
        avg_v_recon = np.mean(v_recon)
        avg_t_cls   = np.mean(t_cls)
        avg_v_cls   = np.mean(v_cls)
        t_acc = t_correct / t_total * 100
        v_acc = v_correct / v_total * 100
        val_loss = avg_v_recon + our_args.lambda_cls * avg_v_cls

        history["train_recon"].append(avg_t_recon)
        history["val_recon"].append(avg_v_recon)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        logger.info(
            f"Epoch {epoch:3d}/{our_args.epochs}  "
            f"recon={avg_t_recon:.4f}/{avg_v_recon:.4f}  "
            f"cls={avg_t_cls:.4f}/{avg_v_cls:.4f}  "
            f"acc={t_acc:.1f}%/{v_acc:.1f}%  "
            f"perp={perplexity:.0f}  lr={scheduler.get_last_lr()[0]:.1e}"
        )

        if use_wandb:
            wandb.log({
                "train/recon":   avg_t_recon,
                "val/recon":     avg_v_recon,
                "train/cls":     avg_t_cls,
                "val/cls":       avg_v_cls,
                "train/acc":     t_acc,
                "val/acc":       v_acc,
                "perplexity":    perplexity,
                "lr":            scheduler.get_last_lr()[0],
                "val/loss":      val_loss,
            }, step=epoch)

        torch.save({"net": net.state_dict(), "cls_head": cls_head.state_dict()},
                   os.path.join(our_args.out_dir, "vqvae_aug_cls_latest.pth"))
        if val_loss < best_val:
            best_val = val_loss
            out_path = os.path.join(our_args.out_dir, "vqvae_aug_cls_best.pth")
            torch.save({"net": net.state_dict(), "cls_head": cls_head.state_dict()},
                       out_path)
            logger.info(f"  ✓ best val_loss={best_val:.4f}  val_acc={v_acc:.1f}%  → {out_path}")

    logger.info("Done.")
    if use_wandb:
        wandb.finish()

    # ── save training curves ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history["train_recon"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_recon"], label="train recon")
        ax1.plot(epochs, history["val_recon"],   label="val recon")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reconstruction Loss")
        ax1.set_title("Reconstruction Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history["train_acc"], label="train acc", color="green")
        ax2.plot(epochs, history["val_acc"],   label="val acc",   color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Classification Accuracy")
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
