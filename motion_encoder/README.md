# Motion Encoder — VQ-VAE

A VQ-VAE that encodes per-frame Guo features (263-d HumanML3D-style
representation) into a discrete sequence of motion tokens. The architecture
follows T2M-GPT (Zhang et al.); the codebook was domain-adapted from the
HumanML3D-trained checkpoint to the strength-training distribution used in
this thesis.

## Contents

```
motion_encoder/
├── models/                  VQ-VAE model code (vqvae, encdec, resnet, quantize_cnn)
├── options/                 argparse helpers used by the loader
├── meta/                    mean.npy + std.npy normalisation stats
├── checkpoints/             best.pth — the in-domain VQ-VAE used everywhere
└── train_vqvae.py           training script (only needed to retrain)
```

## Using the encoder

The VLM training and evaluation scripts already wrap this. If you want to
encode a clip standalone:

```python
import sys, numpy as np, torch
sys.path.insert(0, "motion_encoder")
from models import vqvae as vqvae_module
from options.option_llm import get_args_parser

# Build the network using the same hyperparameters as training:
import argparse as ap
a = ap.Namespace(dataname="hml3d", nb_code=512, code_dim=512,
                 output_emb_width=512, down_t=2, stride_t=2, width=512,
                 depth=3, dilation_growth_rate=3, vq_act="relu",
                 vq_norm=None, quantizer="ema_reset", mu=0.99)
net = vqvae_module.HumanVQVAE(a, 512, 512, 512, 2, 2, 512, 3, 3, "relu", None)

ckpt = torch.load("motion_encoder/checkpoints/vqvae_indomain_best.pth", map_location="cpu")
net.load_state_dict(ckpt["net"], strict=True)
net.eval().cuda()

mean = np.load("motion_encoder/meta/mean.npy")
std  = np.load("motion_encoder/meta/std.npy")

# guofeats: (T, 263) numpy array produced by pipeline/hsmr_to_guofeats.py
feats = (np.load("clip_guofeats.npy") - mean) / std
indices = net.encode(torch.from_numpy(feats).float().unsqueeze(0).cuda()).squeeze(0).cpu().tolist()
print(indices)  # length T'/4 sequence of codebook indices in [0, 512)
```

## Retraining (optional)

`train_vqvae.py` is the training script (originally `finetune_vqvae.py` in
motion-agent). It expects a directory of `_guofeats.npy` files and a
configuration via `options.option_vq`.
