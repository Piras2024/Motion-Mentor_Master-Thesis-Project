"""
Guo-feature augmentations for HumanML3D 263-dim motion sequences.

All functions operate on raw (un-normalized) numpy float32 arrays of shape (T, 263).
Variable-length augmentations (speed, crop) return arrays of different T — callers
that need fixed-length sequences must handle padding/truncation themselves.

Feature layout:
  [0:4]     root: angular_vel, x_vel, z_vel, height
  [4:67]    RIC positions  — joints 1-21 × (x, y, z)
  [67:193]  6D rotations   — joints 1-21 × 6
  [193:259] local velocity — joints 0-21 × (x, y, z)
  [259:261] left foot contacts
  [261:263] right foot contacts
"""

import numpy as np
import torch
import torch.nn.functional as F

# Left/right joint pairs — 0-based indices within the 21-joint RIC/rot arrays
# (joint numbering 1-21 in the kinematic chain; index = joint_number - 1)
_LR_PAIRS_RIC = [(0,1),(3,4),(6,7),(9,10),(12,13),(15,16),(17,18),(19,20)]
# Same pairs within the 22-joint velocity array (joint 0 = root has no partner)
_LR_PAIRS_VEL = [(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(18,19),(20,21)]


def noise(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Add Gaussian noise to all continuous features (not foot contacts)."""
    out = x.copy()
    out[:, :259] += (np.random.randn(*out[:, :259].shape) * sigma).astype(np.float32)
    return out


def speed(x: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """
    Resample to a random speed factor via linear interpolation.
    Returns a variable-length array — no padding or truncation applied.
      factor < 1  →  fewer frames  (faster motion)
      factor > 1  →  more frames   (slower motion)
    """
    T, D = x.shape
    factor = lo + np.random.rand() * (hi - lo)
    new_T = max(1, round(T * factor))
    return F.interpolate(
        torch.from_numpy(x.T[None].astype(np.float32)),
        size=new_T, mode='linear', align_corners=False,
    ).squeeze(0).T.numpy()


def reverse(x: np.ndarray) -> np.ndarray:
    """Reverse the temporal order of the sequence."""
    return x[::-1].copy()


def scale(x: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """Scale all continuous features (positions, velocities, root) by a random factor."""
    out = x.copy()
    f = lo + np.random.rand() * (hi - lo)
    out[:, :259] *= f
    return out


def crop(x: np.ndarray, min_frac: float = 0.8) -> np.ndarray:
    """
    Return a random contiguous sub-sequence of at least min_frac of the original.
    Returns a variable-length array.
    """
    T = x.shape[0]
    min_len = max(1, int(T * min_frac))
    crop_T = min_len + np.random.randint(0, T - min_len + 1)
    start = np.random.randint(0, T - crop_T + 1)
    return x[start:start + crop_T].copy()


def mirror(x: np.ndarray) -> np.ndarray:
    """
    Left-right mirror: swap symmetric joint pairs and negate X-axis components.

    Rotation handling uses the body-frame convention (M·R·M, M=diag(-1,1,1)):
      dims 1,2 of each joint's 6D (r0y, r0z) and dim 3 (r1x) are negated after swapping.
    """
    out = x.copy()

    # Root: negate angular velocity and X linear velocity
    out[:, 0] = -x[:, 0]
    out[:, 1] = -x[:, 1]

    # RIC positions (4:67) — 21 joints × (x, y, z)
    ric_orig = x[:, 4:67].reshape(-1, 21, 3)
    ric = ric_orig.copy()
    for l, r in _LR_PAIRS_RIC:
        ric[:, l] = ric_orig[:, r]
        ric[:, r] = ric_orig[:, l]
    ric[:, :, 0] *= -1
    out[:, 4:67] = ric.reshape(-1, 63)

    # 6D rotations (67:193) — 21 joints × 6, body-frame mirror = M·R·M
    rot_orig = x[:, 67:193].reshape(-1, 21, 6)
    rot = rot_orig.copy()
    for l, r in _LR_PAIRS_RIC:
        rot[:, l] = rot_orig[:, r]
        rot[:, r] = rot_orig[:, l]
    rot[:, :, 1] *= -1   # r0y
    rot[:, :, 2] *= -1   # r0z
    rot[:, :, 3] *= -1   # r1x
    out[:, 67:193] = rot.reshape(-1, 126)

    # Local velocities (193:259) — 22 joints × (x, y, z)
    vel_orig = x[:, 193:259].reshape(-1, 22, 3)
    vel = vel_orig.copy()
    for l, r in _LR_PAIRS_VEL:
        vel[:, l] = vel_orig[:, r]
        vel[:, r] = vel_orig[:, l]
    vel[:, :, 0] *= -1
    out[:, 193:259] = vel.reshape(-1, 66)

    # Foot contacts: swap left ↔ right
    out[:, 259:261] = x[:, 261:263]
    out[:, 261:263] = x[:, 259:261]

    return out


def combo(x: np.ndarray) -> np.ndarray:
    """noise(σ=0.01) + speed + scale combined."""
    return scale(speed(noise(x, sigma=0.01)))


# Convenience dict for external iteration
ALL = {
    "none":        lambda x: x.copy(),
    "noise_small": lambda x: noise(x, sigma=0.01),
    "noise_large": lambda x: noise(x, sigma=0.05),
    "reverse":     reverse,
    "speed":       speed,
    "crop":        crop,
    "scale":       scale,
    "mirror":      mirror,
    "combo":       combo,
}
