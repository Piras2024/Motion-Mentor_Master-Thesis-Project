import torch
import numpy as np
from lib.modeling.pipelines.hsmr import build_inference_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Load the HSMR inference pipeline with pretrained checkpoint
model_root = 'data_inputs/released_models/HSMR-ViTH-r1d1'  # folder containing .hydra/config.yaml and hsmr.ckpt
pipeline = build_inference_pipeline(model_root=model_root, device=device)
pipeline.eval()

# 2️⃣ Load your demo data
demo_data = np.load('/andromeda/personal/mpiras/HSMR/data_outputs/demos/squat_micc/HSMR-squat_butt_wink1.npy', allow_pickle=True)
print(demo_data.shape)
print(demo_data[0].keys())  # should contain 'patch_cam_t', 'poses', 'betas', etc.

# 3️⃣ Convert your numpy data to torch
poses = torch.from_numpy(np.stack([d['poses'] for d in demo_data])).float().to(device)
betas = torch.from_numpy(np.stack([d['betas'] for d in demo_data])).float().to(device)
patch_cam_t = torch.from_numpy(np.stack([d['patch_cam_t'] for d in demo_data])).float().to(device)
bbx_cs = [d['bbx_cs'] for d in demo_data]

B = poses.shape[0]

# 4️⃣ Forward pass through the pipeline (3D + 2D keypoints)
with torch.no_grad():
    outputs = pipeline.forward({'img_patch': torch.zeros(B, 3, 256, 256, device=device)})  # dummy images
    # If you already have poses+betas, you can directly call the skel_model:
    skel_outputs = pipeline.skel_model(poses=poses, betas=betas, skelmesh=False)
    kp3d = skel_outputs.joints.detach().cpu().numpy()  # [B, J, 3]

# 5️⃣ Compute 2D keypoints (weak-perspective)
kp2d = kp3d + patch_cam_t[:, None, :3].cpu().numpy()  # add camera translation

# 6️⃣ Optionally map to image coordinates
def transform_to_image(kp2d_frame, bbx):
    """
    Adapt HSMR weak-perspective to original image coordinates
    bbx: bounding box (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbx
    scale = (x2 - x1)
    return kp2d_frame[:, :2] * scale + np.array([x1, y1])

kp2d_img = np.stack([transform_to_image(kp2d[i], bbx_cs[i]) for i in range(B)], axis=0)

print("3D keypoints shape:", kp3d.shape)
print("2D keypoints shape:", kp2d_img.shape)