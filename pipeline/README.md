# Pipeline — raw mp4 → Guo features

This folder is a placeholder for the glue scripts that take a raw video
through HSMR and into the format the VLM training expects.

The thesis used scripts internal to the HSMR codebase (`hsmr/exp/*.py`) and
to motion-agent. They are not yet packaged for standalone use here; for the
demo, the three test videos under `../demo/test_videos/` come with their
Guo features pre-extracted.

If you want to extract features for your own videos, the rough recipe is:

1. **HSMR pose estimation** — see `../hsmr/` and the upstream HSMR repo.
   Output: per-frame SKEL/SMPL pose parameters as `.npy`.

2. **Guo-feature conversion** — convert the pose stream into the 263-d
   HumanML3D-style representation (the input format the VQ-VAE expects).
   The reference implementation lives in `motion-agent` and uses the
   joints/velocities pipeline described in T2M-GPT.

3. **VQ-VAE encoding** — see `../motion_encoder/README.md`.

A clean self-contained version of steps 1 and 2 is a natural next addition
to this repo. Tracked as a todo.
