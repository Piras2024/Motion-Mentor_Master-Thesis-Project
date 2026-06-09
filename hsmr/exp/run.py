# Import jaraco.collections before setuptools pollutes the jaraco namespace
# (setuptools vendors its own jaraco without 'collections', overwriting sys.modules['jaraco'])
import jaraco.collections  # noqa: F401
from lib.kits.hsmr_demo import *
from pathlib import Path
import gc

def process_single_video(args, video_path, output_dir):
    """Process a single video using the existing HSMR demo pipeline."""
    args.input_path = str(video_path)
    args.output_path = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.
    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args)

            # ✂️ Keep only the first 128 frames (if the video has more)
            #if inputs_meta["type"] == "video" and len(raw_imgs) > 128:
            #    raw_imgs = raw_imgs[:128]
            #    get_logger(brief=True).info(f"📉 Trimmed video to first 128 frames.")

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                batch_size=args.det_bs,
                max_img_size=args.det_mis,
                device=args.device,
            )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(
                raw_imgs, *detector_outputs, args.max_instances
            )
        if len(patches) == 0:
            get_logger(brief=True).error(
                f'🚫 No human instance detected in {video_path}.'
            )
            return
        get_logger(brief=True).info(
            f'🔍 {len(patches)} human instances detected.'
        )

    # Free detector from GPU before loading the heavy HSMR model
    del detector
    torch.cuda.empty_cache()
    gc.collect()

    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = patches[bw.sid:bw.eid]
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
                print("keys in output pipeline",list(outputs.keys())) # ['pd_cam', 'pd_cam_t', 'pd_params', 'focal_length', 'pd_kp3d', 'pd_kp2d', 'pred_keypoints_2d', 'pred_keypoints_3d', 'pd_skin_verts']
            pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
            pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

        pd_params = assemble_dict(pd_params, expand_dim=False)
        pd_cam_t = torch.cat(pd_cam_t, dim=0)
        dump_data = {
            'patch_cam_t': pd_cam_t.numpy(),
            **{k: v.numpy() for k, v in pd_params.items()},
        }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(pipeline, pd_params)
        get_logger(brief=True).info(f'🏁 Done.')

    # ⛩️ 3. Postprocess.
    with monitor('Visualization'):
        if args.ignore_skel:
            m_skel = None
        results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption)
        dump_data['full_cam_t'] = full_cam_t

        seq_name = f'{pipeline.name}-{video_path.stem}'
        save_video(results, output_dir / f'{seq_name}.mp4',fps=20)

        dump_results = []
        cur_patch_j = 0
        for i in range(len(raw_imgs)):
            n_patch_cur_img = det_meta['n_patch_per_img'][i]
            dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
            dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
            cur_patch_j += n_patch_cur_img
            dump_results.append(dump_results_i)
        np.save(output_dir / f'{seq_name}.npy', dump_results)

        get_logger(brief=True).info(f'🎨 Rendering results saved under {output_dir}.')

    get_logger(brief=True).info(f'🎊 Finished processing {video_path.name}!')
    monitor.report()

    del pipeline, raw_imgs, patches, pd_params, pd_cam_t, m_skin, m_skel
    torch.cuda.empty_cache()
    gc.collect()



def main():
    args = parse_args()
    input_path = Path(args.input_path)
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    # 🧭 If input_path is a folder, process all videos in it
    if input_path.is_dir():
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mov"))
        if not video_files:
            print(f"⚠️ No video files found in {input_path}")
            return

        for video_path in video_files:
            # Skip if output .npy already exists (any pipeline prefix)
            existing = list(outputs_root.glob(f"*-{video_path.stem}.npy"))
            if existing:
                print(f"\n⏭️  Skipping {video_path.name} (already processed: {existing[0].name})")
                continue
            print(f"\n🔁 Processing video: {video_path.name}")
            # 🔧 Save all results in the same folder (no per-video subfolders)
            process_single_video(args, video_path, outputs_root)

    else:
        process_single_video(args, input_path, outputs_root)


if __name__ == "__main__":
    main()
