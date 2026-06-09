from lib.kits.hsmr_demo import *
from pathlib import Path
from moviepy.editor import VideoFileClip
import torch
import numpy as np
import gc


def trim_and_resample_video(input_path, max_frames=128, target_fps=20):
    """
    Trim the video to its first `max_frames` frames and resample it to `target_fps`.
    Returns the path to the trimmed video.
    """
    output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
    with VideoFileClip(str(input_path)) as clip:
        duration = max_frames / target_fps
        trimmed = clip.subclip(0, min(duration, clip.duration))
        trimmed = trimmed.set_fps(target_fps)
        trimmed.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None
        )
    return output_path


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

        seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
        save_video(results, output_dir / f'{seq_name}.mp4')

        dump_results = []
        cur_patch_j = 0
        for i in range(len(raw_imgs)):
            n_patch_cur_img = det_meta['n_patch_per_img'][i]
            dump_results_i = {k: v[cur_patch_j:cur_patch_j + n_patch_cur_img] for k, v in dump_data.items()}
            dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
            cur_patch_j += n_patch_cur_img
            dump_results.append(dump_results_i)
        np.save(output_dir / f'{seq_name}.npy', dump_results)

        get_logger(brief=True).info(f'🎨 Rendering results saved under {output_dir}.')

    get_logger(brief=True).info(f'🎊 Finished processing {video_path.name}!')
    monitor.report()

    # 🧹 Free GPU memory
    del detector, pipeline, raw_imgs, patches, pd_params, pd_cam_t, m_skin, m_skel
    torch.cuda.empty_cache()
    gc.collect()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    # 🧭 If input_path is a folder, process all videos in it
    if input_path.is_dir():
        # Support both .mp4 and .mov files
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mov"))
        if not video_files:
            print(f"⚠️ No video files found in {input_path}")
            return

        for video_path in video_files:
            print(f"\n🔁 Processing video: {video_path.name}")

            # ✂️ Trim and resample before processing
            trimmed_path = trim_and_resample_video(video_path, max_frames=128, target_fps=20)

            video_output_dir = outputs_root / video_path.stem
            process_single_video(args, trimmed_path, video_output_dir)

            # 🧹 Clean up to free memory between runs
            torch.cuda.empty_cache()
            gc.collect()

            # Optional: delete the trimmed file to save space
            try:
                trimmed_path.unlink()
            except Exception:
                pass

    else:
        # If it’s a single video file
        trimmed_path = trim_and_resample_video(input_path, max_frames=128, target_fps=20)
        process_single_video(args, trimmed_path, outputs_root)

        torch.cuda.empty_cache()
        gc.collect()

        try:
            trimmed_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
