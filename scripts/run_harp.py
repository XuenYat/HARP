"""
run_harp.py - Clean single-script HARP pipeline
================================================
Usage:
    python scripts/run_harp.py --video "./example.mov"                  # HARP
    python scripts/run_harp.py --video "./example.mov" --method tram    # TRAM
    
Results saved as camera_{method}.npy to avoid conflicts with existing results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track
from lib.camera import calibrate_intrinsics, align_cam_to_world
from lib.camera.masked_droid_slam import run_slam
from lib.camera.slam_utils import get_dimention
from lib.camera.est_scale import est_scale_hybrid
from lib.utils.rotation_conversions import quaternion_to_matrix


# ============================================================
# HARP Core: Boundary-based scale estimation
# ============================================================

def est_scale_boundary(slam_depth, tz, tx, ty, img_focal, img_center):
    """
    α_bound = VIMO_tz / SLAM_boundary
    
    Sample SLAM depth at human boundary (feet, sides) where valid depth exists.
    """
    if tz <= 0.1 or tz > 50:
        return None
    
    H, W = slam_depth.shape
    orig_H, orig_W = img_center[1] * 2, img_center[0] * 2
    
    # Project human center to depth coordinates
    u = img_focal * tx / tz + img_center[0]
    v = img_focal * ty / tz + img_center[1]
    u_d = int(u * W / orig_W)
    v_d = int(v * H / orig_H)
    
    # Estimate human size in depth map
    h_px = int(img_focal * 1.7 / tz * H / orig_H)
    w_px = h_px // 3
    feet_v = v_d + h_px // 3
    
    samples = []
    
    # Below feet
    for dv in range(5, 30, 5):
        v_s = min(H - 1, feet_v + dv)
        for du in range(-20, 21, 5):
            u_s = max(0, min(W - 1, u_d + du))
            d = slam_depth[v_s, u_s]
            if d > 0 and np.isfinite(d):
                samples.append(d)
    
    # Sides
    for side in [-1, 1]:
        u_side = u_d + side * (w_px // 2 + 10)
        if 0 <= u_side < W:
            for dv in range(-20, 21, 5):
                v_s = max(0, min(H - 1, v_d + dv))
                d = slam_depth[v_s, u_side]
                if d > 0 and np.isfinite(d):
                    samples.append(d)
    
    if len(samples) < 10:
        return None
    
    boundary_depth = np.median(samples)
    return tz / boundary_depth if boundary_depth > 0 else None


# ============================================================
# Pipeline stages
# ============================================================

def run_detection(img_folder, seq_folder, visualize_mask=False):
    """Stage 1: Detection + Segmentation + Tracking"""
    print('\n[1/4] Detection + Segmentation + Tracking...')
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    boxes, masks, tracks = detect_segment_track(
        imgfiles, seq_folder, thresh=0.25, min_size=100, save_vos=visualize_mask
    )
    return boxes, masks, tracks


def run_slam_only(img_folder, masks, calib, is_static=False):
    """Stage 2: Masked DROID-SLAM (without scale estimation)"""
    print('\n[2/4] Masked DROID-SLAM...')
    
    if is_static:
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        n = len(imgfiles)
        return torch.eye(3).expand(n, 3, 3), torch.zeros(n, 3), [], [], []
    
    droid, traj = run_slam(img_folder, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamps = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    
    # Get SLAM depths (relative scale)
    slam_depths = [1.0 / disps[i] for i in range(n)]
    
    # Camera trajectory (relative scale)
    traj_t = torch.tensor(traj[:, :3])
    traj_q = torch.tensor(traj[:, 3:])
    traj_R = quaternion_to_matrix(traj_q[:, [3, 0, 1, 2]])
    
    del droid
    torch.cuda.empty_cache()
    
    return traj_R, traj_t, slam_depths, tstamps, traj


def run_vimo(img_folder, tracks_dict, img_focal, img_center, hps_folder, max_humans=20):
    """Stage 3: VIMO human mesh recovery"""
    print('\n[3/4] VIMO Human Mesh Recovery...')
    
    from lib.models import get_hmr_vimo
    
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    os.makedirs(hps_folder, exist_ok=True)
    
    # Load model
    model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar')
    
    # Sort tracks by length
    tid = [k for k in tracks_dict.keys()]
    lens = [len(trk) for trk in tracks_dict.values()]
    rank = np.argsort(lens)[::-1]
    tracks_sorted = [tracks_dict[tid[r]] for r in rank]
    
    all_results = []
    
    for k, trk in enumerate(tracks_sorted):
        valid = np.array([t['det'] for t in trk])
        boxes = np.concatenate([t['det_box'] for t in trk])
        frame = np.array([t['frame'] for t in trk])
        
        results = model.inference(imgfiles, boxes, valid=valid, frame=frame,
                                  img_focal=img_focal, img_center=img_center)
        
        if results is not None:
            np.save(f'{hps_folder}/hps_track_{k}.npy', results)
            all_results.append({
                'pred_trans': results['pred_trans'].numpy() if hasattr(results['pred_trans'], 'numpy') else results['pred_trans'],
                'frame': results['frame'].numpy() if hasattr(results['frame'], 'numpy') else results['frame'],
            })
        
        if k + 1 >= max_humans:
            break
    
    if not all_results:
        return None
    
    # Combine all tracks for scale estimation (use first/longest track primarily)
    combined = {
        'pred_trans': all_results[0]['pred_trans'],
        'frame': all_results[0]['frame'],
        'img_focal': img_focal,
        'img_center': img_center,
    }
    
    return combined


def estimate_scale_harp(slam_depths, tstamps, vimo_data):
    """Estimate scale using HARP (boundary-based)"""
    print('\n[4/4] Scale Estimation (HARP)...')
    
    pred_trans = vimo_data['pred_trans'].reshape(-1, 3)
    img_focal = float(vimo_data['img_focal'])
    img_center = vimo_data['img_center']
    frames = vimo_data['frame']
    
    scales = []
    
    for i, t in enumerate(tstamps):
        if i >= len(slam_depths):
            continue
        
        # Find corresponding VIMO frame
        frame_idx = np.where(frames == t)[0]
        if len(frame_idx) == 0:
            continue
        
        tx, ty, tz = pred_trans[frame_idx[0]]
        slam_depth = slam_depths[i]
        
        s = est_scale_boundary(slam_depth, tz, tx, ty, img_focal, img_center)
        if s is not None and 0.5 < s < 20:
            scales.append(s)
    
    if not scales:
        print("  Warning: No valid boundary scales!")
        return None
    
    scale = np.median(scales)
    print(f"  α_bound = {scale:.3f} (from {len(scales)}/{len(tstamps)} keyframes)")
    return scale


def estimate_scale_tram(slam_depths, tstamps, masks_np, img_folder):
    """Estimate scale using TRAM (ZoeDepth-based)"""
    print('\n[4/4] Scale Estimation (TRAM)...')
    
    repo = "isl-org/ZoeDepth"
    zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True).eval().cuda()
    
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    H, W = get_dimention(img_folder)
    scales = []
    
    for i, t in enumerate(tqdm(tstamps, desc="  ZoeDepth")):
        if i >= len(slam_depths):
            continue
        img = cv2.imread(imgfiles[t])[:, :, ::-1]
        img = cv2.resize(img, (W, H))
        pred_depth = zoe.infer_pil(Image.fromarray(img))
        msk = masks_np[t] if t < len(masks_np) else None
        s = est_scale_hybrid(slam_depths[i], pred_depth, msk=msk)
        scales.append(s)
    
    scale = np.median(scales)
    print(f"  α_bg = {scale:.3f}")
    return scale


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='HARP: Human Anchor for Robust Positioning')
    parser.add_argument("--video", type=str, required=True, help='Input video file')
    parser.add_argument("--static_camera", action='store_true')
    parser.add_argument("--visualize_mask", action='store_true')
    parser.add_argument("--method", choices=['tram', 'harp'], default='harp',
                        help='Scale estimation method (default: harp)')
    parser.add_argument("--max_humans", type=int, default=20,
                        help='Maximum number of humans to reconstruct')
    parser.add_argument("--skip_if_exists", action='store_true',
                        help='Skip if output already exists')
    args = parser.parse_args()
    
    # Setup paths
    video_path = args.video
    seq = os.path.basename(video_path).split('.')[0]
    seq_folder = f'results/{seq}'
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    
    method = args.method
    
    print(f"\n{'='*60}")
    print(f"HARP Pipeline: {seq}")
    print(f"Method: {method}")
    print(f"Output: {seq_folder}/camera_{method}.npy")
    print(f"{'='*60}")
    
    # Check if already done
    if args.skip_if_exists and os.path.exists(f'{seq_folder}/camera_{method}.npy'):
        print(f"\nSkipping: camera_{method}.npy already exists")
        return
    
    # Extract frames
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        print('\n[0/4] Extracting frames...')
        nframes = video2frames(video_path, img_folder)
        print(f"  {nframes} frames extracted")
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    else:
        print(f'\n[0/4] Using existing frames ({len(imgfiles)} images)')
    
    # Stage 1: Detection
    boxes_file = f'{seq_folder}/boxes.npy'
    masks_file = f'{seq_folder}/masks.npy'
    tracks_file = f'{seq_folder}/tracks.npy'
    
    if os.path.exists(boxes_file) and os.path.exists(masks_file) and os.path.exists(tracks_file):
        print('\n[1/4] Loading existing detection results...')
        boxes = np.load(boxes_file, allow_pickle=True)
        masks = np.load(masks_file, allow_pickle=True)
        tracks_dict = np.load(tracks_file, allow_pickle=True).item()
    else:
        boxes, masks, tracks_dict = run_detection(img_folder, seq_folder, args.visualize_mask)
        np.save(boxes_file, boxes)
        np.save(masks_file, masks)
        np.save(tracks_file, tracks_dict)
    
    # Prepare masks
    masks_np = np.array([masktool.decode(m) for m in masks])
    masks_torch = torch.from_numpy(masks_np)
    
    # Calibrate intrinsics
    cam_int, is_static = calibrate_intrinsics(
        img_folder, masks_torch, is_static=args.static_camera
    )
    img_focal = cam_int[0]
    img_center = cam_int[2:]
    
    # Stage 2: SLAM
    cam_R, cam_T_rel, slam_depths, tstamps, traj = run_slam_only(
        img_folder, masks_torch, cam_int, is_static
    )
    
    # Stage 3: VIMO (check if hps folder already has results)
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))
    if len(hps_files) > 0:
        print(f'\n[3/4] Loading existing VIMO results ({len(hps_files)} tracks)')
        # Load first track for scale estimation
        first_hps = np.load(hps_files[0], allow_pickle=True).item()
        vimo_data = {
            'pred_trans': first_hps['pred_trans'].numpy() if hasattr(first_hps['pred_trans'], 'numpy') else first_hps['pred_trans'],
            'frame': first_hps['frame'].numpy() if hasattr(first_hps['frame'], 'numpy') else first_hps['frame'],
            'img_focal': img_focal,
            'img_center': img_center,
        }
    else:
        vimo_data = run_vimo(img_folder, tracks_dict, img_focal, img_center, 
                            hps_folder, args.max_humans)
    
    if vimo_data is None:
        print("\nError: VIMO failed")
        return
    
    # Stage 4: Scale estimation
    if method == 'harp':
        scale = estimate_scale_harp(slam_depths, tstamps, vimo_data)
        if scale is None:
            print("  HARP failed, falling back to TRAM...")
            scale = estimate_scale_tram(slam_depths, tstamps, masks_np, img_folder)
    else:  # tram
        scale = estimate_scale_tram(slam_depths, tstamps, masks_np, img_folder)
    
    # Apply scale
    cam_T = cam_T_rel * scale
    
    # Align to world
    wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
    
    # Save camera with method suffix
    camera = {
        'pred_cam_R': cam_R.numpy(),
        'pred_cam_T': cam_T.numpy(),
        'world_cam_R': wd_cam_R.numpy(),
        'world_cam_T': wd_cam_T.numpy(),
        'img_focal': img_focal,
        'img_center': img_center,
        'spec_focal': spec_f,
        'scale': scale,
        'method': method,
    }
    
    np.save(f'{seq_folder}/camera_{method}.npy', camera)
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Camera: {seq_folder}/camera_{method}.npy")
    print(f"  Scale:  {scale:.3f}")
    print(f"{'='*60}")
    print(f"\nTo visualize:")
    print(f"  python scripts/visualize_tram.py --video {video_path} --method {method}")


if __name__ == '__main__':
    main()