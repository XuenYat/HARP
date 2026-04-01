"""
HARP: Background Degradation End-to-End Evaluation
====================================================
Uses normal SLAM results but re-runs ZoeDepth on black-background images.
Then compares TRAM (bg-only scale) vs HARP (pooled scale) on full trajectory metrics.

This isolates the effect of background degradation on scale estimation only.

Usage:
    cd /workspace/HARP
    PYTHONPATH=/workspace/HARP CUDA_VISIBLE_DEVICES=0 python scripts/harp_degradation_e2e.py
"""

import sys
import os
sys.path.insert(0, '/workspace/HARP')

import numpy as np
import cv2
from glob import glob
from PIL import Image
import torch
import pickle as pkl
from torchvision.transforms import ToTensor
from tqdm import tqdm
from collections import defaultdict

from lib.camera.est_scale import est_scale_hybrid
from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.models.smpl import SMPL

smpls = {g: SMPL(gender=g) for g in ['neutral', 'male', 'female']}
tt = lambda x: torch.from_numpy(x).float()
m2mm = 1e3


def find_emdb2_sequences(emdb_root):
    seqs = []
    for p in range(10):
        folder = f'{emdb_root}/P{p}'
        if not os.path.exists(folder):
            continue
        for seq_path in sorted(glob(f'{folder}/*')):
            parent = os.path.basename(os.path.dirname(seq_path))
            seq = os.path.basename(seq_path)
            pkl_file = f'{seq_path}/{parent}_{seq}_data.pkl'
            if not os.path.exists(pkl_file):
                continue
            ann = pkl.load(open(pkl_file, 'rb'))
            if ann.get('emdb2', False):
                seqs.append({'path': seq_path, 'parent': parent,
                             'name': seq, 'ann': ann})
    return seqs


def predict_depth_gpu(model, img_np):
    x = ToTensor()(Image.fromarray(img_np)).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)['metric_depth']
    return out.squeeze().cpu().numpy()


def compute_degraded_scales(model, seq_info, int_dir, smpl_dir):
    """
    Re-run ZoeDepth on black-background images, recompute scales.
    Returns TRAM scale (bg only) and HARP scale (pooled) under degradation.
    """
    seq = seq_info['name']
    parent = seq_info['parent']
    emdb_root = os.path.dirname(os.path.dirname(seq_info['path']))

    meta = dict(np.load(f'{int_dir}/{seq}.npz'))
    smpl_data = dict(np.load(f'{smpl_dir}/{seq}.npz', allow_pickle=True))

    tstamps = meta['tstamps']
    scales_bg_orig = meta['scales_bg']
    pred_trans = smpl_data['pred_trans'].reshape(-1, 3)
    img_focal = float(smpl_data['img_focal'])
    img_center = smpl_data['img_center']

    img_folder = f'{seq_info["path"]}/images'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    scales_bg_degraded = []
    scales_human_degraded = []

    for i in range(len(tstamps)):
        t = tstamps[i]

        # Load SLAM depth (from normal conditions)
        kf_file = f'{int_dir}/{seq}_depths/keyframe_{i:04d}.npz'
        if not os.path.exists(kf_file):
            continue
        kf = dict(np.load(kf_file))
        slam_depth = kf['slam_depth']
        pred_depth_orig = kf['pred_depth']

        # Load image and create black background
        if t >= len(imgfiles):
            continue
        img = cv2.imread(imgfiles[t])[:, :, ::-1]
        img_resized = cv2.resize(img, (512, 384))

        # Create human mask from original depth
        median_d = np.median(pred_depth_orig[pred_depth_orig > 0])
        human_mask = pred_depth_orig < median_d * 0.8
        mask_img = cv2.resize(human_mask.astype(np.uint8), (512, 384))

        # Black background version
        img_black = img_resized.copy()
        img_black[mask_img == 0] = 0

        # Re-run ZoeDepth on degraded image
        pred_depth_degraded = predict_depth_gpu(model, img_black)
        pred_depth_degraded = cv2.resize(pred_depth_degraded,
                                          (slam_depth.shape[1], slam_depth.shape[0]))

        # Compute TRAM's α_bg on degraded image
        msk_human = cv2.resize(mask_img.astype(np.float32),
                                (slam_depth.shape[1], slam_depth.shape[0]))
        try:
            scale_bg = est_scale_hybrid(slam_depth, pred_depth_degraded, msk=msk_human)
            scales_bg_degraded.append(scale_bg)
        except:
            continue

        # Compute HARP's α_human on degraded image
        if t < len(pred_trans):
            tx, ty, tz = pred_trans[t]
            if tz > 0.1:
                pd_H, pd_W = pred_depth_degraded.shape
                u = img_focal * tx / tz + img_center[0]
                v = img_focal * ty / tz + img_center[1]
                orig_W, orig_H = img_center[0] * 2, img_center[1] * 2
                u_pd = int(u * pd_W / orig_W)
                v_pd = int(v * pd_H / orig_H)
                radius = max(int(pd_H * 0.08), 15)
                v_min, v_max = max(0, v_pd-radius), min(pd_H, v_pd+radius)
                u_min, u_max = max(0, u_pd-radius), min(pd_W, u_pd+radius)
                if v_max > v_min and u_max > u_min:
                    patch = pred_depth_degraded[v_min:v_max, u_min:u_max]
                    valid = patch[(patch > 0) & np.isfinite(patch)]
                    if len(valid) > 10:
                        zoe_z = np.median(valid)
                        if zoe_z > 0:
                            alpha_human = scale_bg * (tz / zoe_z)
                            scales_human_degraded.append(alpha_human)

    if not scales_bg_degraded:
        return None, None

    # TRAM: median of bg scales only
    scale_tram = np.median(scales_bg_degraded)

    # HARP: median of pooled bg + human scales
    if scales_human_degraded:
        all_scales = np.concatenate([scales_bg_degraded, scales_human_degraded])
        scale_harp = np.median(all_scales)
    else:
        scale_harp = scale_tram

    return scale_tram, scale_harp


def evaluate_trajectory(gt_ann, smpl_data, cam_data, scale, scale_bg_orig):
    """Evaluate full trajectory with given scale."""
    ext = gt_ann['camera']['extrinsics']
    valid = gt_ann['good_frames_mask']
    gender = gt_ann['gender']
    poses_body = gt_ann['smpl']['poses_body']
    poses_root = gt_ann['smpl']['poses_root']
    betas = np.repeat(gt_ann['smpl']['betas'].reshape(1, -1),
                      gt_ann['n_frames'], axis=0)
    trans = gt_ann['smpl']['trans']

    # GT
    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root),
                       betas=tt(betas), transl=tt(trans),
                       pose2rot=True, default_smpl=True)
    gt_j3d = gt.joints[:, :24]
    gt_ori = axis_angle_to_matrix(tt(poses_root))

    # Pred
    pred_rotmat = torch.tensor(smpl_data['pred_rotmat'])
    pred_shape = torch.tensor(smpl_data['pred_shape'])
    pred_trans = torch.tensor(smpl_data['pred_trans'])
    mean_shape = pred_shape.mean(0, keepdim=True).repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:, 1:],
                            global_orient=pred_rotmat[:, [0]],
                            betas=mean_shape, transl=pred_trans.squeeze(),
                            pose2rot=False, default_smpl=True)
    pred_j3d = pred.joints[:, :24]

    pred_camr = torch.tensor(cam_data['pred_cam_R'])
    pred_camt_orig = torch.tensor(cam_data['pred_cam_T'])

    # Rescale camera translation with new scale
    unscaled = pred_camt_orig / scale_bg_orig
    pred_camt = unscaled * scale

    T = min(len(gt_j3d), len(pred_j3d))
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr[:T], pred_j3d[:T]) + pred_camt[:T, None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr[:T], pred_rotmat[:T, 0])

    valid = valid[:T]
    gt_j3d = gt_j3d[:T][valid]
    gt_ori = gt_ori[:T][valid]
    pred_j3d_w = pred_j3d_w[valid]
    pred_ori_w = pred_ori_w[valid]

    # RTE
    rte = compute_rte(gt_j3d[:, 0], pred_j3d_w[:, 0]) * 1e2

    # W-MPJPE100
    chunk_length = 100
    w_mpjpe_list = []
    n_valid = valid.sum()
    for start in range(0, n_valid - chunk_length, chunk_length):
        end = start + chunk_length
        if start + 2 * chunk_length > n_valid:
            end = n_valid - 1
        target = gt_j3d[start:end].clone().cpu()
        pred = pred_j3d_w[start:end].clone().cpu()
        w_j3d = first_align_joints(target, pred)
        w_mpjpe_list.append(compute_jpe(target, w_j3d))

    w_mpjpe = np.concatenate(w_mpjpe_list) * m2mm if w_mpjpe_list else np.array([0.0])

    return rte.mean(), w_mpjpe.mean()


def main():
    emdb_root = '/workspace/dataset/emdb'
    int_dir = '/workspace/HARP/results/emdb_harp/intermediates'
    smpl_dir = '/workspace/HARP/results/emdb/smpl'
    cam_dir = '/workspace/HARP/results/emdb/camera'
    save_dir = '/workspace/HARP/results/harp_degradation_e2e'
    os.makedirs(save_dir, exist_ok=True)

    # Load ZoeDepth
    repo = "isl-org/ZoeDepth"
    zoe_model = torch.hub.load(repo, "ZoeD_N", pretrained=True).eval().cuda()

    sequences = find_emdb2_sequences(emdb_root)
    print(f"Found {len(sequences)} EMDB2 sequences\n")

    results = []

    for seq_info in tqdm(sequences, desc="Sequences"):
        seq = seq_info['name']

        smpl_file = f'{smpl_dir}/{seq}.npz'
        cam_file = f'{cam_dir}/{seq}.npz'
        meta_file = f'{int_dir}/{seq}.npz'

        if not all(os.path.exists(f) for f in [smpl_file, cam_file, meta_file]):
            continue

        smpl_data = dict(np.load(smpl_file, allow_pickle=True))
        cam_data = dict(np.load(cam_file))
        meta = dict(np.load(meta_file))
        scale_bg_orig = float(meta['scale_bg_final'])

        # Compute degraded scales
        scale_tram_deg, scale_harp_deg = compute_degraded_scales(
            zoe_model, seq_info, int_dir, smpl_dir
        )
        if scale_tram_deg is None:
            continue

        # Evaluate: normal conditions
        rte_tram_normal, wmpjpe_tram_normal = evaluate_trajectory(
            seq_info['ann'], smpl_data, cam_data, scale_bg_orig, scale_bg_orig
        )

        # Evaluate: degraded with TRAM scale
        rte_tram_deg, wmpjpe_tram_deg = evaluate_trajectory(
            seq_info['ann'], smpl_data, cam_data, scale_tram_deg, scale_bg_orig
        )

        # Evaluate: degraded with HARP scale
        rte_harp_deg, wmpjpe_harp_deg = evaluate_trajectory(
            seq_info['ann'], smpl_data, cam_data, scale_harp_deg, scale_bg_orig
        )

        results.append({
            'seq': seq,
            'rte_tram_normal': rte_tram_normal,
            'rte_tram_degraded': rte_tram_deg,
            'rte_harp_degraded': rte_harp_deg,
            'wmpjpe_tram_normal': wmpjpe_tram_normal,
            'wmpjpe_tram_degraded': wmpjpe_tram_deg,
            'wmpjpe_harp_degraded': wmpjpe_harp_deg,
        })

        print(f"  {seq}: Normal RTE={rte_tram_normal:.2f}% | "
              f"Degraded: TRAM={rte_tram_deg:.2f}% HARP={rte_harp_deg:.2f}%")

    # Summary
    print("\n" + "=" * 85)
    print("End-to-End Evaluation: Normal vs Background Degradation")
    print("=" * 85)

    print(f"\n{'':30s} {'Normal':>12s} {'Degraded (TRAM)':>16s} {'Degraded (HARP)':>16s}")
    print("-" * 85)

    rte_n = np.mean([r['rte_tram_normal'] for r in results])
    rte_td = np.mean([r['rte_tram_degraded'] for r in results])
    rte_hd = np.mean([r['rte_harp_degraded'] for r in results])
    wm_n = np.mean([r['wmpjpe_tram_normal'] for r in results])
    wm_td = np.mean([r['wmpjpe_tram_degraded'] for r in results])
    wm_hd = np.mean([r['wmpjpe_harp_degraded'] for r in results])

    print(f"{'RTE (%)':<30s} {rte_n:>12.2f} {rte_td:>16.2f} {rte_hd:>16.2f}")
    print(f"{'W-MPJPE100 (mm)':<30s} {wm_n:>12.1f} {wm_td:>16.1f} {wm_hd:>16.1f}")

    print(f"\nTRAM degradation: RTE {rte_n:.2f}% → {rte_td:.2f}% "
          f"({(rte_td/rte_n - 1)*100:.0f}% worse)")
    print(f"HARP degradation: RTE {rte_n:.2f}% → {rte_hd:.2f}% "
          f"({(rte_hd/rte_n - 1)*100:.0f}% worse)")
    print(f"HARP vs TRAM under degradation: "
          f"RTE {rte_hd:.2f}% vs {rte_td:.2f}% "
          f"({(1 - rte_hd/rte_td)*100:.0f}% better)")

    # Save
    np.savez(f'{save_dir}/degradation_e2e_results.npz',
             **{f'{r["seq"]}_{k}': v for r in results for k, v in r.items() if k != 'seq'})
    print(f"\nResults saved to {save_dir}/")


if __name__ == '__main__':
    main()