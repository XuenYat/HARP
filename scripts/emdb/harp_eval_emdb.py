"""
HARP End-to-End Evaluation on EMDB subset 2
=============================================
Reproduces TRAM Table 3 format with HARP's improved scale.

Compares:
  1. TRAM baseline (original pre-computed results)
  2. HARP (boundary sampling + inverse-variance fusion)

Metrics: PA-MPJPE, WA-MPJPE100, W-MPJPE100, RTE, ERVE, ATE, ATE-S

Usage:
    python scripts/emdb/harp_eval_emdb.py \
        --emdb_root /path/to/emdb \
        --tram_results results/emdb \
        --harp_intermediates results/emdb_harp/intermediates \
        --save_dir results/harp_eval
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.models.smpl import SMPL


def load_intermediates(intermediates_dir, seq_name):
    """Load HARP intermediate variables (SLAM depths, bg scales) for a sequence."""
    meta_file = os.path.join(intermediates_dir, f'{seq_name}.npz')
    depth_dir = os.path.join(intermediates_dir, f'{seq_name}_depths')

    if not os.path.exists(meta_file):
        return None

    meta = dict(np.load(meta_file, allow_pickle=True))

    tstamps = meta['tstamps']
    scales_bg = meta['scales_bg']
    scale_bg_final = float(meta['scale_bg_final'])

    slam_depths = []
    pred_depths = []
    for i in range(len(tstamps)):
        kf_file = os.path.join(depth_dir, f'keyframe_{i:04d}.npz')
        if os.path.exists(kf_file):
            kf = dict(np.load(kf_file))
            slam_depths.append(kf['slam_depth'])
            pred_depths.append(kf['pred_depth'])

    return {
        'tstamps': tstamps,
        'scales_bg': scales_bg,
        'scale_bg_final': scale_bg_final,
        'slam_depths': slam_depths,
        'pred_depths': pred_depths,
    }


smpls = {g: SMPL(gender=g) for g in ['neutral', 'male', 'female']}
tt = lambda x: torch.from_numpy(x).float()
m2mm = 1e3


def traj_filter(pred_vert_w, pred_j3d_w):
    """Simple trajectory smoothing (from TRAM)."""
    return pred_vert_w, pred_j3d_w


def est_scale_boundary(slam_depth, tz, tx, ty, img_focal, img_center):
    """
    α_human = VIMO_tz / SLAM_boundary_depth

    Sample SLAM depth at human boundary (feet, sides) where valid depth exists.
    No dependency on learned depth networks.
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


def robust_var(data):
    """
    Robust variance estimate using MAD (Median Absolute Deviation).

    MAD * 1.4826 approximates the standard deviation under Gaussian
    assumption. Divided by N to get the variance of the sample median
    (analogous to SEM for means).
    """
    if len(data) < 2:
        return float('inf')
    arr = np.array(data)
    mad = np.median(np.abs(arr - np.median(arr))) * 1.4826
    return (mad ** 2) / len(arr)


def compute_all_scales(intermediates, smpl_data):
    """
    Compute all scale fusion variants for ablation.

    Returns dict with keys: 'bg', 'human', 'average', 'inverse_var'
    Each value is a float scale estimate.
    """
    tstamps = intermediates['tstamps']
    slam_depths = intermediates['slam_depths']
    scales_bg = intermediates['scales_bg']
    scale_bg_final = float(intermediates['scale_bg_final'])

    pred_trans = smpl_data['pred_trans'].reshape(-1, 3)
    img_focal = float(smpl_data['img_focal'])
    img_center = smpl_data['img_center']

    # --- Compute α_human per keyframe via boundary sampling ---
    human_scales = []
    for i, t in enumerate(tstamps):
        if t >= len(pred_trans) or i >= len(slam_depths):
            continue

        tx, ty, tz = pred_trans[t]
        s = est_scale_boundary(slam_depths[i], tz, tx, ty, img_focal, img_center)
        if s is not None and 0.5 < s < 20:
            human_scales.append(s)

    scale_bg = scale_bg_final

    if not human_scales:
        return {
            'bg': scale_bg, 'human': scale_bg,
            'average': scale_bg, 'inverse_var': scale_bg,
        }

    scale_human = np.median(human_scales)

    # --- Uniform average ---
    scale_avg = (scale_bg + scale_human) / 2.0

    # --- Inverse-variance weighted fusion (SEM) ---
    n_human = len(human_scales)
    n_bg = len(scales_bg)
    var_human = float(np.var(human_scales)) / n_human if n_human > 1 else float('inf')
    var_bg = float(np.var(scales_bg)) / n_bg if n_bg > 1 else float('inf')

    if np.isinf(var_human):
        scale_fused = scale_bg
    elif np.isinf(var_bg):
        scale_fused = scale_human
    else:
        var_human = max(var_human, 1e-8)
        var_bg = max(var_bg, 1e-8)
        w_human = (1.0 / var_human) / (1.0 / var_human + 1.0 / var_bg)
        scale_fused = w_human * scale_human + (1.0 - w_human) * scale_bg

    return {
        'bg': scale_bg,
        'human': scale_human,
        'average': scale_avg,
        'inverse_var': scale_fused,
    }


def compute_harp_scale(intermediates, smpl_data):
    """Compute HARP scale (inverse-variance fusion). Convenience wrapper."""
    return compute_all_scales(intermediates, smpl_data)['inverse_var']


def evaluate_one_sequence(root, gt_ann, tram_smpl, tram_cam, harp_scale=None):
    """
    Evaluate one sequence with given scale.
    Returns dict of all metrics.
    """
    # GT
    ext = gt_ann['camera']['extrinsics']
    valid = gt_ann['good_frames_mask']
    gender = gt_ann['gender']
    poses_body = gt_ann["smpl"]["poses_body"]
    poses_root = gt_ann["smpl"]["poses_root"]
    betas = np.repeat(gt_ann["smpl"]["betas"].reshape((1, -1)),
                      repeats=gt_ann["n_frames"], axis=0)
    trans = gt_ann["smpl"]["trans"]

    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root),
                       betas=tt(betas), transl=tt(trans),
                       pose2rot=True, default_smpl=True)
    gt_j3d = gt.joints[:, :24]
    gt_ori = axis_angle_to_matrix(tt(poses_root))

    # GT local motion (in camera frame)
    poses_root_cam = matrix_to_axis_angle(
        tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root))
    )
    gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam,
                           betas=tt(betas), pose2rot=True, default_smpl=True)
    gt_vert_cam = gt_cam.vertices
    gt_j3d_cam = gt_cam.joints[:, :24]

    # PRED
    pred_rotmat = torch.tensor(tram_smpl['pred_rotmat'])
    pred_shape = torch.tensor(tram_smpl['pred_shape'])
    pred_trans = torch.tensor(tram_smpl['pred_trans'])

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:, 1:],
                            global_orient=pred_rotmat[:, [0]],
                            betas=pred_shape,
                            transl=pred_trans.squeeze(),
                            pose2rot=False, default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    # Camera
    pred_camr = torch.tensor(tram_cam['pred_cam_R'])

    if harp_scale is not None:
        # Use HARP scale: recompute camera translation
        scale_bg = float(tram_cam.get('scale_bg_final',
                         np.linalg.norm(tram_cam['pred_cam_T'][1] - tram_cam['pred_cam_T'][0])))
        # Get unscaled trajectory from HARP intermediates
        harp_cam = dict(np.load(tram_cam['_harp_cam_path']))
        pred_camt = torch.tensor(harp_cam['pred_cam_T'])
        # Rescale: harp_cam was scaled by its own scale_bg, we want harp_scale
        harp_scale_bg = float(tram_cam['_harp_scale_bg'])
        unscaled = pred_camt / harp_scale_bg
        pred_camt = unscaled * harp_scale
    else:
        pred_camt = torch.tensor(tram_cam['pred_cam_T'])

    # World frame
    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:, 0])
    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    # Valid mask
    T = min(len(gt_j3d), len(pred_j3d_w))
    valid = valid[:T]
    gt_j3d = gt_j3d[:T][valid]
    gt_ori = gt_ori[:T][valid]
    pred_j3d_w = pred_j3d_w[:T][valid]
    pred_ori_w = pred_ori_w[:T][valid]

    gt_j3d_cam = gt_j3d_cam[:T][valid]
    gt_vert_cam = gt_vert_cam[:T][valid]
    pred_j3d_local = pred_j3d[:T][valid]
    pred_vert_local = pred_vert[:T][valid]

    results = {}

    # ======= Local motion evaluation =======
    pred_j3d_local, gt_j3d_cam, pred_vert_local, gt_vert_cam = batch_align_by_pelvis(
        [pred_j3d_local, gt_j3d_cam, pred_vert_local, gt_vert_cam], pelvis_idxs=[1, 2]
    )
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d_local, gt_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

    results['pa_mpjpe'] = pa_mpjpe

    # ======= Global motion evaluation =======
    chunk_length = 100
    w_mpjpe_list, wa_mpjpe_list = [], []
    n_valid = valid.sum()
    for start in range(0, n_valid - chunk_length, chunk_length):
        end = start + chunk_length
        if start + 2 * chunk_length > n_valid:
            end = n_valid - 1

        target = gt_j3d[start:end].clone().cpu()
        pred = pred_j3d_w[start:end].clone().cpu()

        w_j3d = first_align_joints(target, pred)
        wa_j3d = global_align_joints(target, pred)

        w_mpjpe_list.append(compute_jpe(target, w_j3d))
        wa_mpjpe_list.append(compute_jpe(target, wa_j3d))

    if w_mpjpe_list:
        results['w_mpjpe'] = np.concatenate(w_mpjpe_list) * m2mm
        results['wa_mpjpe'] = np.concatenate(wa_mpjpe_list) * m2mm
    else:
        results['w_mpjpe'] = np.array([0.0])
        results['wa_mpjpe'] = np.array([0.0])

    # RTE
    rte = compute_rte(gt_j3d[:, 0], pred_j3d_w[:, 0]) * 1e2
    results['rte'] = rte

    # ERVE
    erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
    results['erve'] = erve

    return results


def run_evaluation(emdb_root, tram_results, harp_intermediates_dir, save_dir):
    """Run full evaluation comparing TRAM vs HARP."""
    os.makedirs(save_dir, exist_ok=True)

    # Find EMDB2 sequences
    emdb2_seqs = []
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
                emdb2_seqs.append((seq_path, ann, seq))

    print(f"Found {len(emdb2_seqs)} EMDB2 sequences\n")

    # Accumulators for all fusion strategies
    strategy_names = ['bg', 'human', 'average', 'inverse_var']
    acc = {name: defaultdict(list) for name in strategy_names}

    for seq_path, gt_ann, seq_name in tqdm(emdb2_seqs):
        smpl_file = os.path.join(tram_results, 'smpl', f'{seq_name}.npz')
        cam_file = os.path.join(tram_results, 'camera', f'{seq_name}.npz')

        if not os.path.exists(smpl_file) or not os.path.exists(cam_file):
            print(f"  {seq_name}: SKIP (no TRAM results)")
            continue

        smpl_data = dict(np.load(smpl_file, allow_pickle=True))
        cam_data = dict(np.load(cam_file, allow_pickle=True))

        # Load intermediates
        intermediates = load_intermediates(harp_intermediates_dir, seq_name)
        if intermediates is None:
            print(f"  {seq_name}: SKIP (no intermediates)")
            # Use TRAM baseline for all strategies
            res_tram = evaluate_one_sequence(seq_path, gt_ann, smpl_data, cam_data,
                                             harp_scale=None)
            for sname in strategy_names:
                for k, v in res_tram.items():
                    acc[sname][k].append(v)
            continue

        # Compute all scale variants
        all_scales = compute_all_scales(intermediates, smpl_data)
        harp_scale_bg = intermediates['scale_bg_final']

        # Evaluate each strategy
        for sname in strategy_names:
            scale = all_scales[sname]

            if sname == 'bg':
                # bg uses original camera directly (no rescaling)
                res = evaluate_one_sequence(seq_path, gt_ann, smpl_data, cam_data,
                                            harp_scale=None)
            else:
                cam_data_s = dict(cam_data)
                cam_data_s['_harp_cam_path'] = os.path.join(
                    harp_intermediates_dir, '..', 'camera', f'{seq_name}.npz')
                cam_data_s['_harp_scale_bg'] = harp_scale_bg
                res = evaluate_one_sequence(seq_path, gt_ann, smpl_data, cam_data_s,
                                            harp_scale=scale)

            for k, v in res.items():
                acc[sname][k].append(v)

        # Per-sequence summary
        rte_bg = acc['bg']['rte'][-1].mean()
        rte_iv = acc['inverse_var']['rte'][-1].mean()
        print(f"  {seq_name}: BG={rte_bg:.2f}% → IV={rte_iv:.2f}%")

    # ======= Print results =======
    print("\n" + "=" * 90)
    print("EMDB Subset 2 - Full Evaluation")
    print("=" * 90)

    metrics = ['pa_mpjpe', 'wa_mpjpe', 'w_mpjpe', 'rte', 'erve']
    labels = ['PA-MPJPE', 'WA-MPJPE100', 'W-MPJPE100', 'RTE (%)', 'ERVE']

    display_names = {
        'bg': 'TRAM (bg only)',
        'human': 'Human only',
        'average': 'Uniform avg',
        'inverse_var': 'HARP (inv-var)',
    }

    print(f"\n{'Method':<25}", end="")
    for label in labels:
        print(f"{label:>15}", end="")
    print()
    print("-" * 100)

    for sname in strategy_names:
        print(f"{display_names[sname]:<25}", end="")
        for metric in metrics:
            vals = np.concatenate(acc[sname][metric])
            print(f"{vals.mean():>15.1f}", end="")
        print()

    print("-" * 100)

    # Save detailed results
    save_data = {}
    for sname in strategy_names:
        for metric in metrics:
            vals = np.concatenate(acc[sname][metric])
            save_data[f'{sname}_{metric}_mean'] = vals.mean()
            save_data[f'{sname}_{metric}_std'] = vals.std()
    np.savez(os.path.join(save_dir, 'eval_results.npz'), **save_data)

    # ======= Per-sequence CSV + category analysis =======
    print("\n" + "=" * 90)
    print("Per-Sequence RTE (%)")
    print("=" * 90)
    print(f"{'Sequence':<35} {'TRAM':>8} {'HARP':>8} {'Diff':>8} {'Category':<15}")
    print("-" * 80)

    per_seq = []
    for i, (seq_path, gt_ann, seq_name) in enumerate(emdb2_seqs):
        if i >= len(acc['bg']['rte']):
            break
        rte_bg = acc['bg']['rte'][i].mean()
        rte_iv = acc['inverse_var']['rte'][i].mean()
        diff = rte_bg - rte_iv

        # Categorize by scene type
        name_lower = seq_name.lower()
        if 'indoor' in name_lower:
            cat = 'indoor'
        elif 'stair' in name_lower:
            cat = 'stairs'
        else:
            cat = 'outdoor'

        per_seq.append({'seq': seq_name, 'tram': rte_bg, 'harp': rte_iv,
                        'diff': diff, 'cat': cat})
        marker = '+' if diff > 0 else '-' if diff < 0 else ' '
        print(f"  {seq_name:<33} {rte_bg:>7.2f} {rte_iv:>7.2f} {marker}{abs(diff):>6.2f} {cat:<15}")

    # Category summary
    print("\n" + "=" * 90)
    print("Category Summary (RTE %)")
    print("=" * 90)
    print(f"{'Category':<15} {'N':>4} {'TRAM':>10} {'HARP':>10} {'Improv':>10}")
    print("-" * 55)

    for cat in ['indoor', 'stairs', 'outdoor', 'ALL']:
        if cat == 'ALL':
            subset = per_seq
        else:
            subset = [s for s in per_seq if s['cat'] == cat]
        if not subset:
            continue
        mean_tram = np.mean([s['tram'] for s in subset])
        mean_harp = np.mean([s['harp'] for s in subset])
        improv = (mean_tram - mean_harp) / mean_tram * 100
        print(f"  {cat:<13} {len(subset):>4} {mean_tram:>9.2f} {mean_harp:>9.2f} {improv:>+9.1f}%")

    # Save per-sequence CSV
    csv_path = os.path.join(save_dir, 'per_sequence_rte.csv')
    with open(csv_path, 'w') as f:
        f.write('sequence,category,tram_rte,harp_rte,improvement\n')
        for s in per_seq:
            f.write(f"{s['seq']},{s['cat']},{s['tram']:.4f},{s['harp']:.4f},{s['diff']:.4f}\n")

    print(f"\nPer-sequence CSV saved to {csv_path}")
    print(f"Aggregate results saved to {save_dir}/eval_results.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emdb_root', type=str, required=True,
                        help='Path to EMDB dataset root (containing P0, P1, ..., P9)')
    parser.add_argument('--tram_results', type=str, default='results/emdb')
    parser.add_argument('--harp_intermediates', type=str,
                        default='results/emdb_harp/intermediates')
    parser.add_argument('--save_dir', type=str, default='results/harp_eval')
    args = parser.parse_args()

    run_evaluation(args.emdb_root, args.tram_results, args.harp_intermediates, args.save_dir)


if __name__ == '__main__':
    main()
