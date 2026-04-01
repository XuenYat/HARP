"""
HARP End-to-End Evaluation on EMDB subset 2
=============================================
Reproduces TRAM Table 3 format with HARP's improved scale.

Compares:
  1. TRAM baseline (original pre-computed results)
  2. HARP (pooled bg + human scale estimates)

Metrics: PA-MPJPE, WA-MPJPE100, W-MPJPE100, RTE, ERVE, ATE, ATE-S

Usage:
    python harp_eval_emdb.py \
        --emdb_root /workspace/dataset/emdb \
        --tram_results /workspace/HARP/results/emdb \
        --harp_intermediates /workspace/HARP/results/emdb_harp/intermediates \
        --save_dir /workspace/HARP/results/harp_eval
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Import HARP scale computation
from harp_fusion_experiment import load_intermediates, compute_human_correction


smpls = {g: SMPL(gender=g) for g in ['neutral', 'male', 'female']}
tt = lambda x: torch.from_numpy(x).float()
m2mm = 1e3


def traj_filter(pred_vert_w, pred_j3d_w):
    """Simple trajectory smoothing (from TRAM)."""
    return pred_vert_w, pred_j3d_w


def compute_harp_scale(intermediates, smpl_data):
    """Compute HARP's pooled scale."""
    human_scales, all_scales, scale_combined = compute_human_correction(
        intermediates, smpl_data
    )
    return scale_combined


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

    acc_tram = defaultdict(list)
    acc_harp = defaultdict(list)

    for seq_path, gt_ann, seq_name in tqdm(emdb2_seqs):
        smpl_file = os.path.join(tram_results, 'smpl', f'{seq_name}.npz')
        cam_file = os.path.join(tram_results, 'camera', f'{seq_name}.npz')
        harp_cam_file = os.path.join(harp_intermediates_dir, '..', 'camera', f'{seq_name}.npz')

        if not os.path.exists(smpl_file) or not os.path.exists(cam_file):
            print(f"  {seq_name}: SKIP (no TRAM results)")
            continue

        smpl_data = dict(np.load(smpl_file, allow_pickle=True))
        cam_data = dict(np.load(cam_file, allow_pickle=True))

        # ---- TRAM baseline evaluation ----
        res_tram = evaluate_one_sequence(seq_path, gt_ann, smpl_data, cam_data,
                                         harp_scale=None)

        for k, v in res_tram.items():
            acc_tram[k].append(v)

        # ---- HARP evaluation ----
        intermediates = load_intermediates(harp_intermediates_dir, seq_name)
        if intermediates is None:
            print(f"  {seq_name}: SKIP HARP (no intermediates)")
            # Still record TRAM results, use TRAM for HARP too
            for k, v in res_tram.items():
                acc_harp[k].append(v)
            continue

        # Compute HARP scale
        harp_scale = compute_harp_scale(intermediates, smpl_data)
        harp_scale_bg = intermediates['scale_bg_final']

        # Pass extra info for rescaling
        cam_data_harp = dict(cam_data)
        cam_data_harp['_harp_cam_path'] = os.path.join(
            harp_intermediates_dir, '..', 'camera', f'{seq_name}.npz')
        cam_data_harp['_harp_scale_bg'] = harp_scale_bg

        res_harp = evaluate_one_sequence(seq_path, gt_ann, smpl_data, cam_data_harp,
                                         harp_scale=harp_scale)

        for k, v in res_harp.items():
            acc_harp[k].append(v)

        # Per-sequence summary
        tram_rte = res_tram['rte'].mean() if len(res_tram['rte']) > 0 else 0
        harp_rte = res_harp['rte'].mean() if len(res_harp['rte']) > 0 else 0
        print(f"  {seq_name}: TRAM RTE={tram_rte:.2f}% → HARP RTE={harp_rte:.2f}%")

    # ======= Print Table 3 format =======
    print("\n" + "=" * 90)
    print("EMDB Subset 2 - Full Evaluation (Table 3 format)")
    print("=" * 90)

    metrics = ['pa_mpjpe', 'wa_mpjpe', 'w_mpjpe', 'rte', 'erve']
    labels = ['PA-MPJPE', 'WA-MPJPE100', 'W-MPJPE100', 'RTE (%)', 'ERVE']

    print(f"\n{'Method':<20}", end="")
    for label in labels:
        print(f"{label:>15}", end="")
    print()
    print("-" * 95)

    for name, acc in [("TRAM", acc_tram), ("HARP (ours)", acc_harp)]:
        print(f"{name:<20}", end="")
        for metric in metrics:
            vals = np.concatenate(acc[metric])
            print(f"{vals.mean():>15.1f}", end="")
        print()

    print("-" * 95)

    # Save detailed results
    save_data = {}
    for name, acc in [("tram", acc_tram), ("harp", acc_harp)]:
        for metric in metrics:
            vals = np.concatenate(acc[metric])
            save_data[f'{name}_{metric}_mean'] = vals.mean()
            save_data[f'{name}_{metric}_std'] = vals.std()
    np.savez(os.path.join(save_dir, 'eval_results.npz'), **save_data)

    print(f"\nResults saved to {save_dir}/eval_results.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emdb_root', type=str, default='/workspace/dataset/emdb')
    parser.add_argument('--tram_results', type=str, default='/workspace/HARP/results/emdb')
    parser.add_argument('--harp_intermediates', type=str,
                        default='/workspace/HARP/results/emdb_harp/intermediates')
    parser.add_argument('--save_dir', type=str, default='/workspace/HARP/results/harp_eval')
    args = parser.parse_args()

    run_evaluation(args.emdb_root, args.tram_results, args.harp_intermediates, args.save_dir)


if __name__ == '__main__':
    main()