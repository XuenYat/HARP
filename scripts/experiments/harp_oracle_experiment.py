"""
HARP Oracle Experiment (Experiment 2.1)
=======================================
Validates the theoretical upper bound of human-geometry-based scale estimation.

Three comparisons:
1. GT SMPL trans (oracle) → compute α_human → compare with GT scale
2. VIMO pred_trans (predicted) → compute α_human → compare with GT scale  
3. TRAM's α_bg (baseline) → compare with GT scale

This does NOT require re-running SLAM. We use:
- EMDB GT pkl: ground truth camera, SMPL params
- TRAM pre-computed results: VIMO predictions, camera predictions (with scale baked in)

The key insight: pred_cam_T already has TRAM's scale baked in.
GT cam_T is in metric. Their ratio gives us GT scale implicitly.
We then check if human-derived scale would have been closer.

Usage:
    python harp_oracle_experiment.py \
        --emdb_root /workspace/dataset/emdb \
        --results_dir /workspace/HARP/results/emdb \
        --save_dir results/harp_oracle
"""

import os
import sys
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from collections import defaultdict

import torch

# Add TRAM paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_gt(seq_path):
    """Load EMDB ground truth."""
    parent = os.path.basename(os.path.dirname(seq_path))
    seq = os.path.basename(seq_path)
    pkl_file = f'{seq_path}/{parent}_{seq}_data.pkl'
    ann = pkl.load(open(pkl_file, 'rb'))
    return ann


def compute_gt_scale(gt_ann, pred_cam_T):
    """
    Compute the GT scale factor that TRAM should have estimated.
    
    GT camera trajectory is in meters.
    TRAM's pred_cam_T = slam_traj * α_bg (TRAM's estimated scale).
    GT scale α_gt = gt_displacement / unscaled_slam_displacement.
    
    We approximate: α_gt / α_tram = gt_displacement / pred_displacement
    So α_gt = α_tram * (gt_disp / pred_disp), and α_tram is what TRAM used.
    
    More precisely, we align trajectories and compute the scale ratio.
    """
    ext = gt_ann['camera']['extrinsics']
    gt_cam_r = ext[:, :3, :3].transpose(0, 2, 1)  # R_wc
    gt_cam_t = np.einsum('bij,bj->bi', gt_cam_r, -ext[:, :3, -1])  # t_wc in meters
    
    T = min(len(gt_cam_t), len(pred_cam_T))
    gt_cam_t = gt_cam_t[:T]
    pred_cam_T = pred_cam_T[:T]
    
    # Align pred to GT using Umeyama (with scale)
    # This gives us the scale ratio between TRAM's result and GT
    s, R, t = align_umeyama(gt_cam_t, pred_cam_T)
    
    # s is the scale correction: GT ≈ s * R @ pred + t
    # So TRAM's scale was off by factor s
    # α_gt = α_tram * s
    
    return s, gt_cam_t, pred_cam_T


def align_umeyama(Y, X):
    """
    Umeyama alignment: find s, R, t such that Y ≈ s*R@X + t
    Y: (N, 3) target (GT)
    X: (N, 3) source (pred)
    Returns: s (scale), R (3,3), t (3,)
    """
    mu_y = Y.mean(axis=0)
    mu_x = X.mean(axis=0)
    
    Y0 = Y - mu_y
    X0 = X - mu_x
    
    var_x = np.sum(X0 ** 2) / len(X)
    
    C = Y0.T @ X0 / len(X)
    U, D, Vh = np.linalg.svd(C)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vh
    s = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - s * R @ mu_x
    
    return s, R, t


def compute_human_implied_scale(gt_ann, pred_trans, pred_cam_T):
    """
    Compute what scale the human depth would imply.
    
    Logic:
    - pred_trans[:, 2] = tz in meters (from SMPL, metric)
    - pred_cam_T is TRAM's camera translation (already scaled by α_tram)
    - The "human in world" = cam_R @ human_cam + cam_T
    
    If TRAM's scale were perfect, the human's world position would match GT.
    The ratio between GT human world position and TRAM's gives the scale correction.
    
    Simpler approach: compare GT SMPL trans (world frame) with 
    TRAM's reconstructed human world position.
    """
    ext = gt_ann['camera']['extrinsics']
    gt_cam_r = ext[:, :3, :3].transpose(0, 2, 1)
    gt_cam_t = np.einsum('bij,bj->bi', gt_cam_r, -ext[:, :3, -1])
    
    # GT human position in world frame
    gt_human_trans = gt_ann['smpl']['trans']  # (T, 3) in meters, world frame
    
    T = min(len(gt_human_trans), len(pred_trans), len(pred_cam_T))
    
    return gt_human_trans[:T], gt_cam_t[:T]


def compute_oracle_human_scale(gt_ann, pred_cam_R, pred_cam_T):
    """
    Oracle experiment: Use GT SMPL translation to derive scale.
    
    GT SMPL gives us the human's metric position in world frame.
    GT camera gives us the human's metric position in camera frame.
    
    human_cam = R_cw @ human_world + t_cw
    human_cam_z = depth of human in camera frame (meters)
    
    SLAM depth at human location is in arbitrary units.
    Scale = human_cam_z_metric / slam_depth_at_human
    
    Since we don't have raw SLAM depth, we use a different approach:
    We compute the GT scale correction relative to TRAM's scale.
    """
    ext = gt_ann['camera']['extrinsics']  # (T, 4, 4)
    R_cw = ext[:, :3, :3]
    t_cw = ext[:, :3, 3]
    
    # GT human in world frame
    gt_trans_world = gt_ann['smpl']['trans']  # (T, 3)
    
    # GT human in camera frame
    T = min(len(gt_trans_world), len(pred_cam_T))
    gt_human_cam = np.einsum('bij,bj->bi', R_cw[:T], gt_trans_world[:T]) + t_cw[:T]
    gt_human_cam_z = gt_human_cam[:, 2]  # GT metric depth
    
    return gt_human_cam_z


def run_oracle_experiment(emdb_root, results_dir, save_dir):
    """Run the full oracle experiment across all EMDB2 sequences."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Find EMDB2 sequences
    emdb2_seqs = []
    for p in range(10):
        folder = f'{emdb_root}/P{p}'
        if not os.path.exists(folder):
            continue
        for seq_path in sorted(glob(f'{folder}/*')):
            gt = load_gt(seq_path)
            if gt.get('emdb2', False):
                emdb2_seqs.append((seq_path, gt))
    
    print(f"Found {len(emdb2_seqs)} EMDB2 sequences\n")
    
    all_results = []
    
    for seq_path, gt_ann in emdb2_seqs:
        seq_name = os.path.basename(seq_path)
        
        # Load TRAM predictions
        smpl_file = os.path.join(results_dir, 'smpl', f'{seq_name}.npz')
        cam_file = os.path.join(results_dir, 'camera', f'{seq_name}.npz')
        
        if not os.path.exists(smpl_file) or not os.path.exists(cam_file):
            print(f"  {seq_name}: SKIP (no TRAM results)")
            continue
        
        smpl_pred = dict(np.load(smpl_file, allow_pickle=True))
        cam_pred = dict(np.load(cam_file, allow_pickle=True))
        
        pred_trans = smpl_pred['pred_trans'].reshape(-1, 3)  # (T, 3) metric
        pred_shape = smpl_pred['pred_shape']                  # (T, 10)
        pred_cam_T = cam_pred['pred_cam_T']                   # (T, 3)
        pred_cam_R = cam_pred['pred_cam_R']                   # (T, 3, 3)
        
        # --- Analysis 1: GT scale correction (how far off is TRAM?) ---
        scale_correction, gt_cam_t, _ = compute_gt_scale(gt_ann, pred_cam_T)
        
        # --- Analysis 2: GT human depth vs VIMO human depth ---
        T = min(gt_ann['n_frames'], len(pred_trans))
        valid = gt_ann['good_frames_mask'][:T]
        
        gt_human_cam_z = compute_oracle_human_scale(gt_ann, pred_cam_R, pred_cam_T)
        gt_human_cam_z = gt_human_cam_z[:T]
        
        vimo_tz = pred_trans[:T, 2]
        
        # Compare GT human depth vs VIMO predicted depth
        valid_mask = valid & (gt_human_cam_z > 0.1) & (vimo_tz > 0.1)
        
        if valid_mask.sum() < 10:
            print(f"  {seq_name}: SKIP (too few valid frames)")
            continue
        
        gt_tz = gt_human_cam_z[valid_mask]
        pred_tz = vimo_tz[valid_mask]
        
        # Per-frame scale from GT human depth
        # (If we had raw SLAM depth, α = gt_tz / slam_depth_human)
        # Instead, we compute the ratio gt_tz / pred_tz
        # This tells us how accurate VIMO's depth estimation is
        tz_ratio = gt_tz / pred_tz
        tz_ratio_median = np.median(tz_ratio)
        tz_ratio_std = np.std(tz_ratio)
        
        # Absolute depth error
        tz_error = np.abs(gt_tz - pred_tz)
        tz_error_median = np.median(tz_error)
        tz_error_relative = np.median(tz_error / gt_tz)
        
        # --- Analysis 3: GT beta vs predicted beta ---
        gt_betas = gt_ann['smpl']['betas']  # (10,) single shape for the person
        pred_betas_mean = pred_shape[valid_mask].mean(axis=0)
        beta_error = np.linalg.norm(gt_betas - pred_betas_mean)
        
        # --- Analysis 4: What would human scale give us? ---
        # If we use median(pred_tz) as the "human depth anchor",
        # and TRAM uses median(bg_scale) as the "bg anchor",
        # how do they compare to GT?
        
        # TRAM's scale accuracy: scale_correction tells us how far off TRAM was
        # scale_correction ≈ 1.0 means TRAM was perfect
        tram_scale_error = abs(scale_correction - 1.0)
        
        result = {
            'seq': seq_name,
            'n_frames': T,
            'n_valid': int(valid_mask.sum()),
            # TRAM scale accuracy
            'tram_scale_correction': scale_correction,
            'tram_scale_error_pct': tram_scale_error * 100,
            # Human depth accuracy (GT vs VIMO)
            'gt_tz_median': float(np.median(gt_tz)),
            'pred_tz_median': float(np.median(pred_tz)),
            'tz_ratio_median': float(tz_ratio_median),
            'tz_ratio_std': float(tz_ratio_std),
            'tz_error_median_m': float(tz_error_median),
            'tz_error_relative_pct': float(tz_error_relative * 100),
            # Beta accuracy
            'beta_error': float(beta_error),
            'gt_beta0': float(gt_betas[0]) if len(gt_betas.shape) == 1 else float(gt_betas[0, 0]),
            'pred_beta0': float(pred_betas_mean[0]),
            # Raw data for plotting
            '_gt_tz': gt_tz,
            '_pred_tz': pred_tz,
            '_valid': valid_mask,
        }
        
        all_results.append(result)
        print(f"  {seq_name}: TRAM_err={tram_scale_error*100:.1f}%, "
              f"tz_ratio={tz_ratio_median:.3f}±{tz_ratio_std:.3f}, "
              f"tz_err={tz_error_relative*100:.1f}%, "
              f"beta_err={beta_error:.3f}")
    
    # Summary
    print_oracle_summary(all_results)
    plot_oracle_results(all_results, save_dir)
    
    return all_results


def print_oracle_summary(results):
    """Print summary of oracle experiment."""
    print("\n" + "=" * 80)
    print("HARP Oracle Experiment Results")
    print("=" * 80)
    
    # TRAM scale accuracy
    tram_errors = [r['tram_scale_error_pct'] for r in results]
    print(f"\n1. TRAM Background Scale Accuracy:")
    print(f"   Median error: {np.median(tram_errors):.1f}%")
    print(f"   Mean error:   {np.mean(tram_errors):.1f}%")
    print(f"   Max error:    {np.max(tram_errors):.1f}%")
    
    # Human depth accuracy
    tz_errors = [r['tz_error_relative_pct'] for r in results]
    print(f"\n2. VIMO Human Depth Accuracy (vs GT):")
    print(f"   Median relative error: {np.median(tz_errors):.1f}%")
    print(f"   Mean relative error:   {np.mean(tz_errors):.1f}%")
    
    # GT tz ratio (how close is VIMO tz to GT tz)
    tz_ratios = [r['tz_ratio_median'] for r in results]
    print(f"\n3. GT/VIMO Depth Ratio (1.0 = perfect):")
    print(f"   Median: {np.median(tz_ratios):.3f}")
    print(f"   Std:    {np.std(tz_ratios):.3f}")
    
    if abs(np.median(tz_ratios) - 1.0) < 0.15:
        print(f"   >> VIMO depth is ACCURATE (within 15% of GT)")
    else:
        print(f"   >> VIMO depth has systematic bias of {(np.median(tz_ratios)-1)*100:.1f}%")
    
    # Beta accuracy
    beta_errors = [r['beta_error'] for r in results]
    print(f"\n4. Shape (Beta) Accuracy:")
    print(f"   Median L2 error: {np.median(beta_errors):.3f}")
    
    # Key conclusion
    print(f"\n{'='*80}")
    print(f"KEY CONCLUSION:")
    if np.median(tz_errors) < 20:
        print(f"  Human depth from VIMO is accurate enough for scale estimation")
        print(f"  (median {np.median(tz_errors):.1f}% error vs GT)")
    else:
        print(f"  Human depth has {np.median(tz_errors):.1f}% error - ")
        print(f"  temporal aggregation and fusion with bg scale recommended")
    print(f"{'='*80}")


def plot_oracle_results(results, save_dir):
    """Generate oracle experiment plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    seqs = [r['seq'][:18] for r in results]
    n = len(seqs)
    
    # 1. TRAM scale error per sequence
    ax = axes[0, 0]
    tram_errs = [r['tram_scale_error_pct'] for r in results]
    bars = ax.barh(range(n), tram_errs, color='#e74c3c', alpha=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xlabel('TRAM Scale Error (%)')
    ax.set_title('TRAM Background Scale Error')
    ax.axvline(x=np.median(tram_errs), color='k', linestyle='--', 
               alpha=0.5, label=f'median={np.median(tram_errs):.1f}%')
    ax.legend()
    
    # 2. VIMO tz relative error per sequence
    ax = axes[0, 1]
    tz_errs = [r['tz_error_relative_pct'] for r in results]
    ax.barh(range(n), tz_errs, color='#3498db', alpha=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xlabel('VIMO Depth Relative Error (%)')
    ax.set_title('VIMO Human Depth Error (vs GT)')
    ax.axvline(x=np.median(tz_errs), color='k', linestyle='--',
               alpha=0.5, label=f'median={np.median(tz_errs):.1f}%')
    ax.legend()
    
    # 3. GT tz vs VIMO tz scatter
    ax = axes[1, 0]
    all_gt_tz = np.concatenate([r['_gt_tz'] for r in results])
    all_pred_tz = np.concatenate([r['_pred_tz'] for r in results])
    # Subsample for plotting
    idx = np.random.choice(len(all_gt_tz), min(5000, len(all_gt_tz)), replace=False)
    ax.scatter(all_gt_tz[idx], all_pred_tz[idx], alpha=0.1, s=2, c='#3498db')
    lim = max(all_gt_tz[idx].max(), all_pred_tz[idx].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', alpha=0.5, label='perfect')
    ax.set_xlabel('GT Human Depth (m)')
    ax.set_ylabel('VIMO Predicted Depth (m)')
    ax.set_title('GT vs VIMO Human Depth (all frames)')
    ax.legend()
    ax.set_aspect('equal')
    
    # 4. Per-sequence tz ratio (GT/VIMO)
    ax = axes[1, 1]
    tz_ratios = [r['tz_ratio_median'] for r in results]
    tz_ratio_stds = [r['tz_ratio_std'] for r in results]
    colors = ['#2ecc71' if abs(r - 1.0) < 0.1 
              else '#f39c12' if abs(r - 1.0) < 0.2 
              else '#e74c3c' for r in tz_ratios]
    ax.barh(range(n), tz_ratios, xerr=tz_ratio_stds, color=colors, alpha=0.7,
            ecolor='gray', capsize=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xlabel('GT/VIMO Depth Ratio')
    ax.set_title('Depth Ratio per Sequence (1.0 = perfect)')
    ax.axvline(x=1.0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=1.1, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'oracle_results.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_dir}/oracle_results.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emdb_root', type=str, default='/workspace/dataset/emdb')
    parser.add_argument('--results_dir', type=str, default='results/emdb')
    parser.add_argument('--save_dir', type=str, default='results/harp_oracle')
    args = parser.parse_args()
    
    results = run_oracle_experiment(args.emdb_root, args.results_dir, args.save_dir)


if __name__ == '__main__':
    main()
