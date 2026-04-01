"""
HARP Scale Fusion Experiment (Experiment 3.1)
=============================================
After running the modified SLAM pipeline that saves intermediates,
this script:
1. Loads raw SLAM depths + VIMO pred_trans
2. Computes α_human per keyframe
3. Fuses α_bg and α_human with confidence weighting
4. Evaluates all scale variants against GT

Usage:
    python harp_fusion_experiment.py \
        --emdb_root /workspace/dataset/emdb \
        --results_dir /workspace/HARP/results/emdb \
        --intermediates_dir /workspace/HARP/results/emdb/intermediates \
        --save_dir /workspace/HARP/results/harp_fusion
"""

import os
import sys
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.camera.harp_scale import (
    est_scale_human, 
    est_scale_human_temporal,
    est_scale_harp,
    compute_bg_confidence,
    compute_human_confidence,
    weighted_median
)


def load_intermediates(intermediates_dir, seq_name):
    """Load HARP intermediate variables for a sequence."""
    meta_file = os.path.join(intermediates_dir, f'{seq_name}.npz')
    depth_dir = os.path.join(intermediates_dir, f'{seq_name}_depths')
    
    if not os.path.exists(meta_file):
        return None
    
    meta = dict(np.load(meta_file, allow_pickle=True))
    
    tstamps = meta['tstamps']
    scales_bg = meta['scales_bg']
    scale_bg_final = float(meta['scale_bg_final'])
    
    # Load per-keyframe depth maps
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


def compute_human_correction(intermediates, smpl_data):
    """
    Compute per-keyframe scale estimates from human region.
    
    Key insight: SLAM depth at human region is invalid (masked during SLAM).
    But we can derive a scale estimate through ZoeDepth as bridge:
    
    α_bg = ZoeDepth_bg / SLAM_bg   (TRAM's per-keyframe estimate, background only)
    
    For human region, VIMO gives us metric tz, and ZoeDepth gives predicted depth.
    Their ratio tells us ZoeDepth's accuracy at the human location.
    So: α_human_i = α_bg_i * (VIMO_tz / ZoeDepth_human)
    
    We then pool ALL estimates (α_bg from all keyframes + α_human from all keyframes)
    and take a single robust median. This naturally increases observation diversity.
    
    Returns:
        human_scales: per-keyframe α_human estimates
        all_scales: combined bg + human scales for robust median
        scale_combined: final robust median of all scales
    """
    tstamps = intermediates['tstamps']
    pred_depths = intermediates['pred_depths']
    scales_bg = intermediates['scales_bg']
    scale_bg_final = float(intermediates['scale_bg_final'])
    
    pred_trans = smpl_data['pred_trans'].reshape(-1, 3)
    img_focal = float(smpl_data['img_focal'])
    img_center = smpl_data['img_center']
    
    human_scales = []
    
    for i, t in enumerate(tstamps):
        if t >= len(pred_trans) or i >= len(pred_depths):
            continue
        
        tx, ty, tz = pred_trans[t]
        if tz <= 0.1:
            continue
        
        pred_depth = pred_depths[i]
        pd_H, pd_W = pred_depth.shape
        
        # Project human center to ZoeDepth resolution
        u = img_focal * tx / tz + img_center[0]
        v = img_focal * ty / tz + img_center[1]
        orig_W = img_center[0] * 2
        orig_H = img_center[1] * 2
        
        u_pd = int(u * pd_W / orig_W)
        v_pd = int(v * pd_H / orig_H)
        
        radius = max(int(pd_H * 0.08), 15)
        v_min = max(0, v_pd - radius)
        v_max = min(pd_H, v_pd + radius)
        u_min = max(0, u_pd - radius)
        u_max = min(pd_W, u_pd + radius)
        
        if v_max <= v_min or u_max <= u_min:
            continue
        
        patch = pred_depth[v_min:v_max, u_min:u_max]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        
        if len(valid) < 20:
            continue
        
        zoe_z = np.median(valid)
        if zoe_z <= 0:
            continue
        
        # α_human_i = α_bg_i * (VIMO_tz / ZoeDepth_human)
        correction = tz / zoe_z
        alpha_human_i = scales_bg[i] * correction
        human_scales.append(alpha_human_i)
    
    human_scales = np.array(human_scales) if human_scales else np.array([])
    
    # Pool all scale estimates together and take robust median
    all_scales = np.concatenate([scales_bg, human_scales])
    scale_combined = np.median(all_scales)
    
    return human_scales, all_scales, scale_combined


def evaluate_scale(gt_ann, pred_cam_R, pred_cam_T_unscaled, scale, label=""):
    """
    Evaluate a specific scale by computing camera trajectory error (ATE).
    
    Args:
        gt_ann: EMDB GT annotation
        pred_cam_R: (T, 3, 3) predicted camera rotation
        pred_cam_T_unscaled: (T, 3) unscaled camera translation (SLAM output / original_scale)
        scale: the metric scale to evaluate
        label: name for printing
    
    Returns:
        ate: Absolute Trajectory Error (meters)
    """
    from lib.utils.eval_utils import align_pcl
    import torch
    
    # GT camera trajectory
    ext = gt_ann['camera']['extrinsics']
    gt_cam_r = ext[:, :3, :3].transpose(0, 2, 1)
    gt_cam_t = np.einsum('bij,bj->bi', gt_cam_r, -ext[:, :3, -1])
    
    # Predicted trajectory with this scale
    pred_cam_t = pred_cam_T_unscaled * scale
    
    T = min(len(gt_cam_t), len(pred_cam_t))
    gt_traj = torch.from_numpy(gt_cam_t[:T]).float()
    pred_traj = torch.from_numpy(pred_cam_t[:T]).float()
    
    # ATE without scale correction (tests our scale accuracy)
    s, R, t = align_pcl(gt_traj.unsqueeze(0), pred_traj.unsqueeze(0), fixed_scale=True)
    pred_aligned = (torch.einsum("tij,tnj->tni", R, pred_traj.unsqueeze(0)) + t.unsqueeze(1))[0]
    ate = torch.norm(gt_traj - pred_aligned, dim=-1).mean().item()
    
    # Also compute scale accuracy
    s_free, _, _ = align_pcl(gt_traj.unsqueeze(0), pred_traj.unsqueeze(0), fixed_scale=False)
    scale_error = abs(s_free.item() - 1.0)
    
    return ate, scale_error


def run_fusion_experiment(emdb_root, results_dir, intermediates_dir, save_dir,
                          bias_correction=1.0):
    """
    Run the full fusion experiment.
    
    Compare four scale strategies:
    1. α_bg only (TRAM baseline)
    2. α_human only 
    3. Simple average: (α_bg + α_human) / 2
    4. HARP confidence-weighted fusion
    """
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
    
    all_results = []
    
    for seq_path, gt_ann, seq_name in emdb2_seqs:
        # Load TRAM predictions
        smpl_file = os.path.join(results_dir, 'smpl', f'{seq_name}.npz')
        cam_file = os.path.join(results_dir, 'camera', f'{seq_name}.npz')
        
        if not os.path.exists(smpl_file) or not os.path.exists(cam_file):
            print(f"  {seq_name}: SKIP (no TRAM results)")
            continue
        
        smpl_data = dict(np.load(smpl_file, allow_pickle=True))
        cam_data = dict(np.load(cam_file, allow_pickle=True))
        
        # Load intermediates
        intermediates = load_intermediates(intermediates_dir, seq_name)
        if intermediates is None:
            print(f"  {seq_name}: SKIP (no intermediates - run modified SLAM first)")
            continue
        
        # Compute human-derived scale estimates and pool with bg estimates
        human_scales, all_scales, scale_combined = \
            compute_human_correction(intermediates, smpl_data)
        
        if len(human_scales) == 0:
            print(f"  {seq_name}: SKIP (cannot compute human scales)")
            continue
        
        # Get α_bg
        scale_bg = intermediates['scale_bg_final']
        scales_bg = intermediates['scales_bg']
        
        # Three strategies:
        # 1. TRAM baseline: median of α_bg only
        # 2. Human only: median of α_human only
        # 3. HARP: median of (α_bg + α_human) pooled together
        scale_human = float(np.median(human_scales))
        scale_harp = scale_combined
        scale_avg = (scale_bg + scale_human) / 2
        
        # Get unscaled camera translation
        # pred_cam_T = unscaled_T * scale_bg, so unscaled_T = pred_cam_T / scale_bg
        pred_cam_T = cam_data['pred_cam_T']
        pred_cam_R = cam_data['pred_cam_R']
        unscaled_T = pred_cam_T / scale_bg
        
        # Evaluate each strategy
        ate_bg, serr_bg = evaluate_scale(gt_ann, pred_cam_R, unscaled_T, scale_bg, "BG")
        ate_human, serr_human = evaluate_scale(gt_ann, pred_cam_R, unscaled_T, scale_human, "HUMAN")
        ate_avg, serr_avg = evaluate_scale(gt_ann, pred_cam_R, unscaled_T, scale_avg, "AVG")
        ate_harp, serr_harp = evaluate_scale(gt_ann, pred_cam_R, unscaled_T, scale_harp, "HARP")
        
        result = {
            'seq': seq_name,
            'scale_bg': scale_bg,
            'scale_human': scale_human,
            'scale_avg': scale_avg,
            'scale_harp': scale_harp,
            'ate_bg': ate_bg,
            'ate_human': ate_human,
            'ate_avg': ate_avg,
            'ate_harp': ate_harp,
            'scale_err_bg': serr_bg,
            'scale_err_human': serr_human,
            'scale_err_avg': serr_avg,
            'scale_err_harp': serr_harp,
        }
        
        all_results.append(result)
        print(f"  {seq_name}: ATE bg={ate_bg:.3f} human={ate_human:.3f} "
              f"avg={ate_avg:.3f} harp={ate_harp:.3f} "
              f"[scales: {len(scales_bg)} bg + {len(human_scales)} human = {len(all_scales)} total]")
    
    # Summary
    print_fusion_summary(all_results)
    plot_fusion_results(all_results, save_dir)
    
    return all_results


def print_fusion_summary(results):
    """Print fusion experiment summary."""
    print("\n" + "=" * 80)
    print("HARP Fusion Experiment Results")
    print("=" * 80)
    
    methods = ['bg', 'human', 'avg', 'harp']
    labels = ['TRAM (α_bg)', 'Human (α_human)', 'Simple Avg', 'HARP (ours)']
    
    print(f"\n{'Method':<25} {'ATE (m) ↓':>12} {'Scale Err ↓':>12}")
    print("-" * 50)
    
    for method, label in zip(methods, labels):
        ates = [r[f'ate_{method}'] for r in results]
        serrs = [r[f'scale_err_{method}'] for r in results]
        print(f"{label:<25} {np.mean(ates):>10.3f}m {np.mean(serrs)*100:>10.1f}%")
    
    # Per-category analysis
    print(f"\n--- Breakdown by difficulty ---")
    
    # Find sequences where TRAM struggles (scale_err > 15%)
    hard_seqs = [r for r in results if r['scale_err_bg'] > 0.15]
    easy_seqs = [r for r in results if r['scale_err_bg'] <= 0.15]
    
    if hard_seqs:
        print(f"\nHard sequences (TRAM scale err > 15%, n={len(hard_seqs)}):")
        for method, label in zip(methods, labels):
            ates = [r[f'ate_{method}'] for r in hard_seqs]
            print(f"  {label:<25} {np.mean(ates):.3f}m")
    
    if easy_seqs:
        print(f"\nEasy sequences (TRAM scale err <= 15%, n={len(easy_seqs)}):")
        for method, label in zip(methods, labels):
            ates = [r[f'ate_{method}'] for r in easy_seqs]
            print(f"  {label:<25} {np.mean(ates):.3f}m")


def plot_fusion_results(results, save_dir):
    """Plot fusion comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    seqs = [r['seq'][:18] for r in results]
    n = len(seqs)
    x = np.arange(n)
    w = 0.2
    
    # ATE comparison
    ax = axes[0]
    ate_bg = [r['ate_bg'] for r in results]
    ate_human = [r['ate_human'] for r in results]
    ate_avg = [r['ate_avg'] for r in results]
    ate_harp = [r['ate_harp'] for r in results]
    
    ax.barh(x - 1.5*w, ate_bg, w, label='TRAM (α_bg)', color='#e74c3c', alpha=0.8)
    ax.barh(x - 0.5*w, ate_human, w, label='Human (α_human)', color='#3498db', alpha=0.8)
    ax.barh(x + 0.5*w, ate_avg, w, label='Simple Avg', color='#f39c12', alpha=0.8)
    ax.barh(x + 1.5*w, ate_harp, w, label='HARP (ours)', color='#2ecc71', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xlabel('ATE (meters, lower is better)')
    ax.set_title('Camera Trajectory Error (ATE-S)')
    ax.legend(fontsize=8)
    
    # Scale error comparison
    ax = axes[1]
    serr_bg = [r['scale_err_bg'] * 100 for r in results]
    serr_human = [r['scale_err_human'] * 100 for r in results]
    serr_harp = [r['scale_err_harp'] * 100 for r in results]
    
    ax.barh(x - w, serr_bg, w, label='TRAM (α_bg)', color='#e74c3c', alpha=0.8)
    ax.barh(x, serr_human, w, label='Human (α_human)', color='#3498db', alpha=0.8)
    ax.barh(x + w, serr_harp, w, label='HARP (ours)', color='#2ecc71', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xlabel('Scale Error (%)')
    ax.set_title('Scale Estimation Error')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fusion_results.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_dir}/fusion_results.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emdb_root', type=str, default='/workspace/dataset/emdb')
    parser.add_argument('--results_dir', type=str, default='/workspace/HARP/results/emdb')
    parser.add_argument('--intermediates_dir', type=str, 
                        default='/workspace/HARP/results/emdb/intermediates')
    parser.add_argument('--save_dir', type=str, default='/workspace/HARP/results/harp_fusion')
    parser.add_argument('--bias_correction', type=float, default=1.0,
                        help='Bias correction for human scale (from oracle experiment)')
    args = parser.parse_args()
    
    results = run_fusion_experiment(
        args.emdb_root, args.results_dir, args.intermediates_dir, args.save_dir,
        bias_correction=args.bias_correction
    )


if __name__ == '__main__':
    main()