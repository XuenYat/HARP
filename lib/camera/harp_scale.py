"""
HARP: Human Anchor for Robust Positioning
Core module: Human geometry-based scale estimation and confidence-aware fusion.

This module provides:
1. est_scale_human(): Compute metric scale from SMPL translation (human depth anchor)
2. est_scale_harp(): Confidence-aware fusion of background scale and human scale
3. analyze_scale_sources(): Diagnostic tool comparing both scale sources against GT
"""

import numpy as np
import torch
from scipy import stats


def est_scale_human(slam_depth, pred_trans, masks, method='median'):
    """
    Estimate metric scale using human body as anchor.
    
    Core insight: VIMO outputs pred_trans in metric units (meters, from SMPL),
    while SLAM depth is in arbitrary units. Their ratio gives the scale.
    
    Args:
        slam_depth: (H, W) SLAM disparity-inverted depth for a keyframe (arbitrary scale)
        pred_trans: (1, 3) or (3,) VIMO predicted translation [tx, ty, tz] in meters
        masks: (H, W) human mask (1=human, 0=background)
        method: 'median' or 'robust' aggregation within the mask
    
    Returns:
        scale: float, the estimated metric scale α_human
        confidence: float, confidence of this estimate (0-1)
    """
    trans = pred_trans.reshape(-1)
    tz_metric = trans[2]  # human depth in meters (from SMPL)
    
    if tz_metric <= 0.1:
        # Human too close or invalid prediction
        return None, 0.0
    
    # Get SLAM depth in human region
    if masks is not None:
        import cv2
        msk = cv2.resize(masks.astype(np.float32), 
                         (slam_depth.shape[1], slam_depth.shape[0]))
        human_region = (msk > 0.5) & (slam_depth > 0)
    else:
        return None, 0.0
    
    if human_region.sum() < 100:
        # Too few pixels in human region
        return None, 0.0
    
    slam_depth_human = slam_depth[human_region]
    
    if method == 'median':
        slam_z_human = np.median(slam_depth_human)
    elif method == 'robust':
        # Use trimmed mean (remove top/bottom 10%)
        slam_z_human = stats.trim_mean(slam_depth_human, 0.1)
    else:
        slam_z_human = np.mean(slam_depth_human)
    
    if slam_z_human <= 0:
        return None, 0.0
    
    scale = tz_metric / slam_z_human
    
    # Confidence based on:
    # 1. Number of valid pixels (more = better)
    # 2. Variance of SLAM depth in human region (lower = more consistent)
    # 3. Human depth range (too far = less reliable)
    n_pixels = human_region.sum()
    depth_cv = np.std(slam_depth_human) / (np.mean(slam_depth_human) + 1e-8)
    
    conf_pixels = min(n_pixels / 5000.0, 1.0)
    conf_depth_var = np.exp(-depth_cv)  # lower CV = higher confidence
    conf_distance = np.exp(-max(tz_metric - 10.0, 0.0) / 5.0)  # penalize far humans
    
    confidence = conf_pixels * conf_depth_var * conf_distance
    
    return scale, confidence


def est_scale_human_temporal(slam_depths, pred_trans_seq, masks_seq, tstamps=None):
    """
    Temporally robust human scale estimation across a video sequence.
    
    Args:
        slam_depths: list of (H, W) SLAM depths for keyframes
        pred_trans_seq: (T, 1, 3) or (T, 3) VIMO translations for all frames
        masks_seq: list/tensor of (H, W) human masks
        tstamps: keyframe indices into the full sequence
    
    Returns:
        scale: float, robust median scale
        scales_per_frame: list of per-frame scales (for analysis)
        confidences: list of per-frame confidences
    """
    n = len(slam_depths)
    scales = []
    confidences = []
    
    for i in range(n):
        # Map keyframe index to full sequence index
        t = tstamps[i] if tstamps is not None else i
        
        if t >= len(pred_trans_seq):
            continue
            
        trans = pred_trans_seq[t]
        msk = masks_seq[t] if masks_seq is not None else None
        
        s, c = est_scale_human(slam_depths[i], trans, msk)
        
        if s is not None:
            scales.append(s)
            confidences.append(c)
    
    if len(scales) == 0:
        return None, [], []
    
    # Weighted median using confidences
    scale = weighted_median(np.array(scales), np.array(confidences))
    
    return scale, scales, confidences


def est_scale_harp(scale_bg, scale_human, var_bg, var_human):
    """
    HARP: Uncertainty-aware fusion of background and human scale estimates
    via inverse-variance weighting (Maximum Likelihood Estimation).

    Assuming both estimates are independent Gaussian observations of the
    true scale α*:
        α_bg    ~ N(α*, σ²_bg)
        α_human ~ N(α*, σ²_human)

    The MLE is the inverse-variance weighted mean:
        α* = (α_bg/σ²_bg + α_human/σ²_human) / (1/σ²_bg + 1/σ²_human)

    Args:
        scale_bg: float, background-based scale (from ZoeDepth)
        scale_human: float, human geometry-based scale
        var_bg: float, measurement variance of background scale (σ²_bg)
        var_human: float, measurement variance of human scale (σ²_human)

    Returns:
        scale_fused: float, the fused metric scale
        source: str, which source dominated ('bg', 'human', 'fused')
        w_human: float, weight assigned to human scale (0-1)
    """
    if scale_human is None or np.isinf(var_human):
        return scale_bg, 'bg', 0.0

    if scale_bg is None or np.isinf(var_bg):
        return scale_human, 'human', 1.0

    # Clamp variances to avoid division by zero
    var_bg = max(var_bg, 1e-8)
    var_human = max(var_human, 1e-8)

    # Inverse-variance weighting (MLE under Gaussian assumption)
    w_human = (1.0 / var_human) / (1.0 / var_human + 1.0 / var_bg)
    w_bg = 1.0 - w_human

    scale_fused = w_bg * scale_bg + w_human * scale_human

    # Determine dominant source
    if w_human > 0.7:
        source = 'human'
    elif w_bg > 0.7:
        source = 'bg'
    else:
        source = 'fused'

    return scale_fused, source, w_human


def compute_bg_confidence(scales_bg_per_frame, slam_depths, pred_depths, masks_seq):
    """
    Estimate confidence of background-based scale.
    
    Args:
        scales_bg_per_frame: list of per-keyframe background scales
        slam_depths: list of SLAM depths
        pred_depths: list of ZoeDepth predictions
        masks_seq: list of human masks
    
    Returns:
        confidence: float (0-1)
    """
    if len(scales_bg_per_frame) < 3:
        return 0.3
    
    arr = np.array(scales_bg_per_frame)
    
    # 1. Scale consistency: low variance = high confidence
    cv = np.std(arr) / (np.median(arr) + 1e-8)
    conf_consistency = np.exp(-cv * 2)
    
    # 2. Background texture richness (averaged over frames)
    conf_texture = 0.5  # default; compute from image gradients if available
    if slam_depths and masks_seq is not None:
        texture_scores = []
        for i, (sd, msk) in enumerate(zip(slam_depths, masks_seq)):
            bg_region = (msk < 0.5) if msk is not None else np.ones_like(sd, dtype=bool)
            # Use depth variance as proxy for texture (structured scene = varied depth)
            if bg_region.sum() > 100:
                bg_depth = sd[bg_region & (sd > 0)]
                if len(bg_depth) > 100:
                    depth_range = np.percentile(bg_depth, 90) - np.percentile(bg_depth, 10)
                    texture_scores.append(min(depth_range / 5.0, 1.0))
        if texture_scores:
            conf_texture = np.mean(texture_scores)
    
    # 3. Number of valid keyframes
    conf_nframes = min(len(scales_bg_per_frame) / 20.0, 1.0)
    
    confidence = conf_consistency * conf_texture * conf_nframes
    return float(np.clip(confidence, 0, 1))


def compute_human_confidence(pred_trans_seq, pred_shape_seq):
    """
    Estimate confidence of human-based scale.
    
    Args:
        pred_trans_seq: (T, 1, 3) VIMO translations
        pred_shape_seq: (T, 10) VIMO shape (beta) parameters
    
    Returns:
        confidence: float (0-1)
    """
    trans = pred_trans_seq.reshape(-1, 3)
    
    # 1. Shape consistency: beta should be stable across frames
    beta_std = np.std(pred_shape_seq, axis=0).mean()
    conf_shape = np.exp(-beta_std * 5)
    
    # 2. Translation smoothness: tz should vary smoothly
    tz = trans[:, 2]
    if len(tz) > 2:
        tz_accel = np.abs(np.diff(tz, n=2))
        conf_smooth = np.exp(-np.median(tz_accel) * 10)
    else:
        conf_smooth = 0.5
    
    # 3. Reasonable depth range (human should be 1-20m from camera)
    tz_median = np.median(tz)
    if 1.0 < tz_median < 15.0:
        conf_range = 1.0
    elif 0.5 < tz_median < 20.0:
        conf_range = 0.5
    else:
        conf_range = 0.1
    
    confidence = conf_shape * conf_smooth * conf_range
    return float(np.clip(confidence, 0, 1))


# ============================================================
# Diagnostic / Analysis Tools
# ============================================================

def analyze_scale_sources(results_dir, gt_data, seq_name):
    """
    Compare background scale vs human scale against ground truth.
    This is the key diagnostic experiment for the HARP paper.
    
    Args:
        results_dir: path to TRAM results (containing camera/ and smpl/)
        gt_data: dict from EMDB GT pkl file
        seq_name: sequence name
    
    Returns:
        analysis: dict with detailed comparison
    """
    import os
    
    # Load TRAM predictions
    smpl_path = os.path.join(results_dir, 'smpl', f'{seq_name}.npz')
    cam_path = os.path.join(results_dir, 'camera', f'{seq_name}.npz')
    
    smpl_data = dict(np.load(smpl_path, allow_pickle=True))
    cam_data = dict(np.load(cam_path, allow_pickle=True))
    
    pred_trans = smpl_data['pred_trans']       # (T, 1, 3) metric translation
    pred_shape = smpl_data['pred_shape']       # (T, 10) beta parameters
    pred_cam_T = cam_data['pred_cam_T']        # (T, 3) camera translation (already scaled)
    pred_cam_R = cam_data['pred_cam_R']        # (T, 3, 3) camera rotation
    
    # GT camera trajectory
    ext = gt_data['camera']['extrinsics']      # (T, 4, 4) R_cw, t_cw
    gt_cam_r = ext[:, :3, :3].transpose(0, 2, 1)  # R_wc
    gt_cam_t = np.einsum('bij,bj->bi', gt_cam_r, -ext[:, :3, -1])  # t_wc
    
    # GT scale: ratio between GT trajectory and SLAM trajectory
    # (We can compute this from the camera translations)
    gt_displacement = np.linalg.norm(np.diff(gt_cam_t, axis=0), axis=1).sum()
    pred_displacement = np.linalg.norm(np.diff(pred_cam_T, axis=0), axis=1).sum()
    
    # Human-based scale analysis
    tz_values = pred_trans.reshape(-1, 3)[:, 2]  # metric depth of human
    
    # Shape analysis
    beta_mean = pred_shape.mean(axis=0)
    beta_std = pred_shape.std(axis=0)
    
    # Estimate human height from mean beta
    # (This requires SMPL forward pass, simplified here)
    
    analysis = {
        'seq_name': seq_name,
        'n_frames': len(pred_trans),
        'gt_total_displacement': gt_displacement,
        'pred_total_displacement': pred_displacement,
        'human_tz_median': float(np.median(tz_values)),
        'human_tz_std': float(np.std(tz_values)),
        'human_tz_min': float(np.min(tz_values)),
        'human_tz_max': float(np.max(tz_values)),
        'beta_mean': beta_mean,
        'beta_std_mean': float(beta_std.mean()),
        'pred_cam_T_range': float(np.linalg.norm(
            pred_cam_T.max(axis=0) - pred_cam_T.min(axis=0)
        )),
    }
    
    return analysis


def analyze_all_sequences(results_dir, emdb_root, split=2):
    """
    Run analysis across all EMDB sequences.
    
    Args:
        results_dir: path to TRAM results
        emdb_root: path to EMDB dataset root
        split: 1 or 2
    """
    import pickle as pkl
    from glob import glob
    
    all_analyses = []
    
    roots = []
    for p in range(10):
        folder = f'{emdb_root}/P{p}'
        if os.path.exists(folder):
            root = sorted(glob(f'{folder}/*'))
            roots.extend(root)
    
    for root in roots:
        ann_file = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        if not os.path.exists(ann_file):
            continue
        ann = pkl.load(open(ann_file, 'rb'))
        if not ann[f'emdb{split}']:
            continue
        
        seq = root.split('/')[-1]
        try:
            analysis = analyze_scale_sources(results_dir, ann, seq)
            all_analyses.append(analysis)
            print(f"  {seq}: tz_median={analysis['human_tz_median']:.2f}m, "
                  f"beta_std={analysis['beta_std_mean']:.4f}")
        except Exception as e:
            print(f"  {seq}: FAILED - {e}")
    
    return all_analyses


# ============================================================
# Utility Functions
# ============================================================

def weighted_median(values, weights):
    """Compute weighted median."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    cumw = np.cumsum(sorted_weights)
    cutoff = cumw[-1] / 2.0
    
    idx = np.searchsorted(cumw, cutoff)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_height_from_beta(beta, smpl_model=None):
    """
    Compute human height in meters from SMPL beta parameters.
    
    Args:
        beta: (10,) shape parameters
        smpl_model: SMPL model instance (if None, uses approximate linear model)
    
    Returns:
        height: float in meters
    """
    if smpl_model is not None:
        # Full SMPL forward pass
        beta_t = torch.from_numpy(beta).float().unsqueeze(0)
        output = smpl_model(betas=beta_t)
        verts = output.vertices[0].detach().numpy()
        height = verts[:, 1].max() - verts[:, 1].min()
        return height
    else:
        # Approximate: mean height + linear correction from first beta component
        # Average human height ~1.7m, beta[0] roughly controls height
        return 1.7 + 0.1 * beta[0]