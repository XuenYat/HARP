"""
Qualitative visualization: Global trajectory comparison (TRAM vs HARP)
Plots xy-plane trajectories like TRAM Figure 5.
"""
import sys
sys.path.insert(0, '/workspace/HARP')

import numpy as np
import pickle as pkl
from glob import glob
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.utils.eval_utils import align_pcl
from lib.utils.rotation_conversions import *
from lib.models.smpl import SMPL

smpls = {g: SMPL(gender=g) for g in ['neutral', 'male', 'female']}
tt = lambda x: torch.from_numpy(x).float()

emdb_root = '/workspace/dataset/emdb'
tram_dir = '/workspace/HARP/results/emdb'
int_dir = '/workspace/HARP/results/emdb_harp/intermediates'

def compute_harp_scale(meta, smpl_data):
    tstamps = meta['tstamps']
    scales_bg = meta['scales_bg']
    pred_trans = smpl_data['pred_trans'].reshape(-1, 3)
    img_focal = float(smpl_data['img_focal'])
    img_center = smpl_data['img_center']
    human_scales = []
    for i, t in enumerate(tstamps):
        if t >= len(pred_trans):
            continue
        tx, ty, tz = pred_trans[t]
        if tz <= 0.1:
            continue
        kf_file = f'{int_dir}/{meta["_seq"]}_depths/keyframe_{i:04d}.npz'
        try:
            kf = dict(np.load(kf_file))
        except:
            continue
        pd = kf['pred_depth']
        pd_H, pd_W = pd.shape
        u = img_focal * tx / tz + img_center[0]
        v = img_focal * ty / tz + img_center[1]
        orig_W, orig_H = img_center[0]*2, img_center[1]*2
        u_pd, v_pd = int(u*pd_W/orig_W), int(v*pd_H/orig_H)
        radius = max(int(pd_H * 0.08), 15)
        v_min, v_max = max(0,v_pd-radius), min(pd_H,v_pd+radius)
        u_min, u_max = max(0,u_pd-radius), min(pd_W,u_pd+radius)
        if v_max<=v_min or u_max<=u_min:
            continue
        patch = pd[v_min:v_max, u_min:u_max]
        valid = patch[(patch>0) & np.isfinite(patch)]
        if len(valid)<10:
            continue
        zoe_z = np.median(valid)
        if zoe_z > 0:
            human_scales.append(scales_bg[i] * tz / zoe_z)
    human_scales = np.array(human_scales) if human_scales else np.array([])
    all_scales = np.concatenate([scales_bg, human_scales])
    return np.median(all_scales)

# Select representative sequences
selected = [
    ('P3', '27_indoor_walk_off_mvs'),
    ('P3', '30_outdoor_stairs_down'),
    ('P4', '35_indoor_walk'),
    ('P6', '49_outdoor_big_stairs_down'),
    ('P8', '65_outdoor_walk_straight'),
    ('P3', '29_outdoor_stairs_up'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (parent, seq) in enumerate(selected):
    ax = axes[idx]
    
    # Load GT
    pkl_file = f'{emdb_root}/{parent}/{seq}/{parent}_{seq}_data.pkl'
    ann = pkl.load(open(pkl_file, 'rb'))
    ext = ann['camera']['extrinsics']
    valid = ann['good_frames_mask']
    gender = ann['gender']
    
    # GT human trajectory in world frame
    gt_trans = ann['smpl']['trans']
    poses_body = ann['smpl']['poses_body']
    poses_root = ann['smpl']['poses_root']
    betas = np.repeat(ann['smpl']['betas'].reshape(1,-1), ann['n_frames'], axis=0)
    
    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root),
                       betas=tt(betas), transl=tt(gt_trans),
                       pose2rot=True, default_smpl=True)
    gt_j3d = gt.joints[:, :24]
    
    # TRAM prediction
    smpl_data = dict(np.load(f'{tram_dir}/smpl/{seq}.npz', allow_pickle=True))
    cam_data = dict(np.load(f'{tram_dir}/camera/{seq}.npz'))
    
    pred_rotmat = torch.tensor(smpl_data['pred_rotmat'])
    pred_shape = torch.tensor(smpl_data['pred_shape'])
    pred_trans_vimo = torch.tensor(smpl_data['pred_trans'])
    mean_shape = pred_shape.mean(0, keepdim=True).repeat(len(pred_shape), 1)
    
    pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0]],
                            betas=mean_shape, transl=pred_trans_vimo.squeeze(),
                            pose2rot=False, default_smpl=True)
    pred_j3d = pred.joints[:, :24]
    
    pred_camr = torch.tensor(cam_data['pred_cam_R'])
    pred_camt_tram = torch.tensor(cam_data['pred_cam_T'])
    
    # TRAM world trajectory
    T = min(len(gt_j3d), len(pred_j3d))
    pred_j3d_tram = torch.einsum('bij,bnj->bni', pred_camr[:T], pred_j3d[:T]) + pred_camt_tram[:T, None]
    
    # HARP prediction (rescaled)
    meta = dict(np.load(f'{int_dir}/{seq}.npz'))
    meta['_seq'] = seq
    harp_scale = compute_harp_scale(meta, smpl_data)
    scale_bg = float(meta['scale_bg_final'])
    unscaled = pred_camt_tram / scale_bg
    pred_camt_harp = unscaled * harp_scale
    pred_j3d_harp = torch.einsum('bij,bnj->bni', pred_camr[:T], pred_j3d[:T]) + pred_camt_harp[:T, None]
    
    # Align predictions to GT (first 2 frames)
    from lib.utils.eval_utils import first_align_joints
    gt_valid = gt_j3d[:T][valid[:T]]
    tram_valid = pred_j3d_tram[valid[:T]]
    harp_valid = pred_j3d_harp[valid[:T]]
    
    tram_aligned = first_align_joints(gt_valid, tram_valid)
    harp_aligned = first_align_joints(gt_valid, harp_valid)
    
    # Plot xy trajectory (root joint)
    gt_xy = gt_valid[:, 0, [0, 2]].numpy()
    tram_xy = tram_aligned[:, 0, [0, 2]].numpy()
    harp_xy = harp_aligned[:, 0, [0, 2]].numpy()
    
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'gray', linewidth=2, alpha=0.5, label='GT')
    ax.plot(tram_xy[:, 0], tram_xy[:, 1], '#e74c3c', linewidth=1.5, alpha=0.8, label='TRAM')
    ax.plot(harp_xy[:, 0], harp_xy[:, 1], '#2ecc71', linewidth=1.5, alpha=0.8, label='HARP')
    
    # Compute RTE for title
    from lib.utils.eval_utils import compute_rte
    rte_tram = compute_rte(gt_valid[:,0], tram_valid[:,0]).mean() * 100
    rte_harp = compute_rte(gt_valid[:,0], harp_valid[:,0]).mean() * 100
    
    ax.set_title(f'{seq}\nTRAM RTE={rte_tram:.1f}% → HARP RTE={rte_harp:.1f}%', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/HARP/results/trajectory_comparison.png', dpi=150)
plt.close()
print("Saved to /workspace/HARP/results/trajectory_comparison.png")