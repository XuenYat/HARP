"""
HARP: Run modified SLAM on EMDB subset 2
=========================================
This is a modified version of TRAM's eval_emdb_cam.py that additionally 
saves intermediate variables (raw SLAM depth, ZoeDepth predictions, per-frame scales).

Usage:
    python harp_run_emdb_slam.py \
        --emdb_root /workspace/dataset/emdb \
        --output_dir /workspace/HARP/results/emdb/camera \
        --intermediates_dir /workspace/HARP/results/emdb/intermediates
        
Estimated time: ~15-30 min per sequence on RTX 3090 (25 sequences total)
"""

import sys
import os

# Add project root and thirdparty paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'thirdparty', 'Tracking-Anything-with-DEVA'))
sys.path.insert(0, os.path.join(project_root, 'thirdparty', 'DROID-SLAM', 'droid_slam'))
sys.path.insert(0, os.path.join(project_root, 'thirdparty', 'DROID-SLAM'))

import cv2
import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm

from torch.amp import autocast
from segment_anything import SamPredictor, sam_model_registry
from detectron2.config import LazyConfig

from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from lib.camera import align_cam_to_world, run_metric_slam_harp

parser = argparse.ArgumentParser()
parser.add_argument('--emdb_root', type=str, default='/workspace/dataset/emdb')
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--output_dir', type=str, default='/workspace/HARP/results/emdb/camera')
parser.add_argument('--intermediates_dir', type=str, 
                    default='/workspace/HARP/results/emdb/intermediates')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = 'cuda'

# Find EMDB2 sequences
emdb_root = args.emdb_root
emdb = []
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
        if ann.get(f'emdb{args.split}', False):
            emdb.append((seq_path, ann))

print(f"Found {len(emdb)} EMDB{args.split} sequences")

# Save folders
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.intermediates_dir, exist_ok=True)

# ViTDet detector
cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)

# SAM
sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
_ = sam.to(device)
predictor = SamPredictor(sam)


# Process each sequence
for seq_path, ann in emdb:
    seq_name = os.path.basename(seq_path)
    print(f'\n{"="*60}')
    print(f'Processing: {seq_name}')
    print(f'{"="*60}')
    
    # Check if already done
    cam_save = f'{args.output_dir}/{seq_name}.npz'
    int_save = f'{args.intermediates_dir}/{seq_name}.npz'
    if os.path.exists(cam_save) and os.path.exists(int_save):
        print(f'  Already processed, skipping.')
        continue
    
    img_folder = f'{seq_path}/images'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    
    # Detection + SAM segmentation
    masks_ = []
    for t, imgpath in enumerate(tqdm(imgfiles, desc="Detection+SAM")):
        img_cv2 = cv2.imread(imgpath)
        
        with torch.no_grad():
            with autocast('cuda'):
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()
                boxes = np.hstack([boxes, confs[:, None]])
                boxes = arrange_boxes(boxes, mode='size', min_size=100)
        
        if len(boxes) > 0:
            with autocast('cuda'):
                predictor.set_image(img_cv2, image_format='BGR')
                bb = torch.tensor(boxes[:, :4]).cuda()
                bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=bb, multimask_output=False
                )
                masks = masks.cpu().squeeze(1)
                mask = masks.sum(dim=0)
        else:
            if len(masks_) > 0:
                mask = torch.zeros_like(masks_[-1])
            else:
                H, W = img_cv2.shape[:2]
                mask = torch.zeros(H, W)
        
        masks_.append(mask.byte())
    
    masks = torch.stack(masks_)
    
    # Camera intrinsics from GT
    intr = ann['camera']['intrinsics']
    cam_int = [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]
    
    # Run HARP-modified SLAM (saves intermediates)
    cam_R, cam_T = run_metric_slam_harp(
        img_folder, masks=masks, calib=cam_int,
        save_intermediates=int_save
    )
    
    wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
    
    # Save camera results (same format as TRAM)
    camera = {
        'pred_cam_R': cam_R.numpy(),
        'pred_cam_T': cam_T.numpy(),
        'world_cam_R': wd_cam_R.numpy(),
        'world_cam_T': wd_cam_T.numpy(),
        'img_focal': cam_int[0],
        'img_center': cam_int[2:],
        'spec_focal': spec_f
    }
    np.savez(cam_save, **camera)
    print(f'  Saved: {cam_save}')
    print(f'  Intermediates: {int_save}')

print(f'\nDone! All results saved to {args.output_dir}')
print(f'Intermediates saved to {args.intermediates_dir}')
print(f'\nNext step: run harp_fusion_experiment.py')