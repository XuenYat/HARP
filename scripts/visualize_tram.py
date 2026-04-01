import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
from glob import glob
from lib.pipeline import visualize_tram

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='input video')
parser.add_argument('--method', type=str, default=None, choices=['tram', 'harp'],
                    help='Which results to visualize (default: auto-detect)')
parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size')
parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor')
args = parser.parse_args()

# File and folders
file = args.video
seq = os.path.basename(file).split('.')[0]
seq_folder = f'results/{seq}'

# Determine method
method = args.method
if method is None:
    # Auto-detect: prefer harp > tram > legacy
    if os.path.exists(f'{seq_folder}/camera_harp.npy'):
        method = 'harp'
    elif os.path.exists(f'{seq_folder}/camera_tram.npy'):
        method = 'tram'
    elif os.path.exists(f'{seq_folder}/camera.npy'):
        method = None  # Legacy
    else:
        print(f"Error: No camera results found in {seq_folder}/")
        sys.exit(1)

# Setup paths
if method:
    camera_file = f'{seq_folder}/camera_{method}.npy'
    output_video = f'{seq_folder}/output_{method}.mp4'
    print(f'Visualizing {method.upper()} results')
else:
    camera_file = f'{seq_folder}/camera.npy'
    output_video = f'{seq_folder}/tram_output.mp4'
    print('Visualizing legacy results')

# Check camera file exists
if not os.path.exists(camera_file):
    print(f"Error: {camera_file} not found")
    sys.exit(1)

##### Visualize #####
print('Rendering...')
visualize_tram(
    seq_folder,
    camera_file=camera_file,
    output_video=output_video,
    floor_scale=args.floor_scale,
    bin_size=args.bin_size
)