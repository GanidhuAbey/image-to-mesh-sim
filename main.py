# sam2 video segmentation test

import argparse
import subprocess
import pickle
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# model location
sam_loc = '/Users/ganidhu/sam2'
sam2_checkpoint = f'{sam_loc}/checkpoints/sam2.1_hiera_small.pt'
model_cfg = f'configs/sam2.1/sam2.1_hiera_s.yaml'

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def generate_segmentations(video_dir, point):
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    # frame_idx = 0
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0 # the frame where the input point is being registered
    ann_obj_id = 1 # an identifier for the 'selected' object.
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=point[np.newaxis, :, :],
        labels=[0],
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for out_frame_idx in range(0, len(frame_names)):
        plt.figure(frameon=False)
        plt.axes([0., 0., 1., 1.])
        plt.axis('off')
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id);
        plt.savefig(f"{output_dir}/{out_frame_idx: 0{6}d}.jpg")

def generate_video_frames(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    outdir = f'tmp/{name}'
    subprocess.run(['ffmpeg', '-i', video_path, '-qscale:v', '2', f'{outdir}/%04d.jpg'])

    return outdir

def find_initial_point(waypoint_path):
    # load pickle file
    with open(waypoint_path, 'rb') as f:
        data = pickle.load(f)

    # access first element of 'centroids' list in pickle file.
    centroids = data['centroids']

    return centroids[0][0] # first frame, first centroid for now.

def main():
    argparser = argparse.ArgumentParser(description="SAM2 Video Segmentation Test")
    argparser.add_argument("--video_path", type=str, help="Path to the .mp4 video file", required=True)
    argparser.add_argument("--waypoint_path", type=str, help="Path of the generated waypoint data for video generation model", required=True)
    argparser.add_argument("--output_dir", type=str, help="Directory to save segmented frames", default="output")

    args = argparser.parse_args()

    video_path = args.video_path
    
    outdir = generate_video_frames(video_path)
    point = find_initial_point(args.waypoint_path)

    generate_segmentations(outdir, point)
