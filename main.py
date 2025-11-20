# sam2 video segmentation test

import argparse
import subprocess
import pickle
import os
import cv2
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rp import *
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

# visualize generated mask for frame, the mask should be white on a black background
def show_mask(mask, ax, obj_id=None, random_color=False):
    mask_color = np.array([255, 255, 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    print(f"pos_points: {pos_points}, neg_points: {neg_points}")
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# transforms points from original resolution to new transformed resolution
# necessary since gwtf transformed videos to a different resolution.
def transform_image(orig_image, centroid):
    image = orig_image.copy()
    orig_h, orig_w = orig_image.shape[:2]
    image_fit = resize_image_to_fit(image, height=1440, allow_growth=False)
    h_fit, w_fit = image_fit.shape[:2]

    SCALE_FACTOR = 1
    HEIGHT = 480 * SCALE_FACTOR
    WIDTH = 720 * SCALE_FACTOR

    image_hold = resize_image_to_hold(image_fit, height=HEIGHT, width=WIDTH)
    h_hold, w_hold = image_hold.shape[:2]

    image_final = crop_image(image_hold, height=HEIGHT, width=WIDTH, origin="center")
    h_final, w_final = image_final.shape[:2]  # expected HEIGHT, WIDTH

    # Compute image transformation parameters
    # orig ---> fit : scale_fit = (w_fit/orig_w, h_fit/orig_h)
    # fit ---> hold : scale_hold = (w_hold/w_fit, h_hold/h_fit)
    # hold -> final : center-crop with offset (offset_x, offset_y)
    scale_fit_x = w_fit / float(orig_w)
    scale_fit_y = h_fit / float(orig_h)
    scale_hold_x = w_hold / float(w_fit) if w_fit != 0 else 1.0
    scale_hold_y = h_hold / float(h_fit) if h_fit != 0 else 1.0

    total_scale_x = scale_fit_x * scale_hold_x
    total_scale_y = scale_fit_y * scale_hold_y

    offset_x = (w_hold - w_final) / 2.0
    offset_y = (h_hold - h_final) / 2.0

    def map_points_from_orig_to_final(pts):
        pts = np.asarray(pts, dtype=np.float32).copy()
        pts[:, 0] = pts[:, 0] * total_scale_x - offset_x
        pts[:, 1] = pts[:, 1] * total_scale_y - offset_y
        pts[:, 0] = np.clip(pts[:, 0], 0, w_final - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_final - 1)
        return pts
    
    return map_points_from_orig_to_final(centroid)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def generate_segmentations(video_dir, output_dir, point):
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

    #point = np.array([[440, 636]])
    label = np.array([1])  # positive point

    ann_frame_idx = 0 # the frame where the input point is being registered
    ann_obj_id = 1 # an identifier for the 'selected' object.
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=point,
        labels=label, # positive point
    )

    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(point, label, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    plt.savefig(f"{output_dir}/frame_{ann_frame_idx:0{6}d}_with_input_point.jpg")

    # video_segments = {}  # video_segments contains the per-frame segmentation results
    # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    #     video_segments[out_frame_idx] = {
    #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #         for i, out_obj_id in enumerate(out_obj_ids)
    #     }

    # for out_frame_idx in range(0, len(frame_names)):
    #     plt.figure(frameon=False)
    #     plt.background('black')
    #     plt.axes([0., 0., 1., 1.])
    #     plt.axis('off')
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id);
    #     plt.savefig(f"{output_dir}/{out_frame_idx: 0{6}d}.jpg")

def generate_video_frames(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    outdir = f'tmp/{name}'
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', video_path, '-qscale:v', '2', f'{outdir}/%04d.jpg'])

    return outdir

def find_initial_point(waypoint_path):
    # load pickle file
    with open(waypoint_path, 'rb') as f:
        data = pickle.load(f)

    # access first element of 'centroids' list in pickle file.
    # TODO: doesn't seem to have saved rotation data in waypoint file?
    print(data["centroids"])
    centroid = data["centroids"][0]

    return np.array(centroid) # first frame, first centroid for now.

def main():
    argparser = argparse.ArgumentParser(description="SAM2 Video Segmentation Test")
    argparser.add_argument("--sam2_path", type=str, help="Path to the installed sam2 codebase", default="/home/ganidhu/sam2")
    argparser.add_argument("--video_path", type=str, help="Path to the .mp4 video file", required=True)
    argparser.add_argument("--data_path", type=str, help="Path of the dataset file", required=True)
    argparser.add_argument("--output_dir", type=str, help="Directory to save segmented frames", default="output")

    args = argparser.parse_args()

    # TODO
    #initialize_sam2(args.sam2_path)

    video_path = args.video_path
    
    vid_dir = generate_video_frames(video_path)

    waypoint_path = os.path.join(args.data_path, 'waypoints.pkl')
    image_path = os.path.join(args.data_path, 'rgba_00000.jpg')
    point = find_initial_point(waypoint_path    )

    # transform centroid point based on image transformation
    orig_image = cv2.imread(image_path)
    point = transform_image(orig_image, point)

    os.makedirs(args.output_dir, exist_ok=True)
    generate_segmentations(vid_dir, args.output_dir, point)

if __name__ == "__main__":
    main()