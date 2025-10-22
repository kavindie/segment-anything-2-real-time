
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import display

# Doing Grounded-Segment-Anything detector
import sys
sys.path.append("/scratch3/kat049/Grounded-Segment-Anything/GroundingDINO")
sys.path.append("/scratch3/kat049/Grounded-Segment-Anything/segment_anything")
sys.path.append("/scratch3/kat049/concept-graphs")
sys.path.append("/scratch3/kat049/Grounded-Segment-Anything/recognize-anything")

try:
    from groundingdino import _C
except ImportError:
    print("GroundingDINO C++ extension not built. Please run `python setup.py build_ext --inplace` in the GroundingDINO directory.")
    raise

import argparse
from pathlib import Path
import re
from typing import Any, List
import json
import imageio
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import gzip
import open_clip

from ultralytics import YOLO
import torchvision
from torch.utils.data import Dataset
import supervision as sv
from tqdm import trange

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from conceptgraph.utils.model_utils import compute_clip_features
import torch.nn.functional as F

from conceptgraph.scripts.generate_gsa_results import get_sam_predictor, process_tag_classes, get_sam_mask_generator, get_sam_segmentation_from_xyxy, get_sam_segmentation_dense

import torchvision.transforms as TS
from ram.models import ram
from ram.models import tag2text
from ram import inference_tag2text, inference_ram

from groundingdino.util.inference import Model

if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")

import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)
# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

device_number = 2
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")


# TODO use bfloat16 for the entire notebook - this gives an error in segement anything, maybe I can change the code to use bfloat16
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))

# Load video
start_idx = 1148
vis_gap = 30

video_dir = "/scratch3/kat049/datasets/DARPA/p14_fr/results" #args.video_path
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(p[len("frame"):].split('.')[0]))
frame_names = frame_names[start_idx:-1:vis_gap]


frame = Image.open(os.path.join(video_dir, frame_names[0])) #image_pil
frame_RGB = np.array(frame) #image_rgb
frame_BGR = frame_RGB[:, :, ::-1]  # image
width, height = frame.size[0], frame.size[1]

# SAM2 + streaming
# Load SAM camera predictor (tracking)
from sam2.build_sam import build_sam2_camera_predictor
sam2_checkpoint = "/scratch3/kat049/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint).to(device)
predictor.load_first_frame(frame, device)
print("Loaded SAM2")

# SAM in concept graph
# TODO following params need to be passed in as args
sam_variant = "sam"
detector = "yolo"
class_set = "none"
save_video = True
add_bg_classes = True
accumu_classes = True
box_threshold = 0.25
text_threshold = 0.25
nms_threshold = 0.5
detector = "yolo"


# grounding_dino_model = Model(
#         model_config_path=GROUNDING_DINO_CONFIG_PATH, 
#         model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
#         device=device
#     )

if class_set == "none":
    mask_generator = get_sam_mask_generator(sam_variant, device)
else:
    sam_predictor = get_sam_predictor(sam_variant, device)


clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

global_classes = set()

yolo_model_w_classes = YOLO('/scratch3/kat049/concept-graphs/yolov8l-world.pt')  # or choose yolov8m/l-world.pt


if class_set == "none":
    classes = ['item']
    print("Skipping tagging and detection models. ")
elif class_set == "ram":
    tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,image_size=384,vit='swin_l')
    tagging_model = tagging_model.eval().to(device)
    tagging_transform = TS.Compose([
                TS.Resize((384, 384)),
                TS.ToTensor(), 
                TS.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
    classes = None
    print(f"{class_set} will be used to detect classes. ")
else:
    raise ValueError("Unknown args.class_set: ", class_set)

save_name = f"{class_set}"
save_name += f"_{sam_variant}"
# if save_video:
#     video_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}.mp4"
#     frames = []

color_path = str(Path(os.path.join(video_dir, frame_names[0])))
# image = cv2.imread(color_path) # This will in BGR color space
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
# image_pil = Image.fromarray(image_rgb)


if class_set == "ram":
    raw_image = frame.resize((384, 384))
    raw_image = tagging_transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image , tagging_model)
    caption="NA"

    text_prompt=res[0].replace(' |', ',')

    with open('/scratch3/kat049/concept-graphs/conceptgraph/ram_classes_4500.txt', 'r') as file:
        lines = [line.strip() for line in file]
    # lines = []
    add_classes = lines + ["tunnel entrance",  "yellow 4-legged robot", "orange tank-like robot", "orange drill", "red_backpack",  "wall lamp", "entrance" "wall lamp"]
    
    remove_classes = [
        "room", "kitchen", "office", "house", "home", "building", "corner",
        "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
        "apartment", "image", "city", "skylight", "hallway", "bureau", "modern", "salon", "doorway"
    ]

    bg_classes = ["wall", "floor", "ceiling"]

    if add_bg_classes:
        add_classes += bg_classes
    else:
        remove_classes += bg_classes

    classes = process_tag_classes(
                    text_prompt, 
                    add_classes = add_classes,
                    remove_classes = remove_classes,
                )
    
    # add classes to global classes
global_classes.update(classes)

if accumu_classes:
    # Use all the classes that have been seen so far
    classes = list(global_classes)

print("Loaded SAM and YPOLOv8 models. ")

if class_set == "none":
    mask, xyxy, conf = get_sam_segmentation_dense(
                sam_variant, mask_generator, frame_RGB)
    detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
    image_crops, image_feats, text_feats = compute_clip_features(
                frame_RGB, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)
    annotated_image, labels = vis_result_fast(
                frame_BGR, detections, classes, instance_random_color=True)
    plt.imshow(annotated_image)
    plt.savefig("test.png")
    plt.close()
    # cv2.imwrite(vis_save_path, annotated_image)
else:
    if detector == "yolo":
        yolo_model_w_classes.set_classes(classes)
        yolo_results_w_classes = yolo_model_w_classes.predict(color_path)
        yolo_results_w_classes[0].save("/scratch3/kat049/sam2/test_YOLO.png")
        xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
        xyxy_np = xyxy_tensor.cpu().numpy()
        confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()

        detections = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
            mask=None,
        )

    if len(detections.class_id) > 0:
        ### Segment Anything ###
        # sam_predictor.model = sam_predictor.model.float()
        detections.mask = get_sam_segmentation_from_xyxy(
            sam_predictor=sam_predictor,
            image=frame_RGB,
            xyxy=detections.xyxy
        )

        # Compute and save the clip features of detections  
        image_crops, image_feats, text_feats = compute_clip_features(
            frame_RGB, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)
    else:
        image_crops, image_feats, text_feats = [], [], []

    ### Visualize results ###
    annotated_image, labels = vis_result_fast(frame_BGR, detections, classes)
    plt.imshow(annotated_image)
    plt.savefig("test.png")
    plt.close()
    # plt.imshow(annotated_image)
    # cv2.imwrite(vis_save_path, annotated_image)

# if save_video:
#     frames.append(annotated_image)
    

# Convert the detections to a dict. The elements are in np.array
results = {
    "xyxy": detections.xyxy,
    "confidence": detections.confidence,
    "class_id": detections.class_id,
    "mask": detections.mask,
    "classes": classes,
    "image_crops": image_crops,
    "image_feats": image_feats,
    "text_feats": text_feats,
}

if class_set == "ram":
    results["tagging_caption"] = caption
    results["tagging_text_prompt"] = text_prompt


