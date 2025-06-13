import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import os
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import requests
import re

# My imports
import sys
sys.path.extend([
    '/scratch3/kat049/segment-anything-2-real-time',
    '/scratch3/kat049/AutoSeg-SAM2/submodule/segment-anything-1',
    '/scratch3/kat049/AutoSeg-SAM2'
])
from my_scripts.utils_vis import get_masked_image
from my_scripts.utils_vlms import answerwAria
from my_scripts.utils_graph import get_relative_distance2obj, load_data_files, find_closest_pose, calculate_motion_between_frames, create_mixed_graph, visualize_mixed_graph
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from auto_mask_batch import masks_update, Prompts

class MultiObjectSAM2CameraPredictor:
    def __init__(self, model_cfg, sam2_checkpoint, device="cuda"):
        """
        Initialize multi-object SAM2 camera predictor
        
        Args:
            model_cfg: Path to model configuration file
            sam2_checkpoint: Path to SAM2 checkpoint
            device: Device to run inference on
        """
        self.predictor = build_sam2_camera_predictor(
            model_cfg, 
            sam2_checkpoint, 
            device=device
        )
        self.is_initialized = False
        self.tracked_objects = {}  # Store object IDs and their info
        
    def load_first_frame(self, frame):
        """
        Load the first frame for tracking initialization
        
        Args:
            frame: First frame (BGR format)
        """
        self.predictor.load_first_frame(frame)
        self.is_initialized = True
        
    def add_new_object(self, prompt, prompt_type="point", object_ID=None):
        """
        Add a new object to track
        
        Args:
            prompt: Prompt for the object (point coordinates, bounding box, etc.)
            prompt_type: Type of prompt ("point", "box", "mask")
            object_ID: Optional ID for the object (if None, SAM2 will assign one)
            
        Returns:
            object_id: ID of the newly added object
        """
        if not self.is_initialized:
            raise ValueError("Must call load_first_frame() before adding objects")
            
        # Format prompt based on type
        if prompt_type == "point":
            # prompt should be (x, y) coordinates
            point_coords = np.array([prompt], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)  # 1 for foreground
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,  # Current frame
                obj_id=object_ID,  # Let SAM2 assign ID
                points=point_coords,
                labels=point_labels,
            )
            
        elif prompt_type == "bbox":
            # prompt should be (x1, y1, x2, y2) bounding box
            bbox = np.array([prompt], dtype=np.float32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=object_ID,
                bbox=bbox,
            )
            
        elif prompt_type == "mask":
            # prompt should be a binary mask
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=object_ID,
                mask=prompt,
            )
            
        # Store the new object info
        if len(out_obj_ids) > 0:
            new_obj_id = out_obj_ids[-1]  # Get the newly added object ID
            self.tracked_objects[new_obj_id] = {
                'prompt': prompt,
                'prompt_type': prompt_type,
                'active': True
            }
            return new_obj_id, out_mask_logits
            
        return None, None
    
    def add_multiple_objects(self, prompts_list):
        """
        Add multiple objects at once
        
        Args:
            prompts_list: List of tuples (prompt, prompt_type, object_ID)
            
        Returns:
            List of object IDs
        """
        object_ids = []
        for prompt, prompt_type, object_ID in prompts_list:
            obj_id, _ = self.add_new_object(prompt, prompt_type, object_ID)
            if obj_id is not None:
                object_ids.append(obj_id)
        return object_ids
    
    def track_frame(self, frame):
        """
        Track all objects in the current frame
        
        Args:
            frame: Current frame (BGR format)
            
        Returns:
            out_obj_ids: List of tracked object IDs
            out_mask_logits: Corresponding mask logits for each object
        """
        if not self.is_initialized:
            raise ValueError("Must call load_first_frame() before tracking")
            
        return self.predictor.track(frame)
    
    def remove_object(self, obj_id):
        """
        Remove an object from tracking
        
        Args:
            obj_id: ID of object to remove
        """
        if obj_id in self.tracked_objects:
            # SAM2 doesn't have explicit remove, so we mark as inactive
            self.tracked_objects[obj_id]['active'] = False
            
    def get_active_objects(self):
        """
        Get list of currently active object IDs
        
        Returns:
            List of active object IDs
        """
        return [obj_id for obj_id, info in self.tracked_objects.items() 
                if info['active']]
    
    def reset_tracker(self):
        """
        Reset the tracker state
        """
        self.predictor.reset_state()
        self.tracked_objects.clear()
        self.is_initialized = False

def create_seq_graph(video_dir, frame_names, mask_generator, prompts_loader, tracker):
    first_frame = frame_names[0]
    frame = Image.open(os.path.join(video_dir, first_frame)) #image_pil
    frame_RGB = np.array(frame) #image_rgb
    frame_BGR = frame_RGB[:, :, ::-1]  # image
    width, height = frame.size[0], frame.size[1]

    # generate masks for the first frame
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(frame_RGB)
    masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    masks = [mask for mask in masks_l]
    other_masks = [mask for mask in masks_s] + [mask for mask in masks_m]
    ann_obj_id_list = range(len(masks))
    for ann_obj_id in tqdm(ann_obj_id_list):
        seg = masks[ann_obj_id]['segmentation']
        prompts_loader.add(ann_obj_id,0,seg)
    bboxes = [masks[i]['bbox'] for i in range(len(masks))]
    # Initialize video capture
    # cap = cv2.VideoCapture("/scratch3/kat049/datasets/DARPA/p14_fr/camera0-1024x768-002.mp4")

    ##Create initialze graph
    G_temporal = []
    connect_Gs = []
    captions = []
    image_nodes = [get_masked_image(frame_RGB, masks[i]['segmentation']) for i in range(len(masks))]
    mean_depths = [get_relative_distance2obj(video_dir, first_frame, masks[i]['segmentation']) for i in range(len(masks))]
    G = create_mixed_graph(image_nodes, mean_depths)
    G_temporal.append(G)

    # Aria
    captions_t = []
    for mask in masks:
        masked_rgb_cropped = get_masked_image(frame_RGB, mask['segmentation'])
        question = """Describe what is shown in the image"""
        caption = answerwAria(question, masked_rgb_cropped)
        captions_t.append(caption)
        print(caption)
    captions.append(captions_t)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Load first frame
        tracker.load_first_frame(frame_BGR)
        
        # Add multiple objects to track
        # You can add objects with different prompt types
        # object_prompts = [
        # #     ((100, 100), "point"),     # Point at (100, 100)
        # #     ((200, 150), "point"),     # Another point
        # #     ((50, 50, 150, 150), "bbox"),  # Bounding box (x1, y1, x2, y2)
        #     ((620, 540, 980, 800), "bbox", 0),  # Bounding box (x1, y1, x2, y2)
        #     ((160, 510, 220, 580), "bbox", 1)  # Bounding box (x1, y1, x2, y2)
        # ]
        object_prompts = []
        for obj_id, bbox in zip(ann_obj_id_list, bboxes):
            object_prompts.append(((bbox[0], bbox[1], bbox[2], bbox[3]), "bbox", obj_id))

        object_ids = tracker.add_multiple_objects(object_prompts)
        print(f"Added objects with IDs: {object_ids}")
        
        # fig, axes  = plt.subplots(6, 5, figsize=(16, 12))
        plots = 0
        # ax_iter = iter(axes.flat)

        # Main tracking loop
        for i, f in enumerate(frame_names[1:]):
            plots += 1
            if plots > 30:
                break
            frame = Image.open(os.path.join(video_dir, f)) #image_pil
            frame_RGB = np.array(frame) #image_rgb
            frame_BGR = frame_RGB[:, :, ::-1]  # image

            out_obj_ids, out_mask_logits = tracker.track_frame(frame_BGR)
            
            # Prepare figure
            # ax = next(ax_iter)
            # display_img = frame_RGB.copy()

            rel_motion = np.square(calculate_motion_between_frames(file_prev, f, rgb_df, pose_df)['translation']).sum() ** 0.5
            file_prev = f
            if rel_motion > 0.05: # moved more than 5 cm
                connect_Gs.append(rel_motion)
            else:
                connect_Gs.append(0.0)

            image_nodes = []
            mean_depths = []
            # Process and visualize results
            if len(out_obj_ids) > 0:
                # Convert masks to numpy arrays and overlay on frame
                title = ""
                for i, (obj_id, mask_logit) in enumerate(zip(out_obj_ids, out_mask_logits)):
                    # Convert logits to binary mask
                    mask = (mask_logit.squeeze() > 0).cpu().numpy()  # Shape: (768, 1024)
                    image_nodes.append(get_masked_image(frame_RGB, mask))

                    mean_depth = get_relative_distance2obj(video_dir, f, mask)
                    mean_depths.append(mean_depth)

                    masked_rgb_cropped = get_masked_image(frame_RGB, mask)
                    if masked_rgb_cropped is None:
                        print("No mask found, skipping frame")
                        continue
                    # Create colored overlay for this object
                    # color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                    #         (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
                    
                    # # Apply mask overlay
                    # colored_mask = np.zeros_like(display_img)
                    # colored_mask[mask > 0] = color
                    # alpha = 0.3
                    # display_img = ((1 - alpha) * display_img + alpha * colored_mask).astype(np.uint8)
                    
                    # title += f"{mean_depth:.2f}m, {rel_motion:.2f}m, ID: {obj_id}\n"
                # ax.set_title(title.strip())
            # Display frame
            # ax.imshow(display_img)
            # ax.axis('off')
            
            G = create_mixed_graph(image_nodes, mean_depths)
            G_temporal.append(G)


            # # Break on 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
            # # Example: Add new object on 'a' key press
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('a'):
            #     # Add new object at center of frame
            #     h, w = frame_BGR.shape[:2]
            #     new_obj_id, _ = tracker.add_new_object((w//2, h//2), "point")
            #     if new_obj_id:
            #         print(f"Added new object with ID: {new_obj_id}")
    
        # plt.savefig('test.png')
        # plt.close(fig)


def example_multi_object_tracking():
    """
    Example of how to use the multi-object tracker
    """
    # Initialize the predictor
    sam2_checkpoint = "/scratch3/kat049/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tracker = MultiObjectSAM2CameraPredictor(model_cfg, sam2_checkpoint, device)

    # Initialize the sam to segment
    sam_ckpt_path="/scratch3/kat049/AutoSeg-SAM2/checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,)
    sum_id = 0
    prompts_loader = Prompts(bs=1)
    sum_id = prompts_loader.get_obj_num()

    # intialize the video
    start_idx = 1148
    vis_gap = 30

    video_dir = "/scratch3/kat049/datasets/DARPA/p14_fr/results" #args.video_path
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(p[len("frame"):].split('.')[0]))
    frame_names = frame_names[start_idx:-1:vis_gap]
    file_prev = frame_names[0]

    # initialize the depth
    rgb_df, pose_df = load_data_files(
        "/scratch3/kat049/VSLAM-LAB-Benchmark/DARPA/p14_fr_2/rgb.txt",
        "/scratch3/kat049/VSLAM-LAB-Benchmark/DARPA/p14_fr_2/groundtruth.txt"
    )

    start_frame = Image.open(os.path.join(video_dir, frame_names[0])) #image_pil
    frame_RGB = np.array(frame) #image_rgb
    frame_BGR = frame_RGB[:, :, ::-1]  # image
    width, height = frame.size[0], frame.size[1]

    # generate masks for the first frame
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(frame_RGB)
    masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    masks = [mask for mask in masks_l]
    other_masks = [mask for mask in masks_s] + [mask for mask in masks_m]
    ann_obj_id_list = range(len(masks))
    for ann_obj_id in tqdm(ann_obj_id_list):
        seg = masks[ann_obj_id]['segmentation']
        prompts_loader.add(ann_obj_id,0,seg)
    bboxes = [masks[i]['bbox'] for i in range(len(masks))]
    # Initialize video capture
    # cap = cv2.VideoCapture("/scratch3/kat049/datasets/DARPA/p14_fr/camera0-1024x768-002.mp4")


    ##Create initialze graph
    G_temporal = []
    connect_Gs = []
    image_nodes = [get_masked_image(frame_RGB, masks[i]['segmentation']) for i in range(len(masks))]
    mean_depths = [get_relative_distance2obj(video_dir, frame_names[0], masks[i]['segmentation']) for i in range(len(masks))]
    G = create_mixed_graph(image_nodes, mean_depths)
    G_temporal.append(G)

    # Aria
    masked_rgb_cropped = get_masked_image(frame_RGB, masks[10]['segmentation'])
    question = """Describe what is shown in the image"""
    caption = answerwAria(question, masked_rgb_cropped)
    print(caption)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Load first frame
        tracker.load_first_frame(frame_BGR)
        
        # Add multiple objects to track
        # You can add objects with different prompt types
        # object_prompts = [
        # #     ((100, 100), "point"),     # Point at (100, 100)
        # #     ((200, 150), "point"),     # Another point
        # #     ((50, 50, 150, 150), "bbox"),  # Bounding box (x1, y1, x2, y2)
        #     ((620, 540, 980, 800), "bbox", 0),  # Bounding box (x1, y1, x2, y2)
        #     ((160, 510, 220, 580), "bbox", 1)  # Bounding box (x1, y1, x2, y2)
        # ]
        object_prompts = []
        for obj_id, bbox in zip(ann_obj_id_list, bboxes):
            object_prompts.append(((bbox[0], bbox[1], bbox[2], bbox[3]), "bbox", obj_id))

        object_ids = tracker.add_multiple_objects(object_prompts)
        print(f"Added objects with IDs: {object_ids}")
        
        # fig, axes  = plt.subplots(6, 5, figsize=(16, 12))
        plots = 0
        # ax_iter = iter(axes.flat)

        # Main tracking loop
        for i, f in enumerate(frame_names[1:]):
            plots += 1
            if plots > 30:
                break
            frame = Image.open(os.path.join(video_dir, f)) #image_pil
            frame_RGB = np.array(frame) #image_rgb
            frame_BGR = frame_RGB[:, :, ::-1]  # image

            out_obj_ids, out_mask_logits = tracker.track_frame(frame_BGR)
            
            # Prepare figure
            # ax = next(ax_iter)
            # display_img = frame_RGB.copy()

            rel_motion = np.square(calculate_motion_between_frames(file_prev, f, rgb_df, pose_df)['translation']).sum() ** 0.5
            file_prev = f
            if rel_motion > 0.05: # moved more than 5 cm
                connect_Gs.append(rel_motion)
            else:
                connect_Gs.append(0.0)

            image_nodes = []
            mean_depths = []
            # Process and visualize results
            if len(out_obj_ids) > 0:
                # Convert masks to numpy arrays and overlay on frame
                title = ""
                for i, (obj_id, mask_logit) in enumerate(zip(out_obj_ids, out_mask_logits)):
                    # Convert logits to binary mask
                    mask = (mask_logit.squeeze() > 0).cpu().numpy()  # Shape: (768, 1024)
                    image_nodes.append(get_masked_image(frame_RGB, mask))

                    mean_depth = get_relative_distance2obj(video_dir, f, mask)
                    mean_depths.append(mean_depth)

                    masked_rgb_cropped = get_masked_image(frame_RGB, mask)
                    if masked_rgb_cropped is None:
                        print("No mask found, skipping frame")
                        continue
                    # Create colored overlay for this object
                    # color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                    #         (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
                    
                    # # Apply mask overlay
                    # colored_mask = np.zeros_like(display_img)
                    # colored_mask[mask > 0] = color
                    # alpha = 0.3
                    # display_img = ((1 - alpha) * display_img + alpha * colored_mask).astype(np.uint8)
                    
                    # title += f"{mean_depth:.2f}m, {rel_motion:.2f}m, ID: {obj_id}\n"
                # ax.set_title(title.strip())
            # Display frame
            # ax.imshow(display_img)
            # ax.axis('off')
            
            G = create_mixed_graph(image_nodes, mean_depths)
            G_temporal.append(G)


            # # Break on 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
            # # Example: Add new object on 'a' key press
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('a'):
            #     # Add new object at center of frame
            #     h, w = frame_BGR.shape[:2]
            #     new_obj_id, _ = tracker.add_new_object((w//2, h//2), "point")
            #     if new_obj_id:
            #         print(f"Added new object with ID: {new_obj_id}")
    
        # plt.savefig('test.png')
        # plt.close(fig)

    # cv2.destroyAllWindows()
    fig, axes  = plt.subplots(6, 5, figsize=(20, 20))
    ax_iter = iter(axes.flat)

    for i, g in enumerate(G_temporal):
        file = frame_names[1:][i]
        try:
            ax = next(ax_iter)
            visualize_mixed_graph(g, ax=ax)
            ax.set_title(f'{file}', fontsize=12, weight='bold')
        except StopIteration:
            print(f"Warning: More graphs ({len(G_temporal)}) than subplots (12)")
            break

    # Hide unused subplots
    for j in range(i+1, 12):
        axes.flat[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_multi_object_tracking()