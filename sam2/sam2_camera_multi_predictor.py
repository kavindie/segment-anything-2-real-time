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
from tqdm.auto import tqdm

# My imports
import sys
sys.path.extend([
    '/scratch3/kat049/segment-anything-2-real-time',
    '/scratch3/kat049/AutoSeg-SAM2/submodule/segment-anything-1',
    '/scratch3/kat049/AutoSeg-SAM2'
])
from my_scripts.utils_vis import get_masked_image
from my_scripts.utils_vlms import answerwAria
from my_scripts.utils_graph import get_relative_distance2obj, load_data_files, find_closest_pose, calculate_motion_between_frames, create_mixed_graph, visualize_mixed_graph, GraphNode, GraphState
from my_scripts.LLM_planning import SpatioTemporalAgent, ConversationalAgent, ChainOfThoughtAgent
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from auto_mask_batch import masks_update, Prompts

# initialize the depth
rgb_df, pose_df = load_data_files(
    "/scratch3/kat049/VSLAM-LAB-Benchmark/DARPA/p14_fr_2/rgb.txt",
    "/scratch3/kat049/VSLAM-LAB-Benchmark/DARPA/p14_fr_2/groundtruth.txt"
)

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


def get_mask_properties_from_array(mask):
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Find non-zero pixel indices
    non_zero_indices = np.argwhere(binary_mask)

    if non_zero_indices.size == 0:
        return None

    # Get bounding box coordinates
    y_min, x_min = non_zero_indices.min(axis=0)
    y_max, x_max = non_zero_indices.max(axis=0)

    # Compute width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Compute area
    area = int(binary_mask.sum())

    return [float(x_min), float(y_min), float(width), float(height)]

def create_seq_graph(start_idx, video_dir, frame_names, mask_generator, prompts_loader, tracker):
    first_frame = frame_names[0]
    file_prev = first_frame
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
    obj_IDs = list(range(len(masks)))
    for ann_obj_id in tqdm(ann_obj_id_list):
        seg = masks[ann_obj_id]['segmentation']
        prompts_loader.add(ann_obj_id,0,seg)
    bboxes = [masks[i]['bbox'] for i in range(len(masks))]

    def Aria_caption(masked_rgb_cropped):
        """
        Generate caption for the masked image using Aria model.
        Args:
            masked_rgb_cropped: Masked RGB image to generate caption for.
        Returns:
            Generated caption.
        """
        question = """Describe what is shown in the image"""
        return answerwAria(question, masked_rgb_cropped)

    ## Create initialze graph
    G_temporal = []
    connect_Gs = []
    captions = []

    image_nodes = [get_masked_image(frame_RGB, masks[i]['segmentation']) for i in range(len(masks))]
    mean_depths = [get_relative_distance2obj(video_dir, first_frame, masks[i]['segmentation']) for i in range(len(masks))]
    captions = None #[Aria_caption(get_masked_image(frame_RGB, masks[i]['segmentation'])) for i in range(len(masks))]
    G = create_mixed_graph(obj_IDs=obj_IDs, image_nodes=image_nodes, rel_distances=mean_depths, bboxes=bboxes, timestamp=start_idx, captions=captions)
    G_temporal.append(G)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Load first frame
        tracker.load_first_frame(frame_BGR)
        object_prompts = []
        for obj_id, bbox in zip(ann_obj_id_list, bboxes):
            object_prompts.append(((bbox[0], bbox[1], bbox[2], bbox[3]), "bbox", obj_id))

        object_ids = tracker.add_multiple_objects(object_prompts)
        print(f"Added objects with IDs: {object_ids}")
        
        # Main tracking loop
        for i, f in tqdm(enumerate(frame_names[1:])):
            frame = Image.open(os.path.join(video_dir, f)) #image_pil
            frame_RGB = np.array(frame) #image_rgb
            frame_BGR = frame_RGB[:, :, ::-1]  # image

            out_obj_ids, out_mask_logits = tracker.track_frame(frame_BGR)

            rel_motion = np.square(calculate_motion_between_frames(file_prev, f, rgb_df, pose_df)['translation']).sum() ** 0.5
            file_prev = f
            if rel_motion > 0.05: # moved more than 5 cm
                connect_Gs.append(rel_motion)
            else:
                connect_Gs.append(0.0)
            
            obj_IDs = []
            image_nodes = []
            mean_depths = []
            bboxes_nodes = []
            captions_t = []
            # Process and visualize results
            if len(out_obj_ids) > 0:
                # Convert masks to numpy arrays and overlay on frame
                for i, (obj_id, mask_logit) in enumerate(zip(out_obj_ids, out_mask_logits)):
                    # Convert logits to binary mask
                    obj_IDs.append(obj_id)
                    mask = (mask_logit.squeeze() > 0).cpu().numpy()  # Shape: (768, 1024)
                    
                    if not (mask==True).any():
                        print(f"No mask found for object {obj_id}, skipping")
                        mean_depths.append(0)
                        image_nodes.append(None)
                        bboxes_nodes.append(None)
                        captions_t.append("")
                        continue

                    mean_depth = get_relative_distance2obj(video_dir, f, mask)
                    mean_depths.append(mean_depth)

                    masked_rgb_cropped = get_masked_image(frame_RGB, mask)
                    image_nodes.append(masked_rgb_cropped)
                    caption = None #Aria_caption(masked_rgb_cropped)
                    captions_t.append(caption)
                    bboxes_nodes.append(get_mask_properties_from_array(mask))
            
            G = create_mixed_graph(obj_IDs, image_nodes, mean_depths, bboxes_nodes, timestamp=start_idx+i+1, captions=captions_t)
            G_temporal.append(G)
            # captions.append(captions_t)

    return G_temporal, connect_Gs, captions

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
    length_seq = 5 #60

    video_dir = "/scratch3/kat049/datasets/DARPA/p14_fr/results" #args.video_path
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(p[len("frame"):].split('.')[0]))
    
    G_temporal_seq = []
    connect_Gs_seq = []
    captions_seq = []
    for i in range(3):
        end_index = start_idx + vis_gap*length_seq
        frame_names_seq = frame_names[start_idx:end_index:vis_gap]
        G_temporal, connect_Gs, captions = create_seq_graph(start_idx, video_dir, frame_names_seq, mask_generator, prompts_loader, tracker)

        sample_data = {}
        for timestamp in range(len(G_temporal)):
            nodes_org = G_temporal[timestamp].nodes
            nodes = {
                key: GraphNode(
                    id=key,
                    position=nodes_org[key]['position'], 
                    image=nodes_org[key]['content'],
                    text=nodes_org[key]['caption'],
                    timestamp=nodes_org[key]['timestamp'],
                )
                for key in nodes_org.keys()
            }
            sample_data[timestamp] = GraphState(nodes, timestamp)


        # SIMPLE TOOL EXECUTION
        agent = SpatioTemporalAgent(sample_data)
        
        # Get available tools
        print("Available tools:")
        for tool_name, description in agent.get_available_tools().items():
            print(f"- {tool_name}: {description.strip()}")
    
        print("\n" + "="*50 + "\n")
        
        # Answer movement question
        answer = agent.answer_movement_question('mask8')
        print(answer)
        
        # Use individual tools
        print("="*50)
        print("Individual tool usage:")
        
        # Check spatial relationship
        spatial_result = agent.use_tool('spatial_relationship', 
                                    timestamp=3, 
                                    object1_id='mask8', 
                                    object2_id='self')
        print(f"Spatial relationship at 3: {spatial_result}")
        
        # Get trajectory
        trajectory_result = agent.use_tool('trajectory_analysis', object_id='mask8')
        print(f"Trajectory summary: {trajectory_result['total_distance']:.2f} units total distance")
        
        # ADVANCED
        agent = ConversationalAgent(sample_data)
    
        # Example conversation
        questions = [
            "How did mask8 move relative to self?",
            "What's the distance between mask8 and mask0 at 2?",
            # "Compare the positions of mask8 at 1 and 2"
            # "Tell me about the trajectory of mask8"
        ]
        
        print("=== Demo Conversation ===\n")
        
        for i, question in enumerate(questions, 1):
            print(f"Q{i}: {question}")
            answer = agent.chat(question)
            print(f"A{i}: {answer}\n" + "-"*50 + "\n")
        
        print("\n=== Chain of Thought Demo ===\n")
    
        # Demo chain of thought reasoning
        cot_agent = ChainOfThoughtAgent(SpatioTemporalAgent(sample_data))
        
        complex_queries = [
            "Which object moved the fastest?",
            "What object is closest to self?",
            "Which object changed the most?"
        ]
        
        for query in complex_queries:
            print(f"Query: {query}")
            result = cot_agent.solve_complex_query(query)
            print(f"Answer: {result['final_result']}")
            print(f"Reasoning steps: {', '.join(result['reasoning_steps'])}")
            print("-" * 50)



        tracker.reset_tracker()
        start_idx = end_index

        G_temporal_seq.append(G_temporal)
        connect_Gs_seq.append(connect_Gs)
        captions_seq.append(captions)

        ## Visualize the results
        # no_plots = len(frame_names_seq)
        # fig, axes  = plt.subplots(10, int(no_plots/10), figsize=(20, 20))
        # ax_iter = iter(axes.flat)

        # for i, g in enumerate(G_temporal):
        #     file = frame_names_seq[1:][i]
        #     try:
        #         ax = next(ax_iter)
        #         visualize_mixed_graph(g, ax=ax)
        #         ax.set_title(f'{file}', fontsize=12, weight='bold')
        #     except StopIteration:
        #         print(f"Warning: More graphs ({len(G_temporal)}) than subplots (12)")
        #         break

        # # Hide unused subplots
        # for j in range(i+1, no_plots):
        #     axes.flat[j].axis('off')

        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    example_multi_object_tracking()