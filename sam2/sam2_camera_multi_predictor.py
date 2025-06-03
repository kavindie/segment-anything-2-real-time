import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import os
from PIL import Image
from matplotlib import pyplot as plt

def show_mask(mask, ax, obj_id=None, random_color=False):
    "mask of shape (H, W)"
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


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


def example_multi_object_tracking():
    """
    Example of how to use the multi-object tracker
    """
    # Initialize the predictor
    sam2_checkpoint = "/scratch3/kat049/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tracker = MultiObjectSAM2CameraPredictor(model_cfg, sam2_checkpoint, device)

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
    
    # Initialize video capture
    # cap = cv2.VideoCapture("/scratch3/kat049/datasets/DARPA/p14_fr/camera0-1024x768-002.mp4")
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Read first frame
        # ret, frame = cap.read()
        # if not ret:
        #     return
            
        # Load first frame
        tracker.load_first_frame(frame_BGR)
        
        # Add multiple objects to track
        # You can add objects with different prompt types
        object_prompts = [
        #     ((100, 100), "point"),     # Point at (100, 100)
        #     ((200, 150), "point"),     # Another point
        #     ((50, 50, 150, 150), "bbox"),  # Bounding box (x1, y1, x2, y2)
            ((620, 540, 980, 800), "bbox", 0),  # Bounding box (x1, y1, x2, y2)
            ((160, 510, 220, 580), "bbox", 1)  # Bounding box (x1, y1, x2, y2)
        ]

        
        object_ids = tracker.add_multiple_objects(object_prompts)
        print(f"Added objects with IDs: {object_ids}")
        
        # Main tracking loop
        for i, file in enumerate(frame_names[1:]):
            frame = Image.open(os.path.join(video_dir, file)) #image_pil
            frame_RGB = np.array(frame) #image_rgb
            frame_BGR = frame_RGB[:, :, ::-1]  # image
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
                
            # Track all objects in current frame
            out_obj_ids, out_mask_logits = tracker.track_frame(frame_BGR)
            
            # Process and visualize results
            if len(out_obj_ids) > 0:
                # Convert masks to numpy arrays and overlay on frame
                for i, (obj_id, mask_logit) in enumerate(zip(out_obj_ids, out_mask_logits)):
                    # Convert logits to binary mask
                    mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)
                    
                    # Create colored overlay for this object
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                            (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
                    
                    # Apply mask overlay
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 0] = color
                    frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
                    
                    # Add object ID text
                    cv2.putText(frame, f"ID: {obj_id}", 
                              (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, color, 2)
            
            # Display frame
            cv2.imshow('Multi-Object Tracking', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Example: Add new object on 'a' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                # Add new object at center of frame
                h, w = frame.shape[:2]
                new_obj_id, _ = tracker.add_new_object((w//2, h//2), "point")
                if new_obj_id:
                    print(f"Added new object with ID: {new_obj_id}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    example_multi_object_tracking()