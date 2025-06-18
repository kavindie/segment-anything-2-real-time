import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
from PIL import Image
import re
import networkx as nx
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# @dataclass
# class GraphNode:
#     id: str
#     node_type: str
#     content: Any  # Image data or text
#     caption: str
#     timestamp: str
#     position: Tuple[float, float]

@dataclass
class GraphNode:
    id: str
    position: Tuple[float, float]
    image: Any  # Your image data
    text: str
    timestamp: str

@dataclass
class GraphState:
    nodes: Dict[str, GraphNode]
    timestamp: str

def get_relative_distance2obj(video_dir, image_file, mask):
    "it is important that there is one video_dir which has RGb and depth frames  termed frame00001.jpg and depth00001.png"
    try:
        depth_frame = Image.open(os.path.join(video_dir, re.sub(r'frame(\d+)\.jpg', r'depth\1.png', image_file))) #image_pil
    except FileNotFoundError:
        print(f"Depth file for {image_file} not found.")
        # TODO convert RGB to depth
        return None

    depth_frame = np.array(depth_frame)

    # masked_depth_image = depth_frame * mask # Shape: (h, w)
    masked_depths = depth_frame[mask]  # Extract depths where mask is True
    # plt.imshow(masked_depth_image)
    # plt.show()
    # plt.imshow(depth_frame)
    # plt.show()
    mean_depth = np.mean(masked_depths)/1000 # Convert to meters
    return mean_depth

# get_relative_distance2obj(frame_names[0], mask)

def load_data_files(rgb_file_path, pose_file_path):
    """
    Load and parse the two data files
    
    Args:
        rgb_file_path: Path to RGB timestamp file
        pose_file_path: Path to pose data file
    
    Returns:
        rgb_df: DataFrame with columns ['timestamp', 'filename']
        pose_df: DataFrame with columns ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    """
    # Load RGB file
    rgb_df = pd.read_csv(rgb_file_path, sep=' ', header=None, 
                        names=['timestamp', 'rgb_path'])
    rgb_df['filename'] = rgb_df['rgb_path'].apply(lambda x: os.path.basename(x))
    
    # Load pose file  
    pose_df = pd.read_csv(pose_file_path, sep=' ', header=None,
                         names=['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
    
    return rgb_df, pose_df

def find_closest_pose(timestamp, pose_df):
    """
    Find the pose entry with closest timestamp
    
    Args:
        timestamp: Target timestamp
        pose_df: DataFrame with pose data
    
    Returns:
        Series with closest pose data
    """
    time_diffs = np.abs(pose_df['timestamp'] - timestamp)
    closest_idx = time_diffs.idxmin()
    return pose_df.loc[closest_idx]

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix using scipy"""
    # Note: scipy uses [x, y, z, w] format
    quat = [qx, qy, qz, qw]
    r = Rotation.from_quat(quat)
    return r.as_matrix()

def calculate_motion_between_frames(frame1_name, frame2_name, rgb_df, pose_df):
    """
    Calculate motion between two frames
    
    Args:
        frame1_name: First frame filename (e.g., 'frame00001.jpg')
        frame2_name: Second frame filename (e.g., 'frame00002.jpg')
        rgb_df: RGB timestamp DataFrame
        pose_df: Pose timestamp DataFrame
    
    Returns:
        dict: Motion information including translation, rotation, and distances
    """
    # Find timestamps for both frames
    frame1_row = rgb_df[rgb_df['filename'] == frame1_name]
    frame2_row = rgb_df[rgb_df['filename'] == frame2_name]
    
    if frame1_row.empty:
        raise ValueError(f"Frame {frame1_name} not found in RGB data")
    if frame2_row.empty:
        raise ValueError(f"Frame {frame2_name} not found in RGB data")
    
    timestamp1 = frame1_row.iloc[0]['timestamp']
    timestamp2 = frame2_row.iloc[0]['timestamp']
    
    # Find closest poses
    pose1 = find_closest_pose(timestamp1, pose_df)
    pose2 = find_closest_pose(timestamp2, pose_df)
    
    # Calculate translation
    pos1 = np.array([pose1['x'], pose1['y'], pose1['z']])
    pos2 = np.array([pose2['x'], pose2['y'], pose2['z']])
    translation = pos2 - pos1
    
    # Calculate rotation
    R1 = quaternion_to_rotation_matrix(pose1['qw'], pose1['qx'], pose1['qy'], pose1['qz'])
    R2 = quaternion_to_rotation_matrix(pose2['qw'], pose2['qx'], pose2['qy'], pose2['qz'])
    
    # Relative rotation from frame1 to frame2
    R_relative = R2 @ R1.T
    
    # Convert back to quaternion and angle-axis
    r_relative = Rotation.from_matrix(R_relative)
    quat_relative = r_relative.as_quat()  # [x, y, z, w]
    angle_axis = r_relative.as_rotvec()
    rotation_angle = np.linalg.norm(angle_axis)  # in radians
    
    # Calculate distances
    euclidean_distance = np.linalg.norm(translation)
    
    # Time differences
    time_diff = timestamp2 - timestamp1
    pose_time_diff1 = abs(timestamp1 - pose1['timestamp'])
    pose_time_diff2 = abs(timestamp2 - pose2['timestamp'])
    
    return {
        'frame1': frame1_name,
        'frame2': frame2_name,
        'timestamp1': timestamp1,
        'timestamp2': timestamp2,
        'time_difference': time_diff,
        'pose_time_diff1': pose_time_diff1,
        'pose_time_diff2': pose_time_diff2,
        'translation': translation,
        'translation_magnitude': euclidean_distance,
        'rotation_matrix': R_relative,
        'rotation_quaternion': quat_relative,  # [x, y, z, w]
        'rotation_angle_rad': rotation_angle,
        'rotation_angle_deg': np.degrees(rotation_angle),
        'position1': pos1,
        'position2': pos2,
        'velocity': translation / time_diff if time_diff != 0 else np.zeros(3),
        'angular_velocity': angle_axis / time_diff if time_diff != 0 else np.zeros(3)
    }

def create_mixed_graph(image_nodes, rel_distances, timestamp=None, captions=None):
    """Create a graph with mixed node types"""
    if timestamp is None:
        timestamp = 0

    if captions is None:
        captions = [f"Object {i}" for i in range(len(image_nodes))]

    G = nx.Graph()
    
    # Add text node at center
    G.add_node("self", node_type="text", content="Observer", caption="Observer position", timestamp=timestamp, position=(0, 0))
    
    # Calculate image sizes and determine spacing
    image_sizes = []
    for image_node in image_nodes:
        if image_node is not None:
            h, w = image_node.shape[:2]
            max_dim = max(h, w)
            image_sizes.append(max_dim)
        else:
            image_sizes.append(100)
    
    # Determine base radius based on largest image
    max_size = max(image_sizes) if image_sizes else 100
    base_radius = max_size * 0.01  # Adjust this multiplier as needed
    min_radius = 2.0  # Minimum distance from center
    
    # Add image nodes with adaptive positioning
    for i, image_node in enumerate(image_nodes):
        angle = 2 * math.pi * i / len(image_nodes)
        
        # Calculate radius based on image size
        img_size = image_sizes[i] if image_sizes else 100
        radius = max(min_radius, base_radius * (img_size / max_size) + min_radius)
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        G.add_node(f"mask{i}", node_type="text" if image_node is None else "mask", content=image_node, caption=captions[i], timestamp=timestamp, position=(x, y))
    
    
    # Add nodes with different types
    # G.add_node("text1", type="text", content="Self", position=(0, 0))
    # for i, image_node in enumerate(image_nodes):
    #     G.add_node(f"mask{i}", type="mask", content=image_node, position=(1, i))
    
    edges = []
    for i in range(len(image_nodes)):
        edges.append(("self", f"mask{i}", {"weight": rel_distances[i]}))

    G.add_edges_from(edges)
    
    return G

def visualize_mixed_graph(G, ax):
    """Visualize graph with text and image nodes"""    
    # Get positions
    position = nx.get_node_attributes(G, 'position')

    # Draw edges with weights
    edges = G.edges(data=True)
    edge_weights = [d.get('weight', 1.0) for u, v, d in edges]
    
    nx.draw_networkx_edges(G, position, ax=ax, edge_color='gray', width=2)
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{d.get('weight', 1.0):.1f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, position, edge_labels, ax=ax, font_size=8)
    
    # Process each node
    for node, (x, y) in position.items():
        node_data = G.nodes[node]
        
        if node_data['node_type'] == 'text':
            # Draw text nodes as boxes
            bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8)
            ax.text(x, y, node_data['content'], ha='center', va='center', 
                   fontsize=12, bbox=bbox, weight='bold')
                   
        elif node_data['node_type'] == 'mask':
            # Draw mask nodes as images
            mask_img = node_data['content']
            # Handle different mask formats
            if len(mask_img.shape) == 2:
                # Grayscale mask [h, w] -> RGB
                mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
            elif mask_img.shape[2] == 3:
                # Already RGB [h, w, 3]
                mask_rgb = mask_img
            else:
                # Handle other cases (like RGBA)
                mask_rgb = mask_img[:, :, :3]
                
            # Create OffsetImage
            imagebox = OffsetImage(mask_rgb, zoom=0.5)
            ab = AnnotationBbox(imagebox, (x, y), frameon=True, pad=0.1)
            ax.add_artist(ab)
            # if you want to add caption to the image
            # ax.text(x, y, node_data['caption'], ha='center', va='center', 
            #        fontsize=6, bbox=bbox, weight='bold')
    
    # ax.set_xlim(-1, 3)
    # ax.set_ylim(-2, 2)
    # ax.set_aspect('equal')
    # ax.set_title('Graph at time', fontsize=14, weight='bold')
    ax.axis('off')
    
    # plt.tight_layout()
    # return ax