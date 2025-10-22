from langgraph.graph import StateGraph, START, END
from langchain_community.tools import BaseTool
from langchain.agents import tool
import networkx as nx
import clip
import torch
from sentence_transformers import SentenceTransformer
import json
import math
from typing import List, Dict
import random
from langchain_ollama import ChatOllama
from rosa import ROSA

import sys
sys.path.extend([
    '/scratch3/kat049/segment-anything-2-real-time',
])
from my_scripts.system_prompt import get_prompt
import pickle
import yaml
import cv2
import numpy as np

def get_llm(streaming: bool = False):
  
    ollama_model = 'qwen2.5:14b'
    ollama_base_url = 'http://localhost:11434/'
 
    ollama_llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0,
        keep_alive="5m",
        streaming=streaming,
        num_ctx=8192,   # Increase context length for better tool understanding
    )
    return ollama_llm

def camera_matrices(path=None, camera='centre-camera'):
    """
    Intrinsics K=[fx 0 cx; 0 fy cy; 0 0 1]
    Distortion Coefficients D (OpenCV order): [k1,k2,p1,p2,k3]
    """
    if path is None:
        path = '/scratch3/kat049/datasets/DARPA/p14_fr/p14-ar1335-2016x1512.yaml'
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The file 'your_file.yaml' was not found.")
    intrinsics = data[camera]['intrinsics']['K']
    distortion = data[camera]['intrinsics']['D']
    return intrinsics, distortion

@tool
def get_3Dfromuv(u: float, v: float, depth:float) -> List:
    """
    Back-projects 2D image coordinates (u, v) and depth into 3D camera coordinates (X, Y, Z).
    Args:
        u (float): The x-coordinate (horizontal) of the image point in pixels.
        v (float): The y-coordinate (vertical) of the image point in pixels.
        depth (float): The depth value at the image point in meters.

    Returns:
        List: A list containing the 3D point in the camera coordinate frame:
            - ["X" (float): X-coordinate in meters,
            - "Y" (float): Y-coordinate in meters,
            - "Z" (float): Z-coordinate (depth) in meters]
    
    """
    
    [fx, fy, cx, cy], [k1, k2, p1, p2, k3] = camera_matrices()
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    D = np.array([k1, k2, p1, p2, k3])
    uv = np.array([[u, v]], dtype=np.float32)
    uv_undistorted = cv2.undistortPoints(uv, K, D, P=K)[0][0] # Undistort point

    u, v = uv_undistorted

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth  

    return [float(X),float(Y),float(Z)]

class ImageGraphAgent():
    def __init__(self):
        tool_dict = dict(
            closest_node= self._create_closest_node(),
            # sample_random_nodes = self._create_sample_random_nodes(),
            distance_from_observer_to_query_node = self._get_distance_from_observer(),
            distance_between_nodes = self._get_distance_between_nodes()
        )
        self.tools = list(tool_dict.values())# + [get_3Dfromuv]

        self.agent = ROSA(
            ros_version=2,
            llm=get_llm(),
            prompts=get_prompt(),
            tools=self.tools
        )

        with open('/scratch3/kat049/segment-anything-2-real-time/my_graph.pickle', 'rb') as file:
            self.G = pickle.load(file)
        nodes_org = self.G.nodes
        self.nodes = [{"node_id": key, "description": data['caption']} 
                for key, data in nodes_org.items()]
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _create_closest_node(self):
        @tool
        def closest_node(query: str) -> str:
            """Search for nodes that match a text description from candidate nodes which are already provided. Candidate nodes is a list, each with format:
                                                        [{"node_id": "node_123", "description": "description"}, ...]
            
            Args:
                query (str): Text description to search for in nodes
                
            Returns:
                str: JSON string containing the most similar node ID string
            """

            
            query_embedding = self.text_model.encode([query.lower()])
            best_sim = -math.inf
            result = None
            
            for node in self.nodes:
                node_id = node["node_id"]
                desc = node["description"]
                desc_embedding = self.text_model.encode([desc.lower()])
                sim = desc_embedding@query_embedding.T
                if sim > best_sim:
                    result = node_id
                    best_sim =sim
            
            return json.dumps(result) 
        return closest_node

    def _create_sample_random_nodes(self):
        @tool
        def sample_random_nodes(sample_size: int = 10) -> str:
            """Randomly sample nodes from all available nodes which are already provided. when there are too  many nodes. 
            Nodes is a list, each with format:[{"node_id": "node_123", "description": "description"}, ...]
            
            Args:
                sample_size (int): Number of nodes to sample (default: 10)
                
            Returns:
                str: JSON string containing list of randomly sampled nodes with format:
                        [{"node_id": "node_123", "description": "description"}, ...]
            """
            node_items = list(self.nodes.items())
            sampled = random.sample(node_items, min(sample_size, len(node_items)))
            
            results = [{"node_id": node_id, "description": desc} for node_id, desc in sampled]
            
            return json.dumps(results)
        return sample_random_nodes

    def _get_distance_from_observer(self):
        @tool
        def distance_from_observer_to_query_node(node_id: str) -> str:
            """Get the distance to a node from the observer/self node.
            Args:
                node_id (str): The unique identifier of the node to query (e.g., "objID_11").
            Returns:
                str: JSON string containing distance from the self node to the query_node as a string
            """
            print("distance_from_observer_to_query_node")
            self_node =  self.nodes[0]['node_id']
            if not node_id in self.G.nodes():
                return f"No such {node_id}"
            return json.dumps(str(self.G[self_node][node_id]['weight']))
            
        return distance_from_observer_to_query_node
    
    def _get_distance_between_nodes(self):
        @tool
        def distance_between_nodes(query_node_1_id: str, query_node_2_id: str) -> str:
            """Get the relative distance between two nodes in the graph. 
            Args:
                query_node_1_id (str): Node if of the node 
                query_node_2_id (str): Node if of the node
                
            Returns:
                str: JSON string containing distance between the two query nodes as a string
            """
            if not query_node_1_id in self.G.nodes() or not query_node_2_id in self.G.nodes():
                return f"No such nodes {query_node_1_id} or {query_node_2_id}"
            self_node = self.nodes[0]['node_id']
            query_node_1_bbox = self.G.nodes[query_node_1_id]['bbox']
            query_node_2_bbox = self.G.nodes[query_node_2_id]['bbox']
            if query_node_1_bbox is None or query_node_2_bbox is None:
                return f"No bounding detected for {query_node_1_id} or {query_node_2_id}"
            
            u_1,v_1 = query_node_1_bbox[0]+query_node_1_bbox[2]/2, query_node_1_bbox[1]+query_node_1_bbox[3]/2
            depth_1 = self.G[self_node][query_node_1_id]['weight']
            u_2,v_2 = query_node_2_bbox[0]+query_node_2_bbox[2]/2, query_node_2_bbox[1]+query_node_2_bbox[3]/2
            depth_2 = self.G[self_node][query_node_2_id]['weight']
            
            pos_1 = get_3Dfromuv.func(u_1, v_1, depth_1)
            pos_2 = get_3Dfromuv.func(u_2, v_2, depth_2)

            distance = np.sqrt(np.square(np.subtract(np.array(pos_1), np.array(pos_2))).sum())
            return json.dumps(str(distance))
            
        return distance_between_nodes
# class ImageGraphProcessor:
#     def __init__(self):
#         self.graph = nx.Graph()
#         self.image_embeddings = {}
#         self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.image_descriptions = {}
    
#     def add_image_node(self, node_id: str, image_path: str, description: str = ""):
#         self.graph.add_node(node_id, image_path=image_path)
#         if description:
#             self.image_descriptions[node_id] = description
    
#     def add_distance_edge(self, node1: str, node2: str, distance: float):
#         self.graph.add_edge(node1, node2, weight=distance)

# # Define tools as functions (easier approach)
# processor = ImageGraphProcessor()

# class SearchImageNodesTool(BaseTool):
#     name = "search_image_nodes"
#     description = "Search for image nodes that match a text description"
    
#     def _run(self, query: str) -> str:
#         query_embedding = processor.text_model.encode([query])
#         results = []
        
#         for node_id, desc in processor.image_descriptions.items():
#             if query.lower() in desc.lower():
#                 results.append({"node_id": node_id, "description": desc})
        
#         return json.dumps(results[:5])  # Return top 5 matches

# class FindShortestPathTool(BaseTool):
#     name = "find_shortest_path"
#     description = "Find shortest path between two nodes"
    
#     def _run(self, start_node: str, end_node: str) -> str:
#         try:
#             path = nx.shortest_path(processor.graph, start_node, end_node, weight='weight')
#             total_distance = nx.shortest_path_length(processor.graph, start_node, end_node, weight='weight')
#             return json.dumps({"path": path, "total_distance": total_distance})
#         except nx.NetworkXNoPath:
#             return json.dumps({"error": "No path found between nodes"})

# class FindNearbyNodesTool(BaseTool):
#     name = "find_nearby_nodes"
#     description = "Find all nodes within a specified distance from a center node"
    
#     def _run(self, center_node: str, max_distance: float) -> str:
#         try:
#             distances = nx.single_source_dijkstra_path_length(
#                 processor.graph, center_node, cutoff=max_distance, weight='weight'
#             )
#             nearby = {node: dist for node, dist in distances.items() if dist <= max_distance}
#             return json.dumps(nearby)
#         except nx.NetworkXError:
#             return json.dumps({"error": f"Node {center_node} not found in graph"})

# # Initialize tools
# tools = [
#     SearchImageNodesTool(),
#     FindShortestPathTool(), 
#     FindNearbyNodesTool()
# ]

img_graph_agent = ImageGraphAgent()
print(img_graph_agent.agent.invoke(f"How far away the horse is from me?"))
print(img_graph_agent.agent.invoke(f"Distance between triangular shape and horse?"))

#The horse is approximately 7.41 meters away from you.
#The distance between the triangular shape and the horse is approximately 10.41 meters.

# edges for correctness and completeness
# slam based artifacts
# query based approach is not a systematic way to approach graph construction correctness
